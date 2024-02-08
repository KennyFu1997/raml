import os
import time
from functools import partial
import torch
import numpy as np

from lib.layers.regularization import DifferentialEntropyRegularization
from lib.datasets.finegraindataset import FineGrainRegressionTS, TuplesFineGrainDataset, AugFineGrainTS
from lib.datasets.testdataset import FineGrainTestset
from lib.datasets.traindataset import mine_hardest
from lib.layers.loss import RegressionLoss, TripletLoss, SecondOrderLoss
from modelhelpers import get_model_optimizer, resume_model, save_checkpoint
from test import run_tests

def sym_train(input, s_model, criterion, train_loader):
    ps = np.random.choice([40, 30, 24, 20, 15, 12, 10])
    s_model.to(s_model.device)
    ni = train_loader.dataset.samples_per_class
    nn = train_loader.dataset.nnum

    s_outputs = s_model(input[0].to(s_model.device, non_blocking=True), ps=ps).t()

    scores = s_outputs @ s_outputs.t()

    target = input[1].to(s_model.device, non_blocking=True)
    pos = mine_hardest(scores, target, target, ni-1, negative=False)
    neg = s_model(torch.flatten(input[2], end_dim=1).to(s_model.device, non_blocking=True), ps=ps).t()
    neg = neg.reshape(target.shape[0], nn, -1)

    return criterion(s_outputs.unsqueeze(dim=1), s_outputs[pos], neg)

def asym_train(input, s_model, criterion, train_loader, t_model=None):
    ps = 40
    s_outputs = s_model(input[0].flatten(end_dim=input[0].dim() - 4).to(s_model.device, non_blocking=True), ps=ps)

    if t_model is not None:
        # use fixed patch size == 15 for teacher
        t_outputs = t_model(input[1].flatten(end_dim=input[1].dim() - 4).to(t_model.device, non_blocking=True), ps=15)
    else:
        t_outputs = input[1].squeeze().t()

    s_outputs = s_outputs.t().reshape(-1, input[0].shape[1], s_outputs.shape[0]) # (4, 8, 512)
    t_outputs = t_outputs.t().reshape(-1, input[1].shape[1], t_outputs.shape[0]) # (4, 8, 512)

    return criterion(s_outputs, t_outputs.to(s_model.device))

def train_epoch(args, datasets, t_model,  train_loader, s_model, criterion, train_fn, optimizer, epoch, scheduler, regularization, update_every):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    s_model.train()
    s_model.apply(set_batchnorm_eval)

    # zero out gradients
    optimizer.zero_grad()
    end = time.time()

    for i, input in enumerate(train_loader):
        loss = train_fn(input, s_model, criterion, train_loader) # 0.4302, 0.7098, 0.4376, 0.5677, 0.8606(逐渐接近0)
        #loss += regularization(batch_embeddings)
        losses.update(loss.item()/input[0].shape[0])
        loss.backward()

        if (i + 1) % update_every == 0:
            optimizer.step()
            optimizer.zero_grad()
            if type(scheduler) in [torch.optim.lr_scheduler.OneCycleLR, torch.optim.lr_scheduler.CyclicLR, torch.optim.lr_scheduler.CosineAnnealingLR]:
                scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % train_loader.dataset.print_freq == 0 or i == 0 or (i + 1) == len(train_loader):
            #run_tests(datasets['val'], t_model, s_model, args.image_size, args.teacher_image_size, 
                                                  #logger=None, sym=True, asym=args.mode != 'sym')
            out = '>> Train: [{0}][{1}/{2}]\t Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t \
                   Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch + 1, i + 1, len(train_loader), batch_time=batch_time, loss=losses)
            print(out)

    if type(scheduler) == torch.optim.lr_scheduler.ExponentialLR:
        scheduler.step()
    return {'train/student_loss': losses.avg}


def run_train(args, datasets, s_model, t_model, logger, trial=None):
    s_model.to(s_model.device)
    train_loader = torch.utils.data.DataLoader(
            datasets['train'], batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False, sampler=None,
            drop_last=True if args.mode == 'sym' else False
        )

    # define loss function (criterion)
    if args.mode == 'sym':
        s_model.distill_flag = False
        criterion = TripletLoss(margin=args.loss_margin).to(s_model.device)
        train_fn = sym_train
    elif args.mode == 'ts_reg':
        s_model.distill_flag = True
        criterion = RegressionLoss().to(s_model.device)
        train_fn = asym_train
    elif args.mode == 'ts_aug':
        s_model.distill_flag = True
        t_model.distill_flag = True
        criterion = SecondOrderLoss(args.lam_1, args.lam_2).to(s_model.device) # loss 送到 student 的 device
        t_model.to(t_model.device)
        train_fn = partial(asym_train, t_model=t_model)
    else:
        raise(RuntimeError("Invalid training mode {}!".format(args.mode)))

    optimizer, scheduler = get_model_optimizer(args, s_model, datasets['train'])

    # regularization
    regularization = DifferentialEntropyRegularization(weight=0.7).to(s_model.device)

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume_student:
        s_model, optimizer, scheduler, start_epoch = resume_model(s_model, optimizer, scheduler,
                                                                os.path.join(args.directory, args.resume_student))
        logger.set_epoch(start_epoch)
    
    losses = []
    maps = []
    best_val_map = 0.
    for epoch in range(start_epoch, args.epochs):
        # set manual seeds per epoch
        np.random.seed(args.seed + epoch)
        torch.manual_seed(args.seed + epoch)
        torch.cuda.manual_seed_all(args.seed + epoch)

        # train for one epoch on train set
        train_loader.dataset.create_epoch_tuples(s_model)
        loss = train_epoch(args, datasets, t_model, train_loader, s_model, criterion, train_fn, optimizer, epoch, scheduler, regularization, args.update_every)
        
        losses.append(loss['train/student_loss'])
        logger.set_epoch(epoch + 1)
        logger.log_scalars(loss)

        # evaluate on validation set
        if args.val and (epoch + 1) % args.val_freq == 0:
            val_map = run_tests(datasets['val'], t_model, s_model, args.image_size, args.teacher_image_size, 
                                                  logger=logger, sym=True, asym=args.mode != 'sym')

            maps.append(val_map)
            # Optimization part
            #if trial is not None:
                #trial.report(val_map, epoch)

            #elif (epoch + 1) % args.save_freq == 0:
            if val_map > best_val_map:
                save_checkpoint(s_model, optimizer, scheduler, val_map, epoch, args.directory)
                best_val_map = val_map

        np.save(f"{args.directory}/losses.npy", np.array(losses))
        np.save(f"{args.directory}/maps.npy", np.array(maps))

    return s_model


def get_train_splits(args, cfgs, net_meta, feats=None, val=True):
    splits = {'train': None, 'val': None}
    val_name = args.training_dataset + '-val'
    if not val:
        cfgs[args.training_dataset]['train'].extend(cfgs[val_name]['val'])
        if feats.get(args.training_dataset) is not None:
            feats[args.training_dataset] = torch.cat((feats.get(args.training_dataset), feats.get(val_name)), dim=1)

    if args.mode == 'ts_reg':
        tr_dataset = partial(FineGrainRegressionTS, teacher_feat=feats.get(args.training_dataset))
    elif args.mode == 'ts_aug':
        tr_dataset = partial(AugFineGrainTS, teacher_imsize=args.teacher_image_size)
    elif args.mode == 'sym':
        tr_dataset = partial(TuplesFineGrainDataset, spc=4)

    val_dataset = partial(FineGrainTestset, cfgs[args.training_dataset]['val'], args.training_dataset + '-val')

    splits['train'] = tr_dataset(
        name=args.training_dataset,
        cfg=cfgs[args.training_dataset],
        imsize=args.image_size,
        mean=net_meta['mean'],
        std=net_meta['std'],
        nnum=args.neg_num,
        pnum=args.pos_num,
        qsize=args.query_size,
        poolsize=args.pool_size,
        batch_size=args.batch_size,
        num_workers=args.workers
    )

    if val:
        splits['val'] = [val_dataset(
            teacher_feat=feats.get(val_name),
            num_workers=args.workers
        )]

    return splits


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_batchnorm_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        # freeze running mean and std:
        # we do training one image at a time
        # so the statistics would not be per batch
        # hence we choose freezing (ie using imagenet statistics)
        m.eval()
        # # freeze parameters:
        # # in fact no need to freeze scale and bias
        # # they can be learned
        # # that is why next two lines are commented
        # for p in m.parameters():
        # p.requires_grad = False
