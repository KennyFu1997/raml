import os
import numpy as np
import torch
import torch.optim
import torch.utils.data
import subprocess

from lib.datasets.datahelpers import get_dataset_config, keydefaultdict
from lib.datasets.testdataset import get_testsets
from lib import cli
from modelhelpers import load_model
from test import run_tests
from train import get_train_splits, run_train
from logger import get_logger
from extract_features import load_features
from timm.layers import resample_patch_embed, resample_abs_pos_embed


def get_free_gpus():
    try:
        # Run nvidia-smi command to get GPU information
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], encoding='utf-8')
        # Convert output into list of integers representing free memory on each GPU
        gpu_memory = [int(x) for x in result.strip().split('\n')]
        print(f"gpu_memory: {gpu_memory}")

        # Sort GPUs by free memory and get indices of the top two
        sorted_gpus = sorted(range(len(gpu_memory)), key=lambda k: gpu_memory[k], reverse=True)
        top_two_free_gpus = sorted_gpus[:2]
        return top_two_free_gpus
    except Exception as e:
        print(f"Could not execute nvidia-smi: {e}")
        return [0, 1] 

def main():
    print(args)
    print(">> Creating directory if it does not exist:\n>> '{}'".format(args.directory))
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    ####################################
    ########## logger/device ###########
    ####################################
    exp_name = list(filter(None, args.directory.split("/")))[-1]
    logger = get_logger(args.logger, args.directory, exp_name)
    logger.log_metadata(args)
    # 创建一个logger object，writer指定为tnsorboard/exp，刚开始log一些meta data

    devices = []
    free_gpus = get_free_gpus()
    # t_model and s_model on the same device

    #free_gpus[0] = 0
    free_gpus[1] = free_gpus[0]

    for free_gpu in free_gpus:
        if free_gpu == 4:
            device = torch.device(f"cuda:3")
        elif free_gpu == 3:
            device = torch.device(f"cpu")
        else:
            device = torch.device(f"cuda:{free_gpu}")
        devices.append(device)

    if free_gpus[0] == 4:
        print(f"Teacher on cuda:4")
    elif free_gpus[0] == 3:
        print(f"Teacher on cpu")
    else:
        print(f"Teacher on {devices[0]}")

    if free_gpus[1] == 4:
        print(f"Student on cuda:4")
    elif free_gpus[1] == 3:
        print(f"Student on cpu")
    else:
        print(f"Student on {devices[1]}")

    ####################################
    ############### seed ###############
    ####################################
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Loader for each dataset config dictionary
    cfgs = keydefaultdict(lambda x: get_dataset_config(args.data_root, x, val_ratio=0.5))
    test_dataset_names = args.test_datasets.split(',')
    t_model = None
    feats = {}
    if args.mode == 'ts_reg':
        feats, t_model = load_features([args.training_dataset, args.training_dataset + '-val'] + test_dataset_names, args, cfgs, device=devices[0], ps=15)
    elif args.mode == 'ts_aug':
        t_d = test_dataset_names if not args.optimize else [args.training_dataset + '-val'] + test_dataset_names
        t_model = load_model(args.data_root, args, args.teacher, args.teacher_path, device=devices[0])

    # Optionally find the best hyperparameters with optuna
    if args.optimize:
        from optimize import optimize
        optimize(args, cfgs, t_model, feats, logger, device=devices[1])

    ####################################
    ############# datasets #############
    ####################################
    test_datasets = list(get_testsets(test_dataset_names, args, cfgs, feats))
    # test_datasets暂时不保存transform，只是在extract vectors时候传递，已经修改mean 和std
    s_model = load_model(args.data_root, args, args.student, args.student_path, device=devices[1])
    if args.mode == "ts_aug" or args.mode == "ts_reg":
        s_model.load_state_dict(t_model.state_dict())
    train_split = get_train_splits(args, cfgs, s_model.meta if t_model is None else t_model.meta,
                                   feats, val=False)
    # train_split 里面已经保存了修改的mean和std(根据s_model来的)
    train_split['val'] = test_datasets
    

    ####################################
    ##### evaluate at the begining #####
    ####################################
    run_tests(test_datasets, t_model, s_model, args.image_size, args.teacher_image_size, logger=logger, sym=True, asym=args.mode != 'sym')

    ####################################
    ############## train ###############
    ####################################
    run_train(args, train_split, s_model, t_model, logger)


if __name__ == '__main__':
    args = cli.parse_commandline_args()
    main()


