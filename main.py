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


def get_free_gpu():
    try:
        # Run nvidia-smi command to get GPU information
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], encoding='utf-8')
        # Convert output into list of integers representing free memory on each GPU
        gpu_memory = [int(x) for x in result.strip().split('\n')]
        # Select the GPU with the most free memory
        print(f"gpu_memory: {gpu_memory}")
        most_free_gpu = gpu_memory.index(max(gpu_memory))
        return most_free_gpu
    except Exception as e:
        print(f"Could not execute nvidia-smi: {e}")
        return 0  # Default to the first GPU or CPU if nvidia-smi is not available

def main():
    print(args)
    print(">> Creating directory if it does not exist:\n>> '{}'".format(args.directory))
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    exp_name = list(filter(None, args.directory.split("/")))[-1]
    logger = get_logger(args.logger, args.directory, exp_name)
    logger.log_metadata(args)

    # set cuda visible device
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    if torch.cuda.is_available():
        free_gpu = get_free_gpu()
        device = torch.device(f"cuda:{free_gpu}")
    else:
        device = torch.device("cpu")

    print(f"Training on {device}")

    # set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Loader for each dataset config dictionary
    cfgs = keydefaultdict(lambda x: get_dataset_config(args.data_root, x, val_ratio=0.5))
    test_dataset_names = args.test_datasets.split(',')

    t_model = None
    # Teacher embeddings will not change, try to load or preextract them, test embeddings can always be precomputed
    feats = {}
    if args.mode == 'ts_reg':
        feats = load_features([args.training_dataset, args.training_dataset + '-val'] + test_dataset_names, args, cfgs)
    elif args.mode == 'ts_aug':
        t_d = test_dataset_names if not args.optimize else [args.training_dataset + '-val'] + test_dataset_names
        feats = load_features(t_d, args, cfgs)
        t_model = load_model(args.data_root, args, args.teacher, args.teacher_path, device=device)
        t_model.to(f'cuda:{"1" if len(args.gpu_id.split(",")) > 1 else "0"}')
        t_model.eval()

    # Optionally find the best hyperparameters with optuna
    if args.optimize:
        from optimize import optimize
        optimize(args, cfgs, t_model, feats, logger)

    test_datasets = list(get_testsets(test_dataset_names, args, cfgs, feats))
    s_model = load_model(args.data_root, args, args.student, args.student_path, device=device)
    s_model.to(s_model.device)
    train_split = get_train_splits(args, cfgs, s_model.meta if t_model is None else t_model.meta,
                                   feats, val=False)
    train_split['val'] = test_datasets
    



    logger.set_eval()
    run_tests(test_datasets, None, s_model, args.image_size, args.teacher_image_size, logger=logger, sym=True, asym=args.mode != 'sym')
    logger.eval = False


    _, model = run_train(args, train_split, s_model, t_model, logger)

    logger.set_eval()
    run_tests(test_datasets, None, model, args.image_size, args.teacher_image_size,
              logger=logger, sym=True, asym=args.mode != 'sym')


if __name__ == '__main__':
    args = cli.parse_commandline_args()
    main()


