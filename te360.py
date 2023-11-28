import torch
import numpy as np
from PIL import Image
from torch.nn.parameter import Parameter
import subprocess

import json
import argparse
from fvcore.nn import FlopCountAnalysis

from lib.datasets.testdataset import get_testsets
from lib.utils.evaluate import compute_map_and_print
from lib.datasets.datahelpers import fg_eval_transform
from lib.datasets.datahelpers import get_dataset_config, keydefaultdict
from test import find_ranks

import timm
from timm.layers import resample_patch_embed

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

def load_image(image_path, imsize=240):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    preprocess = fg_eval_transform(imsize, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    image = Image.open(image_path)
    image = preprocess(image)
    image = image.unsqueeze(0)
    return image

####################################
############## seed ################
####################################
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
np.random.seed(1234)
#device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    free_gpu = get_free_gpu()
    device = torch.device(f"cuda:{free_gpu}")
else:
    device = torch.device("cpu")

print(f"Infer on {device}")

####################################
########### tes dataset ############
####################################
with open('args.json', 'r') as f:
    args = json.load(f)
args = argparse.Namespace(**args)
cfgs = keydefaultdict(lambda x: get_dataset_config(args.data_root, x, val_ratio=0.5))
test_dataset_names = args.test_datasets.split(',')
feats = {}
test_datasets = list(get_testsets(test_dataset_names, args, cfgs, feats))

####################################
########### compute maps ###########
####################################
img_sizes = [360]
patch_sizes = [40, 30, 24, 20, 15, 12, 10, 8]
model_name = 'flexivit_large'


i = 0
maps = torch.zeros(len(img_sizes), len(patch_sizes))
for ims in img_sizes:
    j = 0

    # model
    #m = timm.create_model('flexivit_small', img_size=240, patch_size=30, dynamic_img_size=True, pretrained=True, num_classes=0)
    #m.patch_embed_tmp.proj.weight = Parameter(m.patch_embed.proj.weight.clone())
    #m.patch_embed_tmp.proj.bias = Parameter(m.patch_embed.proj.bias.clone())
    #m.to(device)
    #m.eval()
    #print(f"device: {device}")

    for ps in patch_sizes:

        m = timm.create_model(model_name, img_size=ims, patch_size=ps, dynamic_img_size=True, pretrained=True, num_classes=0)
        #m.to(device)
        m.eval()

        qvecs = test_datasets[0].extract_query(m, (1,), 1, ims, ps=ps)
        ranks = find_ranks(qvecs, qvecs)
        sym_mAP = compute_map_and_print(test_datasets[0].name + ' + sym', ranks.numpy(), test_datasets[0].cfg, logger=None)
        maps[i, j] = sym_mAP*100

        j += 1
        print(f"{i}th row {j}th col finished")
        #del m
    #torch.save(maps, f"maps_{model_name[9:]}.pt")
    i += 1
torch.save(maps, f"maps360_{model_name[9:]}.pt")


####################################
########## compute GFLOPS ##########
####################################
img_name = test_datasets[0].images[0]
i = 0
gflops = torch.zeros(len(img_sizes), len(patch_sizes))
for ims in img_sizes:
    j = 0

    for ps in patch_sizes:

        m = timm.create_model(model_name, img_size=ims, patch_size=ps, dynamic_img_size=True, pretrained=True, num_classes=0)
        #m.to(device)
        m.eval()

        img = load_image(img_name, imsize=ims)
        flops = FlopCountAnalysis(m, img)
        flops.unsupported_ops_warnings(False)
        gflops[i, j] = flops.total() / 1e9

        j += 1
        print(f"{i}th row {j}th col finished")
        #del m
    i += 1

torch.save(gflops, f"gflops360_{model_name[9:]}.pt")