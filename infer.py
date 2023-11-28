import torch
import torch.nn as nn
from PIL import Image
import subprocess

import argparse
from fvcore.nn import FlopCountAnalysis

from lib.datasets.testdataset import get_testsets
from lib.utils.evaluate import compute_map_and_print
from lib.datasets.datahelpers import fg_eval_transform
from lib.datasets.datahelpers import get_dataset_config, keydefaultdict
from test import find_ranks
from lib.networks.imageretrievalnet import init_network

def get_free_gpu():
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], encoding='utf-8')
        gpu_memory = [int(x) for x in result.strip().split('\n')]
        print(f"gpu_memory: {gpu_memory}")
        most_free_gpu = gpu_memory.index(max(gpu_memory))
        return most_free_gpu
    except Exception as e:
        print(f"Could not execute nvidia-smi: {e}")
        return 0

def load_image(image_path, imsize=240):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    preprocess = fg_eval_transform(imsize, mean=mean, std=std)
    image = Image.open(image_path)
    image = preprocess(image)
    image = image.unsqueeze(0)
    return image

class ModelWrapper(nn.Module):
    def __init__(self, model, ps):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.ps = ps

    def forward(self, img):
        return self.model(img, self.ps)

####################################
############## device ##############
####################################
if torch.cuda.is_available():
    free_gpu = get_free_gpu()
    device = torch.device(f"cuda:{free_gpu}")
else:
    device = torch.device("cpu")

print(f"Infer on {device}")

####################################
########### test dataset ###########
####################################
args = {}
args["data_root"] = "data"
args["test_datasets"] = "cub-test"
args["workers"] = 8
args = argparse.Namespace(**args)
cfgs = keydefaultdict(lambda x: get_dataset_config(args.data_root, x, val_ratio=0.5))
test_dataset_names = args.test_datasets.split(',')
feats = {}
test_datasets = list(get_testsets(test_dataset_names, args, cfgs, feats))

####################################
############## model  ##############
####################################
img_sizes = [480, 360, 240, 120]
patch_sizes = [40, 30, 24, 20, 15, 12, 10, 8]
model_name = 'flexivit_small'
net_params = {
    'architecture': model_name,
    'pooling': 'no_pool',
    'whitening': 0,
    'pretrained': True
}
m = init_network(net_params, device=device)
m.eval()

####################################
########## compute GFLOPS ##########
#################################### 
"""
img_name = test_datasets[0].images[0]
i = 0
gflops = torch.zeros(len(img_sizes), len(patch_sizes))
for ims in img_sizes:
    j = 0
    img = load_image(img_name, imsize=ims)
    for ps in patch_sizes:
        wrapped_model = ModelWrapper(m, ps)
        flops = FlopCountAnalysis(wrapped_model, img)
        flops.unsupported_ops_warnings(False)
        gflops[i, j] = flops.total() / 1e9

        print(f"{i}th row {j}th col finished")
        j += 1
    i += 1

torch.save(gflops, f"gflops_{model_name[9:]}.pt")
"""
####################################
########### compute mAPs ###########
####################################
i = 0
maps = torch.zeros(len(img_sizes), len(patch_sizes))
for ims in img_sizes:
    j = 0
    for ps in patch_sizes:

        qvecs = test_datasets[0].extract_query(m, (1,), 1, ims, ps=ps)
        ranks = find_ranks(qvecs, qvecs)
        sym_mAP = compute_map_and_print(test_datasets[0].name + ' + sym', ranks.numpy(), test_datasets[0].cfg, logger=None)
        maps[i, j] = sym_mAP*100

        print(f"{i}th row {j}th col finished")
        j += 1
    i += 1
torch.save(maps, f"maps_{model_name[9:]}.pt")

