import torch
import numpy as np
import json
import os
from lib.datasets.datahelpers import get_dataset_config, keydefaultdict
from modelhelpers import load_model
from lib.datasets.testdataset import get_testsets
import timm
import argparse


torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
np.random.seed(1234)

# check map of pretrained resnet50 on cub-test
# args
with open('args.json', 'r') as f:
        args = json.load(f)
args = argparse.Namespace(**args)
#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
args.teacher_path = 'cub-448r-resnet50-512-gem-w'

cfgs = keydefaultdict(lambda x: get_dataset_config(args.data_root, x, val_ratio=0.5))
test_dataset_names = args.test_datasets.split(',')
t_model = None
feats = {}

# dataset
#s_model = load_model(args.data_root, args, 'resnet18', args.student_path)
print(f"I am here")
test_datasets = list(get_testsets(test_dataset_names, args, cfgs, feats))

print(f"I am here")


img_sizes = [896, 672, 448, 336, 224, 168, 112]
patch_sizes = [56, 28, 14, 8, 7, 4, 2, 1]
print(f"creating model...")
m = timm.create_model('flexivit_small', dynamic_img_size=True, pretrained=True, num_classes=0)
m.eval()
print(f"model is done")

i = 0
j = 0
heat_map = torch.zeros(len(patch_sizes), len(img_sizes))
for ps in patch_sizes:
    print(f"starting {i}th row...")
    i = i % len(patch_sizes)
    m.patch_embed.proj.weight = nn.Parameter(resample_patch_embed(m.patch_embed.proj.weight, [ps, ps], interpolation='bilinear', antialias=False))
    for ims in img_sizes:
        j = j % len(img_sizes)

        qvecs = test_datasets[0].extract_query(m, (1,), 1, ims)
        ranks = find_ranks(qvecs, qvecs)
        sym_mAP = compute_map_and_print(test_datasets[0].name + ' + sym', ranks.numpy(), test_datasets[0].cfg, logger=None)
        heat_map[i, j] = sym_mAP*100

        j += 1
    i += 1
    print(f"{i} rows finished")
torch.save(heat_map, 'heat_map.pt')







