#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#from torch.utils.model_zoo import load_url
from lib.networks.imageretrievalnet import ImageRetrievalNet, init_network
import torchvision
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.nn.parameter import Parameter
from lib.datasets.datahelpers import get_dataset_config, keydefaultdict
import numpy as np
import itertools as it
from lib.datasets.genericdataset import ImagesFromList
from lib.datasets.testdataset import get_testsets
import os
from lib.datasets.datahelpers import give_cub_datasets, fg_eval_transform
from lib.datasets.traindataset import RetrievalDataset
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from fvcore.nn import FlopCountAnalysis
from copy import deepcopy
from adjustText import adjust_text

from lib.layers.pooling import MAC, SPoC, GeM, GeMmp, RMAC
from lib.layers.normalization import L2N

from lib.utils.evaluate import compute_map_and_print
from modelhelpers import load_model
import json
import argparse
from train import get_train_splits
from test import find_ranks
import timm

from timm.layers import resample_patch_embed, resample_abs_pos_embed

from big_vision.utils import load_params
from big_vision.models.proj.flexi.vit import resample_patchemb


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




