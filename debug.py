import torch
import timm

m = timm.create_model("flexivit_small", pretrained=True, num_classes=0)