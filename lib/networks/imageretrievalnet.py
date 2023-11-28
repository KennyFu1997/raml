import os

import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision
import timm
from timm.layers import resample_patch_embed
from timm.layers import PatchEmbed
from timm.models._manipulate import checkpoint_seq

from lib.layers.pooling import MAC, SPoC, GeM, GeMmp, RMAC
from lib.layers.normalization import L2N
from lib.datasets.genericdataset import ImagesFromList

# possible global pooling layers
POOLING = {
    'mac'    : MAC,
    'spoc'   : SPoC,
    'gem'    : GeM,
    'gemmp'  : GeMmp,
    'rmac'   : RMAC,
    'no_pool': nn.Identity
}

class ImageRetrievalNet(nn.Module):
    
    def __init__(self, backbone, pool, whiten, meta, device=None):
        super(ImageRetrievalNet, self).__init__()
        self.backbone = backbone
        self.pool = pool # Identity
        self.whiten = whiten # None
        self.norm = L2N()
        self.meta = meta # Identity / 0
        self.device = device
    
    def forward(self, x, ps=None):
        #print(f"patch size is: {ps}")
        #print(f"x.shape: {x.shape}")
        if self.meta['architecture'].startswith('flexivit'):
            # PI resize patch emb
            ps = int(ps)
            if ps != self.backbone.patch_embed.proj.weight.shape[-1]:
                self.backbone.patch_embed_tmp.proj.weight.detach().data = resample_patch_embed(self.backbone.patch_embed.proj.weight.detach().data, [ps, ps], interpolation='bilinear', antialias=False)
                self.backbone.patch_embed_tmp.proj.stride = (ps, ps)
            else:
                self.backbone.patch_embed_tmp.proj.weight.detach().data = self.backbone.patch_embed.proj.weight.detach().data#.clone()
                self.backbone.patch_embed_tmp.proj.stride = (self.backbone.patch_embed.proj.weight.shape[-2], self.backbone.patch_embed.proj.weight.shape[-1])

            #print(f"self.backbone.patch_embed_tmp.proj.weight.device: {self.backbone.patch_embed_tmp.proj.weight.device}")
            #print(f"x.device: {x.device}")

            x = self.backbone.patch_embed_tmp(x)
            # (B, h, w, D)
            x = self.backbone._pos_embed(x)
            x = self.backbone.patch_drop(x)
            x = self.backbone.norm_pre(x)
            if self.backbone.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint_seq(self.backbone.blocks, x)
            else:
                x = self.backbone.blocks(x)
            x = self.backbone.norm(x)

            o = self.backbone.forward_head(x)
            # (B, 384)

            # features -> norm
            o = self.norm(o)

            # if whiten exist: pooled features -> whiten -> norm
            """
            if self.whiten is not None:
                o = self.norm(self.whiten(o))
            """
            # (B, 512)
        else:
            o = self.backbone(x)
            # (B, 384)

            # features -> pool -> norm
            o = self.norm(self.pool(o)).squeeze(-1).squeeze(-1)

            # if whiten exist: pooled features -> whiten -> norm
            if self.whiten is not None:
                o = self.norm(self.whiten(o))
            # (B, 512)

        # permute so that it is Dx1 column vector per image (DxN if many images)
        return o.permute(1, 0)

    """
    @property
    def device(self):
        tmp = self.backbone if type(self.backbone) == nn.Sequential else self.backbone.blocks
        return tmp[0].weight.device

    def __repr__(self):
        tmpstr = super(ImageRetrievalNet, self).__repr__()[:-1]
        tmpstr += self.meta_repr()
        tmpstr = tmpstr + ')'
        return tmpstr

    def meta_repr(self):
        tmpstr = '  (' + 'meta' + '): dict( \n' # + self.meta.__repr__() + '\n'
        tmpstr += '     architecture: {}\n'.format(self.meta['architecture'])
        tmpstr += '     pooling: {}\n'.format(self.meta['pooling'])
        tmpstr += '     whitening: {}\n'.format(self.meta['whitening'])
        tmpstr += '     outputdim: {}\n'.format(self.meta['outputdim'])
        tmpstr += '     mean: {}\n'.format(self.meta['mean'])
        tmpstr += '     std: {}\n'.format(self.meta['std'])
        tmpstr = tmpstr + '  )\n'
        return tmpstr
    """


def init_network(params, device=None):
    architecture = params.get('architecture', 'flexivit_small')
    pooling = params.get('pooling', 'gem')
    whitening = params.get('whitening', 0)
    mean = params.get('mean', [0.485, 0.456, 0.406])
    std = params.get('std', [0.229, 0.224, 0.225])
    pretrained = params.get('pretrained', True)

    # load network from torchvision or timm and remove any classifier/pooling
    if architecture.startswith('flexivit'):
        backbone = timm.create_model(architecture, dynamic_img_size=True, pretrained=True, num_classes=0)
        backbone.patch_embed_tmp = deepcopy(backbone.patch_embed)
        """
        embed_args = {}
        embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        backbone.patch_embed_tmp = PatchEmbed(
            img_size=240,
            patch_size=16,
            in_chans=3,
            embed_dim=384,
            bias=True,
            dynamic_img_pad=False,
            **embed_args,
        )
        """


        #backbone.pos_embed_tmp = deepcopy(backbone.pos_embed)
        #backbone.patch_embed_tmp.proj.weight.data = backbone.patch_embed.proj.weight.data.clone()
        #backbone.patch_embed_tmp.proj.bias.data = backbone.patch_embed.proj.bias.data.clone()
        mean = backbone.pretrained_cfg['mean']
        std = backbone.pretrained_cfg['std']
        dim = backbone.num_features

    else:
        if hasattr(torchvision.models, architecture):
            backbone = getattr(torchvision.models, architecture)(pretrained=pretrained)
            if hasattr(backbone, 'features'):
                dim = backbone.classifier[-1].in_features
                backbone = backbone.features
            else:
                backbone = list(backbone.children())
                dim = backbone[-1].in_features
                while any(x in type(backbone[-1]).__name__.lower() for x in ('pool', 'linear')):
                    backbone.pop()
                backbone = nn.Sequential(*backbone)

        elif architecture in timm.list_models():
            backbone = timm.create_model(architecture, pretrained=pretrained, num_classes=0, global_pool=None)
            mean = backbone.pretrained_cfg['mean']
            std = backbone.pretrained_cfg['std']
            dim = backbone.num_features
        else:
            raise ValueError('Architecture not found in torchvision neither timm!')

    # initialize pooling
    if pooling == 'gemmp':
        pool = POOLING[pooling](mp=dim)
    else:
        pool = POOLING[pooling]()


    # initialize whitening layer
    if whitening:
        whiten = nn.Linear(dim, whitening, bias=True)
        dim = whitening
    else:
        whiten = None

    # create meta information to be stored in the network
    meta = {
        'architecture' : architecture,
        'pooling' : pool,
        'whitening' : whitening,
        'mean' : mean,
        'std' : std,
        'outputdim' : dim,
    }

    # create a generic image retrieval network
    net = ImageRetrievalNet(backbone, pool, whiten, meta, device=device)
    return net

# 1. evaluate on a fixed patch size (15)
def extract_vectors(net, images, image_size, transform, bbxs=None, ms=[1], msp=1, print_freq=10, workers=8, batch_size=50, ps=None):
    # moving network to gpu and eval mode
    #device = torch.device('cuda:3')
    #net.to(device)
    net.to(net.device)
    net.eval()

    # creating dataset loader
    loader = torch.utils.data.DataLoader(
        ImagesFromList(root='', images=images, imsize=image_size, bbxs=bbxs, transform=transform),
        batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, drop_last=False
    )

    # extracting vectors
    with torch.no_grad():
        vecs = torch.zeros(net.meta['outputdim'], len(images))
        #vecs = torch.zeros(1024, len(images))
        for i, input in enumerate(loader):
            input = input.to(net.device)
            #ps = np.random.choice([48, 40, 30, 24, 20, 15, 12, 10, 8])
            #ps = 15
            if len(ms) == 1 and ms[0] == 1:
                vecs[:, i * batch_size:(i + 1) * batch_size] = extract_ss(net, input, ps=ps)
            else:
                vecs[:, i * batch_size:(i + 1) * batch_size] = extract_ms(net, input, ms, msp)

            if (i+1) % print_freq == 0 or (i+1) == len(loader):
                print('\r>>>> {}/{} done...'.format((i+1), len(loader)), end='')
        print('')

    return vecs

def extract_ss(net, input, ps=None):
    #ps = np.random.choice([40, 30, 24, 20, 15, 12, 10, 8])
    out = net(input, ps)
    #out /= (out.norm(dim=1, keepdim=True) + 1e-6)
    return out.cpu().data

def extract_ms(net, input, ms, msp):
    
    v = torch.zeros(net.meta['outputdim'])
    
    for s in ms: 
        if s == 1:
            input_t = input.clone()
        else:    
            input_t = nn.functional.interpolate(input, scale_factor=s, mode='bilinear', align_corners=False)
        v += net(input_t).pow(msp).cpu().data.squeeze()
        
    v /= len(ms)
    v = v.pow(1./msp)
    v /= v.norm()

    return v
