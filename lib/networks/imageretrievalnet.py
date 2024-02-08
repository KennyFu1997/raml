import torch
import torch.nn as nn
import torchvision
import timm

from lib.layers.pooling import MAC, SPoC, GeM, GeMmp, RMAC
from lib.layers.normalization import L2N
from lib.datasets.genericdataset import ImagesFromList


VANITY_NAMES = {
    "FlexiViT-L i1k": "gs://big_vision/flexivit/flexivit_l_i1k.npz",
    "FlexiViT-S i1k": "gs://big_vision/flexivit/flexivit_s_i1k.npz",

    "FlexiViT-B i1k 300ep": "gs://big_vision/flexivit/flexivit_b_i1k_300ep.npz",
    "FlexiViT-B i1k 600ep": "gs://big_vision/flexivit/flexivit_b_i1k_600ep.npz",
    "FlexiViT-B i1k 1200ep": "gs://big_vision/flexivit/flexivit_b_i1k.npz",
    "FlexiViT-B i21k 300ep": "gs://big_vision/flexivit/flexivit_b_i21k_300ep.npz",
    "FlexiViT-B i21k 1000ep": "gs://big_vision/flexivit/flexivit_b_i21k_1000ep.npz",

    "ViT-B/16 i21k": "gs://big_vision/flexivit/vit_b16_i21k_300ep.npz",
    "ViT-B/30 i21k": "gs://big_vision/flexivit/vit_b30_i21k_300ep.npz",
}

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
        ps = int(ps)
        if self.meta['architecture'].startswith('flexivit'):
            o = self.norm(self.backbone(x, ps=ps))

            # if whiten exist: pooled features -> whiten -> norm
            if self.whiten is not None:
                o = self.norm(self.whiten(o))
            # (B, 512)
        else:
            o = self.backbone(x)

            # features -> pool -> norm
            o = self.norm(self.pool(o)).squeeze(-1).squeeze(-1)

            # if whiten exist: pooled features -> whiten -> norm
            if self.whiten is not None:
                o = self.norm(self.whiten(o))
            # (B, 512)

        # permute so that it is Dx1 column vector per image (DxN if many images)
        return o.permute(1, 0)

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


def init_network(params, device=None):
    architecture = params.get('architecture', 'flexivit_small') # 决定用什么size + 什么pretrained weights
    pooling = params.get('pooling', 'gem')
    whitening = params.get('whitening', 0)
    mean = params.get('mean', [0.485, 0.456, 0.406])
    std = params.get('std', [0.229, 0.224, 0.225])
    pretrained = params.get('pretrained', True)
    pretrained_source = params.get('pretrained_source', 'i1k')
    ep = params.get('ep', 300)
    version = params.get('version', 'flexivit') # 决定如何训练-评估
    # 只有在distillation的时候True，其他任何时候包括test都是False

    if architecture.startswith('flexivit'):

        model_size = architecture.split("_")[1]
        if model_size == "small":
            arch_full_name = architecture + ".1200ep_" + "in".join(pretrained_source.split("i"))
            pretrained_source = "FlexiViT-" + "S " + pretrained_source
        elif model_size == "base":
            if pretrained_source == 'i1k':
                arch_full_name = architecture + f".{ep}ep_" + "in".join(pretrained_source.split("i"))
                pretrained_source = "FlexiViT-" + "B " + pretrained_source + f" {ep}ep"
            else:
                arch_full_name = architecture + f".{ep}ep_" + "in".join(pretrained_source.split("i"))
                pretrained_source = "FlexiViT-" + "B " + pretrained_source + f" {ep}ep"
        elif model_size == "large":
            arch_full_name = architecture + ".1200ep_" + "in".join(pretrained_source.split("i"))
            pretrained_source = "FlexiViT-" + "L " + pretrained_source
        else:
            #'flexivit_base.patch16_in21k'
            arch_full_name = 'flexivit_base' + f".{model_size}_" + "in".join(pretrained_source.split("i"))
            #"ViT-B/16 i21k"
            pretrained_source = "ViT-B/" + model_size[-2:] + " " + pretrained_source

        backbone = timm.create_model(arch_full_name, patch_size=32, dynamic_img_size=True, pretrained=pretrained, num_classes=0, version=version)

        # [384, 3, 32, 32]
        # [1, 49, 384]
        if pretrained:
            from big_vision.utils import load_params
            state_dict_original = load_params(VANITY_NAMES[pretrained_source])
            # use original 32x32 patch embed and 7x7 pos embed
            with torch.no_grad():
                # [768, 3, 10, 10], [1, 144, 768] | [32, 32, 3, 768], [1, 49, 768]
                backbone.patch_embed.proj.weight.copy_(torch.tensor(state_dict_original['embedding']['kernel'].transpose(3, 2, 0, 1)))
                backbone.pos_embed.copy_(torch.tensor(state_dict_original['pos_embedding']))

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


def extract_vectors(net, images, image_size, transform, bbxs=None, ms=[1], msp=1, print_freq=10, workers=8, batch_size=50, ps=None):
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
        #vecs = torch.zeros(768, len(images))
        for i, input in enumerate(loader):
            input = input.to(net.device)
            if len(ms) == 1 and ms[0] == 1:
                vecs[:, i * batch_size:(i + 1) * batch_size] = extract_ss(net, input, ps=ps)
            else:
                vecs[:, i * batch_size:(i + 1) * batch_size] = extract_ms(net, input, ms, msp)

            if (i+1) % print_freq == 0 or (i+1) == len(loader):
                print('\r>>>> {}/{} done...'.format((i+1), len(loader)), end='')
        print('')

    return vecs

def extract_ss(net, input, ps=None):
    out = net(input, ps)
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
