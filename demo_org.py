import json
import argparse
from lib.datasets.datahelpers import get_dataset_config, keydefaultdict
from lib.datasets.testdataset import get_testsets
from lib.datasets.datahelpers import fg_eval_transform
from PIL import Image


import importlib
import jax.numpy as jnp
import flax
import jax
from big_vision.models.proj.flexi.vit import resample_patchemb




# cub set
#torch.manual_seed(1234)
#torch.cuda.manual_seed_all(1234)
#np.random.seed(1234)
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open('args.json', 'r') as f:
    args = json.load(f)
args = argparse.Namespace(**args)
cfgs = keydefaultdict(lambda x: get_dataset_config(args.data_root, x, val_ratio=0.5))
test_dataset_names = args.test_datasets.split(',')
feats = {}
#s_model = load_model(args.data_root, args, 'resnet18', args.student_path)
test_datasets = list(get_testsets(test_dataset_names, args, cfgs, feats))



# model
imsize = 240
rng = jax.random.PRNGKey(0)
init_file = 'gs://big_vision/flexivit/flexivit_s_i1k.npz'
model_mod = importlib.import_module(f"big_vision.models.proj.flexi.vit")
model = model_mod.Model(num_classes=1000, pool_type='tok', width=384, num_heads=6)
dummy_input = jnp.ones((1, imsize, imsize, 3))
params_init = flax.core.unfreeze(model.init(rng, dummy_input))["params"]
model_cfg = dict(
      variant='S',
      pool_type='tok',
      posemb='learn',
      patch_size=(32, 32),
      posemb_size=(7, 7),
      seqhw=None,
  )
params_pretrained = model_mod.load(params_init, init_file, model_cfg)


# default preprocess
def load_image(image_path, imsize=240):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    preprocess = fg_eval_transform(imsize, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    image = Image.open(image_path)
    image = preprocess(image)
    image = image.unsqueeze(0)
    return image

img = jnp.array(load_image(test_datasets[0].images[0]).permute(0, 2, 3, 1).numpy())
x, out = model.apply({"params": params_pretrained}, img, seqhw=32)
print(x.shape)
print(out.keys())
print(jax.devices())
print(jax.__version__)







