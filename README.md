Additional modification directly on Timm library is not included in this repo: modified vision_transformer.py and patch_embed.py to enable dynamic patch size for FlexiViT training. The main change can be summarized in the two lines of code: 
resized_kernel = F.interpolate(self.proj.weight, size=(ps, ps), mode='bilinear', antialias=True)
x = F.conv2d(x, resized_kernel, bias=self.proj.bias, stride=ps)


## Acknowledgements
This code is based on the these amazing repositories:

CNN Image Retrieval in PyTorch: Training and evaluating CNNs for Image Retrieval in PyTorch by F. Radenović.
Asymmetric metric learning by M. Budnik.

1. [CNN Image Retrieval in PyTorch: Training and evaluating CNNs for Image Retrieval in PyTorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch) by F. Radenović.
2. [Asymmetric metric learning](https://github.com/budnikm/aml) by M. Budnik.
3. [Large-to-small Image Resolution Asymmetry in Deep Metric Learning](https://github.com/pavelsuma/raml/tree/master?tab=readme-ov-file) by P. Suma.
