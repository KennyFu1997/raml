Additional modification directly on Timm library is not included in this repo: modified vision_transformer.py and patch_embed.py to enable dynamic patch size for FlexiViT training. The main change can be summarized in the two lines of code: 
resized_kernel = F.interpolate(self.proj.weight, size=(ps, ps), mode='bilinear', antialias=True)
x = F.conv2d(x, resized_kernel, bias=self.proj.bias, stride=ps)
