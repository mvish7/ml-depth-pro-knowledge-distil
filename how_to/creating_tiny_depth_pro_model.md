# Need of creating a smaller DepthPro model
The original version of depth pro has in all 951.991 millions parameters. For a foundational model like, DepthPro such 
a huge parameter count is acceptable. As this repo wants to build a depth estimator for real-time use cases, we need to 
create a smaller model.

# Basic building blocks of DepthPro
The key component of DepthPro is `vit_large` model, which typically has ~300 millions parameters. This same model is 
being used as `patch_encoder, image encoder and fov_encoder.` The decoder of DepthPro is a convolutional model who's 
size depends upon encoder's feature dimensions. 

# How tiny DepthPro is being created?
The file `src/depth_pro/network/vit_factory.py` defines the configs to create the original model. The original configs 
were adapted by replacing `vit_large` by `vit_small`. The embedding dimensions are also modified as per network 
definition from `timm` (as `timm` is being used to define the models).\

Below table shows parameter counts in original and modified tiny DepthPro models. 



| Blocks                | Original Depth Pro | Tiny Depth Pro |
|-----------------------|--------------------|----------------|
| Patch encoder         | 303.73             | 21.8           |
| Image encoder         | 303.73             | 21.8           |
| Fusion + Upsampling   | 20                 | ~3.0           |
| Encoder Total         | 647.66             | 46.6           |
| Decoder               |19.66|2.7|
| Depth Prediction head |0.397|~0.040|
|FOV Prediction|304.25|21.8|
|total|951.99|~72.5|
