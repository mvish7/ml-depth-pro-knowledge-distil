# Choice of dataset: 
This project uses [Hypersim dataset](https://github.com/apple/ml-hypersim). There are two main reasons behind choosing this dataset:
* DepthPro uses hypersim during training, hence it obviously becomes strong candidate for knowledge distilation-
* Hypersim being a synthetic dataset provides accurate ground truth depth. This is especially important because DepthPro only accepts
images in `1536x1536` resolution and image/depth resizing is necessary for most datasets. As hypersim depth annotations are pixel-accurate
 it introduces less of artifacts when resizing as compared to e.g. DIODE dataset.

# Augmentation pipeline: 
Data augmentation pipeline is made to mimic the DepthPro augmentation pipeline as described in the appendix of the paper. 

# Choosing preview images:
Hypersim provides images in `.hdf5` format. One has to apply tone mapping to make use of these images. The dataset also provides tone mapping
operation guidelines. As the preview images (i.e. images after applying tone mapping) were available, I have choosen to use them directly. 
Only benefit of applying tone mapping yourself is that one can augment images by tweaking tone mapping. For further info, please refer to this [issue](https://github.com/apple/ml-hypersim/issues/45)

# Generating planar depth values:
By default, the `depth_meteres` value in hypersim dataset is distance of each point from the camera center. To use these values in training, on must convert them to planar depth format i.e. depth map having distance from image plane.
Please refer to this [issue](https://github.com/apple/ml-hypersim/issues/9) for detailed clarification.

