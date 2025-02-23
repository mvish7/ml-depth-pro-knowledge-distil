import matplotlib
matplotlib.use("TkAgg")
import cv2
import matplotlib.pyplot as plt
import numpy as np

def visualize_hypersim_sample(sample):
    """
    Visualizes a HyperSim sample (image, depth, mask) using matplotlib subplots.

    Args:
        sample (dict): A dictionary containing 'image', 'depth', and 'valid_mask' tensors.
                       Assumes tensors are in a format suitable for display (e.g., already on CPU and in a suitable data type).
    """

    image = sample['image'].cpu().numpy().transpose(1,2,0)
    depth = sample['depth'].cpu().numpy().squeeze()
    mask = sample['valid_mask'].cpu().numpy().squeeze()

    # Normalize image for display if necessary
    image_min = image.min()
    image_max = image.max()
    image = (image - image_min) / (image_max - image_min)
    # image = (image + 1) / 2.0  # Assuming normalization to [-1, 1] during preprocessing

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title('Image')
    axes[0].axis('off')

    # Adjust depth visualization for better contrast.  Experiment with these parameters if needed.
    depth_viz = np.clip(depth, 0, np.percentile(depth, 99)) #Clip outliers for better visualization
    depth_viz = (depth_viz - depth_viz.min()) / (depth_viz.max() - depth_viz.min()) #Normalize to 0-1
    im = axes[1].imshow(depth_viz, cmap='viridis') #'viridis' is a good colormap for depth
    axes[1].set_title('Depth')
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1])

    axes[2].imshow(mask*10, cmap='gray')
    axes[2].set_title('Mask')
    axes[2].axis('off')

    plt.show()
    cv2.waitKey(0)

