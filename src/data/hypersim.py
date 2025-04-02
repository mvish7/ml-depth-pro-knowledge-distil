"""
KITTI dataset for monocular depth estimation.
"""

import os
import random

from PIL import Image

import json
import numpy as np
import torch
from typing import Tuple, Dict
import h5py
from torch import dtype

from torchvision.transforms import (Compose, Resize, ConvertImageDtype, Lambda,
                                    Normalize, ToTensor, Pad)

from src.data.dataset import BaseDataset
from src.data.augmentation_pipeline import DepthAugmentation
from src.utils.visualize import visualize_hypersim_sample


def obtain_planar_depth_values(npy_dist: np.ndarray) -> np.ndarray:
    """
    converts hypersim's depth (i.e. distance from camera center) to planar depth (i.e. distance from image plane)
    credits: https://github.com/apple/ml-hypersim/issues/9#issuecomment-754935697
    """
    int_height, int_width = npy_dist.shape[:2]
    flt_focal = 886.81
    npy_imageplane_x = np.linspace(
        (-0.5 * int_width) + 0.5, (0.5 * int_width) - 0.5,
        int_width).reshape(1, int_width).repeat(int_height,
                                                0).astype(np.float32)[:, :,
                                                                      None]

    npy_imageplane_y = np.linspace(
        (-0.5 * int_height) + 0.5, (0.5 * int_height) - 0.5,
        int_height).reshape(int_height, 1).repeat(int_width,
                                                  1).astype(np.float32)[:, :,
                                                                        None]

    npy_imageplane_z = np.full([int_height, int_width, 1], flt_focal,
                               np.float32)
    npy_imageplane = np.concatenate(
        [npy_imageplane_x, npy_imageplane_y, npy_imageplane_z], 2)

    npy_depth = npy_dist / np.linalg.norm(npy_imageplane, 2, 2) * flt_focal

    return npy_depth


class HypersimDataset(BaseDataset):
    """
    Hypersim dataset loader for monocular depth estimation.

    Args:
        root_dir (str): Path to the KITTI dataset root directory
        split (str): Dataset split ('train', 'val', or 'test')
    """

    def __init__(self,
                 root_dir: str,
                 split: str = "train",
                 device: str = "cpu",
                 precision: dtype = torch.float16):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        # Define paths for images and depth maps

        # Load file lists
        split_file = os.path.join(root_dir, "data_splits", f"{split}.txt")
        with open(split_file, 'r') as f:
            self.filenames = [json.loads(line) for line in f]

        # transforms -- aligned with DepthPro transforms used during inference demo and needs of hypersim dataset
        self.image_transform = Compose([
            ToTensor(),
            Pad([0, 128, 0, 128], padding_mode="reflect"),
            Resize((1536, 1536)),
            # Lambda(lambda x: x.to(device)),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ConvertImageDtype(precision),
        ])

        self.depth_transform = Compose([
            ToTensor(),
            Pad([0, 128, 0, 128], fill=0),
            Resize((1536, 1536)),
            # Lambda(lambda x: x.to(device)),
            ConvertImageDtype(precision),
        ])
        # useful area of the mask is 0 < mask < 10
        # idea is to transform mask in same way as depth
        self.maks_transform = Compose([
            ToTensor(),
            Pad([0, 128, 0, 128], fill=10),
            Resize((1536, 1536)),
            # Lambda(lambda x: x.to(device))
        ])

        self.apply_augmentation = DepthAugmentation()

    def __len__(self) -> int:
        return len(self.filenames)

    def load_image(self, path: str) -> np.ndarray:
        """
        implements image loading
        """
        img_path = os.path.join(self.root_dir, path)
        img_pil = Image.open(img_path)
        img_arr = np.array(img_pil)
        return img_arr

    def load_depth(self, path: str) -> np.ndarray:
        """
        loads depth image and converts depth values to planner depth
        """
        depth_path = os.path.join(self.root_dir, path)
        with h5py.File(depth_path, "r") as df:
            dist_from_cam = df["dataset"][:].astype(np.float32)
            depth_map = obtain_planar_depth_values(dist_from_cam)

        return depth_map

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary containing the RGB image and corresponding depth map.

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            dict: Contains 'image' and 'depth' tensors
        """
        filename = self.filenames[idx]
        # Load RGB image
        image = self.load_image(filename[0])
        # Load depth map
        depth = self.load_depth(filename[1])

        # in hypersim dataset, gt_depth can have nan values. it results into loss being nan, that's why set nan=0  here
        # as gt_depth <= 0 is masked from loss calc
        if np.any(np.isnan(depth)):
            depth = np.where(np.isnan(depth), 0, depth)

        # apply augmentations
        image, depth = self.apply_augmentation(image, depth)
        # create a mask to apply during loss calculations
        mask = np.zeros_like(depth)
        mask[depth > 0] = 1
        # Apply transforms if specified
        # copy to force contiguous memory -- avoid numpy negative stride error caused by np.fliplr
        image = self.image_transform(np.copy(image))
        depth = self.depth_transform(np.copy(depth))
        mask = self.maks_transform(mask).to(torch.bool)

        return {'image': image, 'depth': depth, 'valid_mask': mask}

    def get_sample_shape(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """
        Returns the shape of image and depth samples.

        Returns:
            tuple: ((C,H,W), (1,H,W)) shapes for RGB image and depth map
        """
        sample = self[0]
        return sample['image'].shape, sample['depth'].shape

    @staticmethod
    def get_validation_metrics() -> Dict[str, callable]:
        """
        Returns metrics used for validation.

        Returns:
            dict: Dictionary of metric names and their corresponding functions
        """
        return {
            'rmse':
            lambda pred, target: torch.sqrt(torch.mean((pred - target)**2)),
            'abs_rel':
            lambda pred, target: torch.mean(torch.abs(pred - target) / target),
            'sq_rel':
            lambda pred, target: torch.mean(((pred - target)**2) / target)
        }


if __name__ == "__main__":
    hypersim_data = HypersimDataset(
        root_dir="/media/vishal/datasets/hypersim/",
        split="val",
        device="cuda",
        precision=torch.float32)

    num_samples = 100
    for _ in range(num_samples):
        idx = random.choice(range(hypersim_data.__len__()))
        sample = hypersim_data[idx]
        # visualize_hypersim_sample(sample)
