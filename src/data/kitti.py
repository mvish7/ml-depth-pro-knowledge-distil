"""
KITTI dataset for monocular depth estimation.
"""

import os
import cv2
import numpy as np
import torch
from typing import Tuple, Dict

from .dataset import BaseDataset
from src.depth_pro.utils import load_rgb


class KITTIDataset(BaseDataset):
    """
    KITTI dataset loader for monocular depth estimation.

    Args:
        root_dir (str): Path to the KITTI dataset root directory
        split (str): Dataset split ('train', 'val', or 'test')
        transform (callable, optional): Optional transform to be applied to images
    """

    def __init__(self, root_dir: str, split: str = "train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Define paths for images and depth maps
        self.images_dir = os.path.join(root_dir, "images")
        self.depth_dir = os.path.join(root_dir, "depth")

        # Load file lists
        split_file = os.path.join(root_dir, f"{split}.txt")
        with open(split_file, 'r') as f:
            self.filenames = [line.strip() for line in f.readlines()]

    def __len__(self) -> int:
        return len(self.filenames)

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
        image_path = os.path.join(self.images_dir, f"{filename}.jpg")
        image = load_rgb(image_path)

        # Load depth map
        depth_path = os.path.join(self.depth_dir, f"{filename}.png")
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype('float32')
        depth = np.array(
            depth, dtype=np.float32) / 256.0  # KITTI depth is stored as uint16

        # Apply transforms if specified
        image = self.image_transform(image)
        depth = self.depth_transform(depth)

        return {'image': image, 'depth': depth, 'valid_mask': depth > 0}

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
