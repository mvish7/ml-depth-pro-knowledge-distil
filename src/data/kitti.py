"""
KITTI dataset for monocular depth estimation.
"""

import os
import numpy as np
from PIL import Image
import torch
from typing import Tuple, Dict
from .dataset import BaseDataset


class KITTIDataset(BaseDataset):
    """
    KITTI dataset loader for monocular depth estimation.
    
    Args:
        root_dir (str): Path to the KITTI dataset root directory
        split (str): Dataset split ('train', 'val', or 'test')
        transform (callable, optional): Optional transform to be applied to images
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform = None
    ):
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
        image = Image.open(image_path).convert('RGB')
        
        # Load depth map
        depth_path = os.path.join(self.depth_dir, f"{filename}.png")
        depth = Image.open(depth_path)
        
        # Convert to numpy arrays
        image = np.array(image, dtype=np.float32) / 255.0
        depth = np.array(depth, dtype=np.float32) / 256.0  # KITTI depth is stored as uint16
        
        # Apply transforms if specified
        if self.transform is not None:
            transformed = self.transform(image=image, depth=depth)
            image = transformed['image']
            depth = transformed['depth']
        
        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
        depth = torch.from_numpy(depth).unsqueeze(0)  # (H,W) -> (1,H,W)
        
        return {
            'image': image,
            'depth': depth
        }

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
            'rmse': lambda pred, target: torch.sqrt(torch.mean((pred - target) ** 2)),
            'abs_rel': lambda pred, target: torch.mean(torch.abs(pred - target) / target),
            'sq_rel': lambda pred, target: torch.mean(((pred - target) ** 2) / target)
        }
