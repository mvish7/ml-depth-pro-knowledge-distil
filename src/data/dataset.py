"""
Base class for datasets following PyTorch dataset conventions.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np
from torchvision.transforms import (
    Compose, Resize,
    ConvertImageDtype,
    Lambda,
    Normalize,
    ToTensor,
    Pad
)


class BaseDataset(ABC):
    """
    Abstract base class for datasets.

    """
    def __init__(self, precision: str = "torch.bfloat16", device: str = "cpu"):
        self.precision = precision
        self.image_transform = None
        self.depth_transform = None

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        """
        Returns a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            Any: The sample at the given index. Could be a tuple of
                 (input, target) or any other format needed by your model
        """
        pass

    def load_image(self, path: str) -> np.ndarray:
        """
        reads the rgb image from the disk
        Returns:
            np.ndarray containing the image
        """
        raise NotImplementedError("image loading not implemented")

    def load_depth(self, path: str) -> np.ndarray:
        """
        reads the depth image from the disk and preprocesses it if necessary
        Returns:
            np.ndarray containing planar depth values
        """
        raise NotImplementedError("depth loading not implemented")

    def get_sample_shape(self) -> Tuple[int, ...]:
        """
        Optional method to get the shape of samples in the dataset.

        Returns:
            Tuple[int, ...]: Shape of samples in the dataset
        """
        raise NotImplementedError("get_sample_shape method not implemented")
