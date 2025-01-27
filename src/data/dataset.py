"""
Base class for datasets following PyTorch dataset conventions.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple


class BaseDataset(ABC):
    """
    Abstract base class for datasets.

    """

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

    def get_sample_shape(self) -> Tuple[int, ...]:
        """
        Optional method to get the shape of samples in the dataset.

        Returns:
            Tuple[int, ...]: Shape of samples in the dataset
        """
        raise NotImplementedError("get_sample_shape method not implemented")
