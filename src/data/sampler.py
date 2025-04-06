from typing import List, Optional
import torch
from torch.utils.data import Sampler


class PartitionedSampler(Sampler):
    """Sampler that divides dataset into roughly equal parts and returns indices for one partition.
    
    Args:
        data_source: Dataset to sample from
        num_partitions: Number of partitions to divide dataset into
        partition_idx: Which partition to use (0-based index)
        shuffle: Whether to shuffle indices within the partition
        seed: Random seed for reproducibility
    """

    def __init__(self,
                 data_source,
                 num_partitions: int = 4,
                 partition_idx: int = 0,
                 shuffle: bool = True,
                 seed: Optional[int] = None):
        self.data_source = data_source
        self.num_partitions = num_partitions
        self.partition_idx = partition_idx
        self.shuffle = shuffle

        if seed is not None:
            torch.manual_seed(seed)

        # Calculate partition sizes
        total_size = len(data_source)
        base_size = total_size // num_partitions
        remainder = total_size % num_partitions

        # Distribute remainder across partitions
        self.partition_sizes = [
            base_size + (1 if i < remainder else 0)
            for i in range(num_partitions)
        ]

        # Calculate start and end indices for this partition
        start_idx = sum(self.partition_sizes[:partition_idx])
        end_idx = start_idx + self.partition_sizes[partition_idx]

        # Get indices for this partition
        self.indices = list(range(start_idx, end_idx))

        if shuffle:
            self.indices = torch.randperm(len(self.indices)).tolist()

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

    def get_partition_size(self, partition_idx: int) -> int:
        """Get the size of a specific partition."""
        return self.partition_sizes[partition_idx]

    def get_total_size(self) -> int:
        """Get the total size of all partitions."""
        return sum(self.partition_sizes)
