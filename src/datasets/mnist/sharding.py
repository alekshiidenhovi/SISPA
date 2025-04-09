from torchvision import transforms
from torch.utils.data import Dataset
import typing as T
import numpy as np
from datasets.mnist.sharded_dataset import ShardedMNIST


def calculate_split_indices(num_shards: int, num_samples: int, idx: int) -> T.List[int]:
    """Calculate split points for a given shard index."""
    splits = [0]
    divisor = num_shards - 1
    for i in range(num_shards):
        if i == idx:
            splits.append(splits[-1] + 0.5)
        else:
            splits.append(splits[-1] + 0.5 / divisor)
    for i, split in enumerate(splits):
        splits[i] = int(split * num_samples)
    return splits


def create_mnist_shards(
    dataset: Dataset,
    num_shards: int = 10,
    transform: T.Optional[transforms.Compose] = None,
    seed: int = 42,
) -> T.List[ShardedMNIST]:
    """
    Create shards of MNIST dataset where each shard has:
    - 50% samples from a specific class
    - 50% samples from other based on the class distribution
    - Each sample appears in exactly one shard (MECE principle)

    Parameters
    ----------
    dataset : Dataset
        MNIST dataset to shard
    num_shards : int, default=10
        Number of shards to create
    transform : Optional[transforms.Compose], default=None
        Transform to apply to images
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    List[ShardedMNIST]
        List of ShardedMNIST datasets, one for each shard
    """
    np.random.seed(seed)

    indices_by_class = [[] for _ in range(num_shards)]
    for idx, (_, label) in enumerate(dataset):
        indices_by_class[label].append(idx)

    for label in range(num_shards):
        np.random.shuffle(indices_by_class[label])

    shard_indices = [[] for _ in range(num_shards)]
    for label in range(num_shards):
        indices = indices_by_class[label]
        num_samples = len(indices)
        splits = calculate_split_indices(num_shards, num_samples, label)

        for i in range(num_shards):
            shard_indices[i].extend(indices[splits[i] : splits[i + 1]])

    shards = [ShardedMNIST(dataset, indices, transform) for indices in shard_indices]

    return shards
