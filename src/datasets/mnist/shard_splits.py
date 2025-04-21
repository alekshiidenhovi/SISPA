from torch.utils.data import Dataset, Subset
from collections import defaultdict
import typing as T
import numpy as np
import math


def calculate_split_indices(
    class_labels: T.Set[int],
    class_label: int,
    num_samples: int,
    sampling_ratio: float,
) -> T.List[int]:
    """Calculate split point indices for a given shard index. Returns a list of indices that is 1 longer than the number of classes.

    The current class label should get a split that is the size of the sampling ratio of the total number of samples. The other classes should get a split that is the size of the remaining ratio of the total number of samples.
    """
    splits = [0]
    divisor = len(class_labels) - 1
    for current_class_label in class_labels:
        if current_class_label == class_label:
            splits.append(splits[-1] + sampling_ratio)
        else:
            splits.append(splits[-1] + (1 - sampling_ratio) / divisor)
    for i, split in enumerate(splits):
        splits[i] = math.floor(split * num_samples)
    return splits


def create_shard_splits(
    dataset: Dataset,
    train_indices: T.List[int],
    class_labels: T.Set[int],
    sampling_ratio: float,
    seed: int = 42,
) -> T.Dict[int, T.List[int]]:
    """
    Create shards of MNIST dataset where each shard has:
    - sampling_ratio of samples from a specific class
    - (1 - sampling_ratio) of samples from other classes
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
    Dict[int, List[int]]
        Dictionary of shard indices
    """
    np.random.seed(seed)

    indices_by_class = defaultdict(list)
    shard_indices_dict = defaultdict(list)
    subset_dataset = Subset(dataset, train_indices)

    for subset_idx, (_, label) in enumerate(subset_dataset):
        original_idx = train_indices[subset_idx]
        indices_by_class[label].append(original_idx)

    for label in class_labels:
        class_indices = np.array(indices_by_class[label])
        np.random.shuffle(class_indices)
        num_samples = len(class_indices)
        splits = calculate_split_indices(
            class_labels=class_labels,
            class_label=label,
            num_samples=num_samples,
            sampling_ratio=sampling_ratio,
        )

        for i in range(len(class_labels)):
            shard_indices_dict[i].extend(class_indices[splits[i] : splits[i + 1]])

    return shard_indices_dict
