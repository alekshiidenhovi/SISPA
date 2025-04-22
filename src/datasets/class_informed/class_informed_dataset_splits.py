import typing as T
import math
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split


def create_class_informed_dataset_splits(
    dataset: Dataset,
    train_val_test_split: T.Tuple[float, float, float],
    sampling_ratio: float,
    seed: int,
) -> T.Tuple[T.List[T.List[int]], T.List[int], T.List[int]]:
    """Creates stratified train/val/test splits and class-informed training shards.

    Splits a dataset into train, validation and test sets while maintaining class balance.
    The training set is further divided into shards, where each shard focuses on a specific class.

    Args:
        dataset: The source dataset to split
        train_val_test_split: Tuple of (train_ratio, val_ratio, test_ratio) that sum to 1.0
        sampling_ratio: Proportion of samples that should go to each class's dedicated shard
        seed: Random seed for reproducibility

    Returns:
        Tuple containing:
        - train_shard_indices: List of lists, where each inner list contains indices for a class-focused shard
        - val_indices: List of indices for the validation set
        - test_indices: List of indices for the test set
    """
    train_ratio, val_ratio, test_ratio = train_val_test_split
    indices = list(range(len(dataset)))
    labels = [dataset[i][1] for i in indices]

    train_val_indices, test_indices, train_val_labels, _ = train_test_split(
        indices, labels, test_size=test_ratio, stratify=labels, random_state=seed
    )

    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_indices, val_indices, _, _ = train_test_split(
        train_val_indices,
        train_val_labels,
        test_size=val_ratio_adjusted,
        stratify=train_val_labels,
        random_state=seed,
    )

    class_labels = set(labels)

    train_shard_indices = create_class_informed_shard_splits(
        dataset=dataset,
        train_indices=train_indices,
        class_labels=class_labels,
        sampling_ratio=sampling_ratio,
        seed=seed,
    )

    return train_shard_indices, val_indices, test_indices


def create_class_informed_shard_splits(
    dataset: Dataset,
    train_indices: T.List[int],
    class_labels: T.Set[int],
    sampling_ratio: float,
    seed: int,
) -> T.List[T.List[int]]:
    """
    Creates balanced shards of a dataset where each shard focuses on a specific class.

    For each class label, creates a dedicated shard that contains:
    - sampling_ratio proportion of that class's samples
    - The remaining (1-sampling_ratio) samples from that class are evenly distributed across other shards
    - This ensures each sample appears in exactly one shard (Mutually Exclusive Collectively Exhaustive)

    Args:
        dataset: The source dataset to create shards from
        train_indices: List of indices for the training split
        class_labels: Set of all class labels in the dataset
        sampling_ratio: Proportion of samples that should go to each class's dedicated shard
        seed: Random seed for reproducibility

    Returns:
        List of lists, where each inner list contains indices for a class-focused shard
    """
    np.random.seed(seed)

    original_indices_by_class = defaultdict(list)
    subset_dataset = Subset(dataset, train_indices)
    shards = [[] for _ in range(len(class_labels))]

    for subset_idx, (_, label) in enumerate(subset_dataset):
        original_idx = train_indices[subset_idx]
        original_indices_by_class[label].append(original_idx)

    for label, original_class_indices in original_indices_by_class.items():
        original_indices = np.array(original_class_indices)
        np.random.shuffle(original_indices)

        num_samples = len(original_indices)
        splits = calculate_class_informed_shard_split_indices(
            class_labels=class_labels,
            class_label=label,
            num_samples=num_samples,
            sampling_ratio=sampling_ratio,
        )

        for i in range(len(class_labels)):
            shards[i].extend(original_indices[splits[i] : splits[i + 1]])

    return shards


def calculate_class_informed_shard_split_indices(
    class_labels: T.Set[int],
    class_label: int,
    num_samples: int,
    sampling_ratio: float,
) -> T.List[int]:
    """Calculate split point indices for a given class label to create balanced shards.

    For each class label, calculates split points such that:
    - The target class gets sampling_ratio proportion of its samples in its dedicated shard
    - The remaining (1-sampling_ratio) samples are evenly distributed across other shards
    - Returns a list of indices that is 1 longer than the number of classes

    Args:
        class_labels: Set of all class labels in the dataset
        class_label: The target class label for this split calculation
        num_samples: Total number of samples for the target class
        sampling_ratio: Proportion of samples that should go to the dedicated shard

    Returns:
        List of split point indices, where splits[i:i+1] defines the samples for shard i
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
