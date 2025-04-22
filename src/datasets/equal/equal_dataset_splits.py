import typing as T
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split


def create_equal_distribution_dataset_splits(
    dataset: Dataset,
    train_val_test_split: T.Tuple[float, float, float],
    num_shards: int,
    seed: int,
) -> T.Tuple[T.List[T.List[int]], T.List[int], T.List[int]]:
    """Creates stratified train/val/test splits and equal training shards.

    Splits a dataset into train, validation and test sets while maintaining class balance.
    The training set is further divided into shards, where each shard has an equal distribution
    of all classes, proportional to the original dataset's class distribution.

    Args:
        dataset: The source dataset to split
        train_val_test_split: Tuple of (train_ratio, val_ratio, test_ratio) that sum to 1.0
        num_shards: Number of equal shards to create from the training set
        seed: Random seed for reproducibility

    Returns:
        Tuple containing:
        - train_shard_indices: List of lists, where each inner list contains indices for a equal shard
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

    train_shard_indices = create_equal_distribution_shard_splits(
        dataset=dataset,
        train_indices=train_indices,
        num_shards=num_shards,
        seed=seed,
    )

    return train_shard_indices, val_indices, test_indices


def create_equal_distribution_shard_splits(
    dataset: Dataset,
    train_indices: T.List[int],
    num_shards: int,
    seed: int,
) -> T.List[T.List[int]]:
    """
    Creates balanced shards of a dataset where each shard has the same class distribution.

    Each shard will contain approximately the same number of samples from each class,
    maintaining the original dataset's class distribution proportions.

    Args:
        dataset: The source dataset to create shards from
        train_indices: List of indices for the training split
        num_shards: Number of balanced shards to create
        seed: Random seed for reproducibility

    Returns:
        List of lists, where each inner list contains indices for a balanced shard
    """
    np.random.seed(seed)

    original_indices_by_class = defaultdict(list)
    subset_dataset = Subset(dataset, train_indices)
    shards = [[] for _ in range(num_shards)]

    for subset_idx, (_, label) in enumerate(subset_dataset):
        original_idx = train_indices[subset_idx]
        original_indices_by_class[label].append(original_idx)

    for label, original_class_indices in original_indices_by_class.items():
        original_indices = np.array(original_class_indices)
        np.random.shuffle(original_indices)

        for i, original_idx in enumerate(original_indices):
            shard_idx = i % num_shards
            shards[shard_idx].append(original_idx)

    return shards
