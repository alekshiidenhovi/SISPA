from datasets.mnist.shard_splits import create_shard_splits
from torch.utils.data import Dataset
import typing as T
from sklearn.model_selection import train_test_split


def create_dataset_splits(
    dataset: Dataset,
    num_shards: int,
    train_val_test_split: T.Tuple[float, float, float],
    sampling_ratio: float,
    seed: int,
) -> T.Tuple[T.List[T.List[int]], T.List[int], T.List[int]]:
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

    train_shard_indices = create_shard_splits(
        dataset=dataset,
        train_indices=train_indices,
        class_labels=class_labels,
        num_shards=num_shards,
        sampling_ratio=sampling_ratio,
        seed=seed,
    )

    return train_shard_indices, val_indices, test_indices
