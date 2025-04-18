from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import typing as T
from pathlib import Path
from datasets.mnist.shard_splits import create_mnist_shards


def get_mnist_dataloader(
    data_dir: T.Union[Path, str],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """
    Create dataloaders for sharded MNIST dataset.

    Parameters
    ----------
    data_dir : Union[Path, str]
        Directory to store MNIST data
    batch_size : int
        Batch size for dataloader
    num_workers : int, default=4
        Number of workers for dataloader
    shuffle : bool, default=True
        Whether to shuffle data
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    List[DataLoader]
        List of dataloaders, one for each shard
    """
    transform = transforms.Compose(
        [
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )

    shards = create_mnist_shards(dataset, num_shards=10, transform=None, seed=seed)

    dataloaders = [
        DataLoader(
            shard,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
        for shard in shards
    ]

    return dataloaders
