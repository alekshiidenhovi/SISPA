from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import typing as T
from pathlib import Path


def get_mnist_dataloader(
    data_dir: T.Union[Path, str],
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
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
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
