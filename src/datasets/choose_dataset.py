from torchvision import datasets, transforms
from common.types import AVAILABLE_DATASETS
import typing as T


def choose_dataset(
    dataset_name: AVAILABLE_DATASETS,
) -> T.Tuple[datasets.VisionDataset, int, int]:
    """
    Returns the specified dataset along with its number of channels and classes.

    Args:
        dataset_name: Name of the dataset to load

    Returns:
        Tuple containing:
        - dataset: The loaded dataset
        - num_channels: Number of channels in the dataset images
        - num_classes: Number of classes in the dataset
    """
    if dataset_name == "cifar100":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        dataset = datasets.CIFAR100(
            root="data", train=True, download=True, transform=transform
        )
        num_channels = 3
        num_classes = 100
        return dataset, num_channels, num_classes
    elif dataset_name == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                ),
            ]
        )
        dataset = datasets.CIFAR10(
            root="data", train=True, download=True, transform=transform
        )
        num_channels = 3
        num_classes = 10
        return dataset, num_channels, num_classes
    elif dataset_name == "mnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        dataset = datasets.MNIST(
            root="data", train=True, download=True, transform=transform
        )
        num_channels = 1
        num_classes = 10
        return dataset, num_channels, num_classes
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
