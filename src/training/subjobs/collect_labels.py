import torch
from torch.utils.data import DataLoader


def collect_labels(dataloader: DataLoader) -> torch.Tensor:
    """
    Iterates through a DataLoader and collects all labels.

    Parameters
    ----------
    dataloader : DataLoader
        The DataLoader to iterate through. Assumes batches yield (data, labels).

    Returns
    -------
    torch.Tensor
        A tensor containing all labels from the dataset, concatenated on the CPU.
    """
    all_labels = torch.tensor([], dtype=torch.int64)
    for _, labels in dataloader:
        all_labels = torch.cat([all_labels, labels], dim=0)

    return all_labels
