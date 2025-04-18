import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator


@torch.no_grad()
def precompute_embeddings(
    accelerator: Accelerator,
    prepared_model: torch.nn.Module,
    dataloader_with_progress_bar: tqdm[DataLoader],
):
    """
    Compute embeddings for a dataset using a trained model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model to use for computing embeddings
    dataloader : DataLoader
        DataLoader for the dataset
    device : torch.device
        Device to compute embeddingss on
    shard_idx : int, optional
        Index of the shard if computing embeddings for a specific shard

    Returns
    -------
    tuple
        Tuple of (embeddings, labels)
    """
    all_embeddings = torch.tensor([], device=accelerator.device, dtype=torch.float16)
    all_labels = torch.tensor([], device=accelerator.device, dtype=torch.int64)
    prepared_model.eval()
    for images, labels in dataloader_with_progress_bar:
        with accelerator.autocast():
            embeddings = prepared_model(images)
            all_embeddings = torch.cat((all_embeddings, embeddings), dim=0)
            all_labels = torch.cat((all_labels, labels), dim=0)

    return all_embeddings, all_labels
