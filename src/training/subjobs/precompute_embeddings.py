import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator


@torch.no_grad()
def precompute_embeddings(
    accelerator: Accelerator,
    prepared_model: torch.nn.Module,
    prepared_dataloader: DataLoader,
    shard_idx: int,
):
    """
    Precompute embeddings for a dataset using a trained model.

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

    dataloader_with_progress_bar = tqdm(prepared_dataloader)
    for batch_idx, (images, labels) in enumerate(dataloader_with_progress_bar):
        with accelerator.autocast():
            embeddings = prepared_model(images)
            all_embeddings = torch.cat((all_embeddings, embeddings), dim=0)
            all_labels = torch.cat((all_labels, labels), dim=0)

            dataloader_with_progress_bar.set_description(
                f"Precomputing embeddings for shard {shard_idx} - Batch {batch_idx}"
            )

    return all_embeddings, all_labels
