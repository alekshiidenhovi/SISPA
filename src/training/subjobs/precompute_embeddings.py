import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator


@torch.no_grad()
def precompute_embeddings(
    accelerator: Accelerator,
    trained_model: torch.nn.Module,
    dataloader: DataLoader,
    shard_idx: int,
):
    """
    Precompute embeddings for a dataset using a trained model.

    Parameters
    ----------
    accelerator : Accelerator
        Accelerator to use for computing embeddings
    trained_model : torch.nn.Module
        Trained model to use for computing embeddings
    dataloader : DataLoader
        DataLoader for the dataset
    shard_idx : int
        Index of the model shard being precomputed

    Returns
    -------
    tuple
        Tuple of (embeddings, labels). Embeddings are of shape (N, D) and labels are of shape (N,).
    """
    prepared_trained_model, prepared_dataloader = accelerator.prepare(
        trained_model, dataloader
    )

    all_embeddings = torch.tensor([], device=accelerator.device, dtype=torch.float16)
    all_labels = torch.tensor([], device=accelerator.device, dtype=torch.int64)

    dataloader_with_progress_bar = tqdm(
        prepared_dataloader, desc=f"Precomputing embeddings for shard {shard_idx}"
    )
    for batch_idx, (images, labels) in enumerate(dataloader_with_progress_bar):
        with accelerator.autocast():
            dataloader_with_progress_bar.set_description(
                f"Precomputing embeddings for shard {shard_idx} - Batch {batch_idx}"
            )

            embeddings = prepared_trained_model(images)
            all_embeddings = torch.cat((all_embeddings, embeddings), dim=0)
            all_labels = torch.cat((all_labels, labels), dim=0)

    return all_embeddings, all_labels
