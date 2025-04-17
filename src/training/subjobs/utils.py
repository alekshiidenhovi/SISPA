import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator


def core_loop(
    prepared_embedding_model: nn.Module,
    prepared_classifier: nn.Module,
    loss_fn: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
):
    """
    Core training/validation loop that processes a single batch of data.

    Parameters
    ----------
    prepared_embedding_model : nn.Module
        Model that generates embeddings from input images
    prepared_classifier : nn.Module
        Model that classifies embeddings into output classes
    loss_fn : nn.Module
        Loss function to compute training/validation loss
    images : torch.Tensor
        Batch of input images
    labels : torch.Tensor
        Ground truth labels for the batch

    Returns
    -------
    tuple
        Contains:
        - loss : torch.Tensor
            Loss value for the batch
        - num_predicted : int
            Number of samples in the batch
        - num_correct : int
            Number of correct predictions in the batch
    """
    embeddings = prepared_embedding_model(images)
    outputs = prepared_classifier(embeddings)
    loss = loss_fn(outputs, labels)
    _, predicted = torch.max(outputs, dim=1)
    num_predicted = labels.size(0)
    num_correct = (predicted == labels).sum().item()
    return loss, num_predicted, num_correct


@torch.no_grad()
def validate(
    accelerator: Accelerator,
    prepared_embedding_model: nn.Module,
    prepared_classifier: nn.Module,
    prepared_val_dataloader: DataLoader,
    loss_fn: nn.Module,
    epoch_idx: int,
    shard_idx: int,
    batch_idx: int,
    epochs: int,
):
    """
    Validate the model on a validation dataset.

    Parameters
    ----------
    accelerator : Accelerator
        Accelerator to use for validation
    prepared_embedding_model : nn.Module
        Prepared embedding model to validate
    prepared_classifier : nn.Module
        Prepared classifier to validate
    prepared_val_dataloader : DataLoader
        Prepared validation dataloader
    loss_fn : nn.Module
        Loss function
    epoch_idx : int
        Index of the epoch
    shard_idx : int
        Index of the shard
    batch_idx : int
        Index of the batch
    epochs : int
        Total number of epochs

    Returns
    -------
    tuple
        (validation loss, validation accuracy)
    """
    prepared_embedding_model.eval()
    prepared_classifier.eval()
    total_validation_loss: float = 0.0
    total_validation_correct: int = 0
    total_validation_predicted: int = 0

    validation_progress_bar = tqdm(
        prepared_val_dataloader,
        desc=f"Validating shard {shard_idx}, epoch {epoch_idx + 1}/{epochs}, batch {batch_idx + 1}/{len(prepared_val_dataloader)}",
    )

    for images, labels in validation_progress_bar:
        with accelerator.autocast():
            loss, num_predicted, num_correct = core_loop(
                prepared_embedding_model,
                prepared_classifier,
                loss_fn,
                images,
                labels,
            )
            total_validation_loss += loss.item()
            total_validation_predicted += num_predicted
            total_validation_correct += num_correct

    val_loss = total_validation_loss / len(prepared_val_dataloader)
    val_accuracy = (
        100 * total_validation_correct / total_validation_predicted
        if total_validation_predicted > 0
        else 0.0
    )

    return val_loss, val_accuracy
