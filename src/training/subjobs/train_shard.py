import torch
import torch.nn as nn
import torch.optim as optim
from common.logger import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from training.subjobs.utils import compute_prediction_statistics


def train_model_on_shard(
    accelerator: Accelerator,
    prepared_embedding_model: nn.Module,
    prepared_classifier: nn.Module,
    prepared_optimizer: optim.Optimizer,
    prepared_train_dataloader: DataLoader,
    prepared_val_dataloader: DataLoader,
    loss_fn: nn.Module,
    val_batch_interval: int,
    epochs: int,
    shard_idx: int,
):
    """
    Train a model on a specific shard of data.

    Parameters
    ----------
    accelerator : Accelerator
        Accelerator to use for training
    prepared_embedding_model : nn.Module
        Prepared embedding model
    prepared_classifier : nn.Module
        Prepared classifier
    prepared_optimizer : optim.Optimizer
        Prepared optimizer
    prepared_train_dataloader : DataLoader
        Prepared train dataloader
    prepared_val_dataloader : DataLoader
        Prepared validation dataloader
    loss_fn : nn.Module
        Loss function
    shard_idx : int
        Index of the shard to train on
    epochs : int
        Number of epochs to train for
    """

    for epoch_idx in range(epochs):
        prepared_embedding_model.train()
        prepared_classifier.train()
        total_training_loss: float = 0.0
        total_training_correct: int = 0
        total_training_predicted: int = 0

        progress_bar = tqdm(
            prepared_train_dataloader,
            desc=f"Training shard {shard_idx}, epoch {epoch_idx + 1}/{epochs}",
        )
        for batch_idx, (images, labels) in enumerate(progress_bar):
            with accelerator.accumulate(prepared_embedding_model):
                with accelerator.autocast():
                    embeddings = prepared_embedding_model(images)
                    outputs = prepared_classifier(embeddings)
                    loss, num_predicted, num_correct = compute_prediction_statistics(
                        loss_fn,
                        outputs,
                        labels,
                    )
                    total_training_loss += loss.item()
                    total_training_predicted += num_predicted
                    total_training_correct += num_correct

                    progress_bar.set_postfix(
                        {
                            "training_loss": loss.item(),
                            "training_accuracy": num_correct / num_predicted,
                        }
                    )

                    accelerator.backward(loss)
                    prepared_optimizer.step()
                    prepared_optimizer.zero_grad()

            if (batch_idx + 1) % val_batch_interval == 0:
                val_loss, val_accuracy = validate_shard_training(
                    accelerator=accelerator,
                    prepared_embedding_model=prepared_embedding_model,
                    prepared_classifier=prepared_classifier,
                    prepared_val_dataloader=prepared_val_dataloader,
                    loss_fn=loss_fn,
                    epoch_idx=epoch_idx,
                    shard_idx=shard_idx,
                    batch_idx=batch_idx,
                    epochs=epochs,
                )
                logger.info(
                    f"Shard {shard_idx}, epoch {epoch_idx + 1}, batch {batch_idx + 1}: "
                    f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
                )

                prepared_embedding_model.train()
                prepared_classifier.train()

        final_accuracy = 100 * total_training_correct / total_training_predicted
        logger.info(
            f"Shard {shard_idx}, epoch {epoch_idx + 1}: "
            f"Loss: {total_training_loss / len(prepared_train_dataloader):.4f}, "
            f"Accuracy: {final_accuracy:.2f}%"
        )


@torch.no_grad()
def validate_shard_training(
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
            embeddings = prepared_embedding_model(images)
            outputs = prepared_classifier(embeddings)
            loss, num_predicted, num_correct = compute_prediction_statistics(
                loss_fn,
                outputs,
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
