import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from common.logger import logger
from models.sispa.sispa_embedding_aggregator import EmbeddingAggregator
from training.subjobs.utils import compute_prediction_statistics


def train_aggregation_classifier(
    accelerator: Accelerator,
    prepared_aggregator: EmbeddingAggregator,
    prepared_optimizer: torch.optim.Optimizer,
    prepared_training_embeddings_dataloader: DataLoader,
    prepared_validation_embeddings_dataloader: DataLoader,
    loss_fn: nn.Module,
    val_batch_interval: int,
    epochs: int,
):
    """
    Train an aggregation classifier on precomputed embeddings from multiple shards.

    Parameters
    ----------
    accelerator : Accelerator
        Accelerator to use for training
    prepared_aggregator : nn.Module
        Prepared embedding aggregator
    prepared_optimizer : torch.optim.Optimizer
        Prepared optimizer for the embedding aggregator
    prepared_training_embeddings_dataloader : DataLoader
        Prepared dataloader for the full training dataset of precomputed embeddings
    prepared_validation_embeddings_dataloader : DataLoader
        Prepared validation dataloader of precomputed embeddings
    loss_fn : nn.Module
        Loss function
    val_batch_interval : int
        Interval for validation, the model is validated every `val_batch_interval` batches
    epochs : int
        Number of epochs to train for
    """

    prepared_aggregator.train()
    for epoch_idx in range(epochs):
        total_training_loss = 0.0
        total_training_correct = 0
        total_training_predicted = 0

        training_progress_bar = tqdm(
            prepared_training_embeddings_dataloader,
            desc=f"Training aggregation classifier, epoch {epoch_idx + 1}/{epochs}",
        )

        for batch_idx, (embeddings, labels) in enumerate(training_progress_bar):
            with accelerator.accumulate(prepared_aggregator):
                with accelerator.autocast():
                    outputs = prepared_aggregator(embeddings)
                    loss, num_predicted, num_correct = compute_prediction_statistics(
                        loss_fn,
                        outputs,
                        labels,
                    )
                    total_training_loss += loss.item()
                    total_training_predicted += num_predicted
                    total_training_correct += num_correct

                    training_progress_bar.set_postfix(
                        {
                            "training_loss": loss.item(),
                            "training_accuracy": num_correct / num_predicted,
                        }
                    )

                    accelerator.backward(loss)
                    prepared_optimizer.step()
                    prepared_optimizer.zero_grad()

            if (batch_idx + 1) % val_batch_interval == 0:
                val_loss, val_accuracy = validate_aggregation_training(
                    accelerator=accelerator,
                    prepared_aggregator=prepared_aggregator,
                    prepared_validation_embeddings_dataloader=prepared_validation_embeddings_dataloader,
                    loss_fn=loss_fn,
                    epoch_idx=epoch_idx,
                    batch_idx=batch_idx,
                    epochs=epochs,
                )
                logger.info(
                    f"Aggregation, epoch {epoch_idx + 1}, batch {batch_idx + 1}: "
                    f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
                )
                prepared_aggregator.train()

        final_accuracy = 100 * total_training_correct / total_training_predicted
        logger.info(
            f"Aggregation, epoch {epoch_idx + 1}: "
            f"Loss: {total_training_loss / len(prepared_training_embeddings_dataloader):.4f}, "
            f"Accuracy: {final_accuracy:.2f}%"
        )


@torch.no_grad()
def validate_aggregation_training(
    accelerator: Accelerator,
    prepared_aggregator: EmbeddingAggregator,
    prepared_validation_embeddings_dataloader: DataLoader,
    loss_fn: nn.Module,
    epoch_idx: int,
    batch_idx: int,
    epochs: int,
):
    """
    Validate the aggregation classifier using embeddings from validation data.

    Parameters
    ----------
    accelerator : Accelerator
        Accelerator to use for validation
    prepared_aggregator : EmbeddingAggregator
        Prepared aggregator to validate
    prepared_embeddings_dataloader : DataLoader
        Prepared validation dataloader
    loss_fn : nn.Module
        Loss function
    epoch_idx : int
        Current epoch index
    batch_idx : int
        Current batch index
    epochs : int
        Total number of epochs

    Returns
    -------
    tuple
        (validation loss, validation accuracy)
    """
    prepared_aggregator.eval()
    total_validation_loss = 0.0
    total_validation_correct = 0
    total_validation_predicted = 0

    validation_progress_bar = tqdm(
        prepared_validation_embeddings_dataloader,
        desc=f"Validating aggregation, epoch {epoch_idx + 1}/{epochs}, batch {batch_idx + 1}",
    )

    for embeddings, labels in validation_progress_bar:
        with accelerator.autocast():
            outputs = prepared_aggregator(embeddings)
            loss, num_predicted, num_correct = compute_prediction_statistics(
                loss_fn,
                outputs,
                labels,
            )

            total_validation_loss += loss.item()
            total_validation_predicted += num_predicted
            total_validation_correct += num_correct

    val_loss = total_validation_loss / len(prepared_validation_embeddings_dataloader)
    val_accuracy = (
        100 * total_validation_correct / total_validation_predicted
        if total_validation_predicted > 0
        else 0.0
    )

    prepared_aggregator.train()

    return val_loss, val_accuracy
