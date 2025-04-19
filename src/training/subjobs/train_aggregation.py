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
    aggregator: EmbeddingAggregator,
    optimizer: torch.optim.Optimizer,
    training_embeddings_dataloader: DataLoader,
    validation_embeddings_dataloader: DataLoader,
    loss_fn: nn.Module,
    val_batch_interval: int,
    epochs: int,
):
    """
    Train an aggregation classifier on precomputed embeddings from multiple shards.

    Args:
        accelerator : Accelerator
            Accelerator to use for training
        aggregator : nn.Module
            Embedding aggregator
        optimizer : torch.optim.Optimizer
            Optimizer for the embedding aggregator
        training_embeddings_dataloader : DataLoader
            Dataloader for the full training dataset of precomputed embeddings
        validation_embeddings_dataloader : DataLoader
            Dataloader for the validation dataset of precomputed embeddings
        loss_fn : nn.Module
            Loss function
        val_batch_interval : int
            Interval for validation, the model is validated every `val_batch_interval` batches
        epochs : int
            Number of epochs to train for

    Returns:
        nn.Module
            Trained embedding aggregator on the CPU
    """

    (
        prepared_aggregator,
        prepared_optimizer,
        prepared_training_embeddings_dataloader,
        prepared_validation_embeddings_dataloader,
    ) = accelerator.prepare(
        aggregator,
        optimizer,
        training_embeddings_dataloader,
        validation_embeddings_dataloader,
    )

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

    return accelerator.unwrap_model(prepared_aggregator).cpu()


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
    prepared_validation_embeddings_dataloader : DataLoader
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
