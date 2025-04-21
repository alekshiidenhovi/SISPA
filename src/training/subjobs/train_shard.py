import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from training.subjobs.utils import compute_prediction_statistics


def train_sharded_embedding_model(
    accelerator: Accelerator,
    embedding_model: nn.Module,
    classifier: nn.Module,
    optimizer: optim.Optimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    loss_fn: nn.Module,
    val_batch_interval: int,
    epochs: int,
    shard_idx: int,
    wandb_run: wandb.wandb_run.Run,
) -> nn.Module:
    """
    Train a model on a specific shard of data.

    Args:
        accelerator : Accelerator
            Accelerator to use for training
        embedding_model : nn.Module
            Untrained embedding model
        classifier : nn.Module
            Untrained classifier
        optimizer : optim.Optimizer
            Optimizer
        train_dataloader : DataLoader
            Train dataloader
        val_dataloader : DataLoader
            Validation dataloader
        loss_fn : nn.Module
            Loss function
        shard_idx : int
            Index of the shard to train on
        epochs : int
            Number of epochs to train for
        wandb_run : wandb.wandb_run.Run
            W&B run to use for logging

    Returns:
        nn.Module
            Trained embedding model on the CPU
    """
    (
        prepared_embedding_model,
        prepared_classifier,
        prepared_optimizer,
        prepared_train_dataloader,
        prepared_val_dataloader,
    ) = accelerator.prepare(
        embedding_model, classifier, optimizer, train_dataloader, val_dataloader
    )

    prepared_embedding_model.train()
    prepared_classifier.train()
    for epoch_idx in range(epochs):
        progress_bar = tqdm(prepared_train_dataloader)
        for batch_idx, (images, labels) in enumerate(progress_bar):
            with accelerator.accumulate(prepared_embedding_model):
                with accelerator.autocast():
                    progress_bar.set_description(
                        f"Training shard {shard_idx}, epoch {epoch_idx + 1}/{epochs}, batch {batch_idx + 1}/{len(prepared_train_dataloader)}"
                    )
                    embeddings = prepared_embedding_model(images)
                    outputs = prepared_classifier(embeddings)
                    loss, num_predicted, num_correct = compute_prediction_statistics(
                        loss_fn,
                        outputs,
                        labels,
                    )

                    training_metrics = {
                        "training_loss": loss.item(),
                        "training_accuracy": num_correct / num_predicted,
                    }

                    progress_bar.set_postfix(training_metrics)
                    wandb_run.log(training_metrics)

                    accelerator.backward(loss)
                    prepared_optimizer.step()
                    prepared_optimizer.zero_grad()

            if (batch_idx + 1) % val_batch_interval == 0:
                validate_shard_training(
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

                prepared_embedding_model.train()
                prepared_classifier.train()

    return accelerator.unwrap_model(prepared_embedding_model).cpu()


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
    wandb_run: wandb.wandb_run.Run,
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
    wandb_run : wandb.wandb_run.Run
        W&B run to use for logging

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

    validation_progress_bar = tqdm(prepared_val_dataloader)
    for images, labels in validation_progress_bar:
        with accelerator.autocast():
            validation_progress_bar.set_description(
                f"Validating shard {shard_idx}, epoch {epoch_idx + 1}/{epochs}, batch {batch_idx + 1}/{len(prepared_val_dataloader)}"
            )
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

    validation_metrics = {
        "validation_loss": val_loss,
        "validation_accuracy": val_accuracy,
    }
    wandb_run.log(validation_metrics)

    prepared_embedding_model.train()
    prepared_classifier.train()

    return val_loss, val_accuracy
