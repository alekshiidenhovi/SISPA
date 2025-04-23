import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import wandb.wandb_run
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from training.utils import compute_prediction_statistics
from common.tracking import init_wandb_run
from common.types import AVAILABLE_DATASETS


def train_sharded_embedding_model(
    accelerator: Accelerator,
    prepared_embedding_model: nn.Module,
    prepared_classifier: nn.Module,
    prepared_optimizer: optim.Optimizer,
    prepared_train_dataloader: DataLoader,
    prepared_val_dataloader: DataLoader,
    loss_fn: nn.Module,
    val_check_interval_percentage: float,
    epochs: int,
    shard_idx: int,
    experiment_group_name: str,
    dataset_name: AVAILABLE_DATASETS,
) -> nn.Module:
    """
    Train a model on a specific shard of data.

    Args:
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
            Validation dataloader
        loss_fn : nn.Module
            Loss function
        val_check_interval_percentage : float
            Percentage of training batches of an epoch after which to run validation
        epochs : int
            Number of epochs to train for
        shard_idx : int
            Index of the shard to train on
        experiment_group_name : str
            Name of the experiment group
        dataset_name : AVAILABLE_DATASETS
            Name of the dataset

    Returns:
        nn.Module
            Trained embedding model on the CPU
    """
    wandb_run = init_wandb_run(
        dataset_name=dataset_name,
        experiment_group_name=experiment_group_name,
        experiment_name=f"Shard {shard_idx} training",
        reinit=True,
    )
    prepared_embedding_model.train()
    prepared_classifier.train()
    val_batch_interval = int(
        val_check_interval_percentage * len(prepared_train_dataloader)
    )
    for epoch_idx in range(epochs):
        training_progress_bar = tqdm(prepared_train_dataloader)
        for training_batch_idx, (images, labels) in enumerate(training_progress_bar):
            with accelerator.accumulate(prepared_embedding_model, prepared_classifier):
                with accelerator.autocast():
                    training_progress_bar.set_description(
                        f"Training shard {shard_idx}, epoch {epoch_idx + 1}/{epochs}, training batch {training_batch_idx + 1}/{len(prepared_train_dataloader)}"
                    )
                    embeddings = prepared_embedding_model(images)
                    outputs = prepared_classifier(embeddings)
                    loss, num_predicted, num_correct = compute_prediction_statistics(
                        loss_fn,
                        outputs,
                        labels,
                    )

                    training_metrics = {
                        f"shard_{shard_idx}/training_loss": loss.item(),
                        f"shard_{shard_idx}/training_accuracy": num_correct
                        / num_predicted,
                    }

                    wandb_run.log(training_metrics)
                    training_progress_bar.set_postfix(
                        {
                            "training_loss": loss.item(),
                            "training_accuracy": num_correct / num_predicted,
                            "shard_idx": shard_idx,
                        }
                    )

                    accelerator.backward(loss)
                    prepared_optimizer.step()
                    prepared_optimizer.zero_grad()

            if (training_batch_idx + 1) % val_batch_interval == 0:
                validate_shard_training(
                    accelerator=accelerator,
                    prepared_embedding_model=prepared_embedding_model,
                    prepared_classifier=prepared_classifier,
                    prepared_val_dataloader=prepared_val_dataloader,
                    loss_fn=loss_fn,
                    epoch_idx=epoch_idx,
                    shard_idx=shard_idx,
                    training_batch_idx=training_batch_idx,
                    epochs=epochs,
                    wandb_run=wandb_run,
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
    training_batch_idx: int,
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
    training_batch_idx : int
        Index of the training batch
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
    for validation_batch_idx, (images, labels) in enumerate(validation_progress_bar):
        with accelerator.autocast():
            validation_progress_bar.set_description(
                f"Validating shard {shard_idx}, during straining epoch {epoch_idx + 1}/{epochs}, training batch {training_batch_idx + 1}, validation batch {validation_batch_idx + 1}/{len(prepared_val_dataloader)}"
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

            validation_progress_bar.set_postfix(
                {
                    "validation_loss": loss.item(),
                    "validation_accuracy": num_correct / num_predicted,
                    "shard_idx": shard_idx,
                }
            )

    val_loss = total_validation_loss / len(prepared_val_dataloader)
    val_accuracy = (
        100 * total_validation_correct / total_validation_predicted
        if total_validation_predicted > 0
        else 0.0
    )

    validation_metrics = {
        f"shard_{shard_idx}/validation_loss": val_loss,
        f"shard_{shard_idx}/validation_accuracy": val_accuracy,
    }
    wandb_run.log(validation_metrics)

    prepared_embedding_model.train()
    prepared_classifier.train()

    return val_loss, val_accuracy
