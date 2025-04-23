import torch
import torch.nn as nn
import wandb
import wandb.wandb_run
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from models.sispa.sispa_embedding_aggregator import SISPAEmbeddingAggregator
from training.subjobs.utils import compute_prediction_statistics
from common.tracking import init_wandb_run


def train_aggregation_classifier(
    accelerator: Accelerator,
    prepared_aggregator: SISPAEmbeddingAggregator,
    prepared_optimizer: torch.optim.Optimizer,
    prepared_training_embeddings_dataloader: DataLoader,
    prepared_validation_embeddings_dataloader: DataLoader,
    loss_fn: nn.Module,
    val_check_interval_percentage: float,
    epochs: int,
    experiment_group_name: str,
):
    """
    Train an aggregation classifier on precomputed embeddings from multiple shards.

    Args:
        accelerator : Accelerator
            Accelerator to use for training
        prepared_aggregator : SISPAEmbeddingAggregator
            Prepared embedding aggregator
        prepared_optimizer : torch.optim.Optimizer
            Optimizer for the embedding aggregator
        prepared_training_embeddings_dataloader : DataLoader
            Prepared dataloader for the full training dataset of precomputed embeddings
        prepared_validation_embeddings_dataloader : DataLoader
            Prepared dataloader for the validation dataset of precomputed embeddings
        loss_fn : nn.Module
            Loss function
        val_check_interval_percentage : float
            Percentage of training batches of an epoch after which to run validation
        epochs : int
            Number of epochs to train for
        experiment_group_name : str
            Name of the experiment group

    Returns:
        nn.Module
            Trained embedding aggregator on the CPU
    """
    wandb_run = init_wandb_run(
        experiment_group_name=experiment_group_name,
        experiment_name="Aggregator training",
        reinit=True,
    )
    val_check_interval = int(
        val_check_interval_percentage * len(prepared_training_embeddings_dataloader)
    )
    prepared_aggregator.train()
    for epoch_idx in range(epochs):
        training_progress_bar = tqdm(prepared_training_embeddings_dataloader)
        for training_batch_idx, (embeddings, labels) in enumerate(
            training_progress_bar
        ):
            with accelerator.accumulate(prepared_aggregator):
                with accelerator.autocast():
                    training_progress_bar.set_description(
                        f"Training aggregation classifier, epoch {epoch_idx + 1}/{epochs}, training batch {training_batch_idx + 1}/{len(prepared_training_embeddings_dataloader)}"
                    )
                    outputs = prepared_aggregator(embeddings)
                    loss, num_predicted, num_correct = compute_prediction_statistics(
                        loss_fn,
                        outputs,
                        labels,
                    )

                    training_metrics = {
                        "aggregation/training_loss": loss.item(),
                        "aggregation/training_accuracy": num_correct / num_predicted,
                    }

                    wandb_run.log(training_metrics)
                    training_progress_bar.set_postfix(
                        {
                            "training_loss": loss.item(),
                            "training_accuracy": num_correct / num_predicted,
                        }
                    )

                    accelerator.backward(loss)
                    prepared_optimizer.step()
                    prepared_optimizer.zero_grad()

            if (training_batch_idx + 1) % val_check_interval == 0:
                validate_aggregation_training(
                    accelerator=accelerator,
                    prepared_aggregator=prepared_aggregator,
                    prepared_validation_embeddings_dataloader=prepared_validation_embeddings_dataloader,
                    loss_fn=loss_fn,
                    epoch_idx=epoch_idx,
                    training_batch_idx=training_batch_idx,
                    epochs=epochs,
                    wandb_run=wandb_run,
                )
                prepared_aggregator.train()

    return accelerator.unwrap_model(prepared_aggregator).cpu()


@torch.no_grad()
def validate_aggregation_training(
    accelerator: Accelerator,
    prepared_aggregator: SISPAEmbeddingAggregator,
    prepared_validation_embeddings_dataloader: DataLoader,
    loss_fn: nn.Module,
    epoch_idx: int,
    training_batch_idx: int,
    epochs: int,
    wandb_run: wandb.wandb_run.Run,
):
    """
    Validate the aggregation classifier using embeddings from validation data.

    Parameters
    ----------
    accelerator : Accelerator
        Accelerator to use for validation
    prepared_aggregator : SISPAEmbeddingAggregator
        Prepared aggregator to validate
    prepared_validation_embeddings_dataloader : DataLoader
        Prepared validation dataloader
    loss_fn : nn.Module
        Loss function
    epoch_idx : int
        Current epoch index
    training_batch_idx : int
        Current training batch index
    epochs : int
        Total number of epochs
    wandb_run : wandb.wandb_run.Run
        Wandb run to use for logging

    Returns
    -------
    tuple
        (validation loss, validation accuracy)
    """
    prepared_aggregator.eval()
    total_validation_loss = 0.0
    total_validation_correct = 0
    total_validation_predicted = 0

    validation_progress_bar = tqdm(prepared_validation_embeddings_dataloader)
    for validation_batch_idx, (embeddings, labels) in enumerate(
        validation_progress_bar
    ):
        with accelerator.autocast():
            validation_progress_bar.set_description(
                f"Validating aggregation, epoch {epoch_idx + 1}/{epochs}, during training batch {training_batch_idx + 1}, validation batch {validation_batch_idx + 1}/{len(prepared_validation_embeddings_dataloader)}"
            )
            outputs = prepared_aggregator(embeddings)
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
                }
            )

    val_loss = total_validation_loss / len(prepared_validation_embeddings_dataloader)
    val_accuracy = (
        100 * total_validation_correct / total_validation_predicted
        if total_validation_predicted > 0
        else 0.0
    )

    validation_metrics = {
        "aggregation/validation_loss": val_loss,
        "aggregation/validation_accuracy": val_accuracy,
    }
    wandb_run.log(validation_metrics)

    prepared_aggregator.train()

    return val_loss, val_accuracy
