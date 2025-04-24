import torch
import click
import typing as T


def majority_vote_class_indices(
    shard_pred_probabilities: T.List[torch.Tensor], device: torch.device
) -> torch.Tensor:
    """
    Perform a majority vote on the predictions of the given shard probabilities.

    Parameters
    ----------
    shard_pred_probabilities : T.List[torch.Tensor]
        List of shard predictionprobabilities, tensors of shape (batch_size, num_classes)

    Returns
    -------
    torch.Tensor
        Majority voted class indices predictions, tensor of shape (batch_size, 1)
    """
    batch_size, num_classes = shard_pred_probabilities[0].shape
    shard_class_predictions = [
        torch.argmax(shard_probability, dim=1)
        for shard_probability in shard_pred_probabilities
    ]  # shape (num_shards, batch_size)
    shard_class_predictions = torch.tensor(shard_class_predictions, device=device)
    shard_class_predictions = shard_class_predictions.transpose(
        0, 1
    )  # shape (batch_size, num_shards)

    final_preds = []
    for i in range(batch_size):
        sample_preds = shard_class_predictions[i]  # shape (num_shards,)
        class_counts = torch.zeros(num_classes, device=device)
        for pred in sample_preds:
            class_counts[pred] += 1
        majority_class = torch.argmax(class_counts)
        final_preds.append(majority_class)

    return torch.tensor(final_preds, device=device).view(-1, 1)


def compute_prediction_statistics(
    class_predictions: torch.Tensor,
    labels: torch.Tensor,
):
    """
    Compute prediction statistics for a batch of data.

    Parameters
    ----------
    class_predictions : torch.Tensor
        Predicted class indices for each class, shape (batch_size, 1)
    labels : torch.Tensor
        Ground truth labels, shape (batch_size, 1)

    Returns
    -------
    tuple
        Contains:
        - num_predicted : int
            Number of samples in the batch
        - num_correct : int
            Number of correct predictions in the batch
    """
    num_predicted = labels.size(0)
    num_correct = (class_predictions == labels).sum().item()
    return num_predicted, num_correct


def parse_int_list(
    ctx: click.Context, param: click.Option, value: T.Optional[str]
) -> T.Optional[T.List[int]]:
    if value is None:
        return None
    return [int(x.strip()) for x in value.split(",")]
