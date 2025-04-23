import torch
import click
import typing as T


def compute_prediction_statistics(
    outputs: torch.Tensor,
    labels: torch.Tensor,
):
    """
    Compute prediction statistics for a batch of data.

    Parameters
    ----------
    outputs : torch.Tensor
        Predicted output probabilities for each class
    labels : torch.Tensor
        Ground truth labels

    Returns
    -------
    tuple
        Contains:
        - num_predicted : int
            Number of samples in the batch
        - num_correct : int
            Number of correct predictions in the batch
    """
    _, predicted = torch.max(outputs, dim=1)
    num_predicted = labels.size(0)
    num_correct = (predicted == labels).sum().item()
    return num_predicted, num_correct


def parse_int_list(
    ctx: click.Context, param: click.Option, value: T.Optional[str]
) -> T.Optional[T.List[int]]:
    if value is None:
        return None
    return [int(x.strip()) for x in value.split(",")]
