import typing as T
import torch.nn as nn
from common.tracking import init_wandb_run
from common.types import ACCELERATOR, AVAILABLE_DATASETS
from torch.utils.data import DataLoader
from tqdm import tqdm
from training.utils import compute_prediction_statistics, majority_vote_class_indices


def test_sisa_framework(
    accelerator: ACCELERATOR,
    prepared_trained_embedding_models: T.List[nn.Module],
    prepared_trained_classifiers: T.List[nn.Module],
    prepared_test_dataloader: DataLoader,
    experiment_group_name: str,
    dataset_name: AVAILABLE_DATASETS,
) -> None:
    """
    Test the SISPA framework on testing data.

    Args:
        accelerator : Accelerator
            Accelerator to use for testing
        prepared_trained_embedding_models : T.List[nn.Module]
            Prepared trained embedding models
        prepared_trained_classifiers : T.List[nn.Module]
            Prepared trained classifiers
        prepared_test_dataloader : DataLoader
            Test dataloader
        experiment_group_name : str
            Name of the experiment group
        dataset_name : AVAILABLE_DATASETS
            Name of the dataset
    """
    wandb_run = init_wandb_run(
        dataset_name=dataset_name,
        experiment_group_name=experiment_group_name,
        experiment_name="E2E testing",
        reinit=True,
    )
    for prepared_trained_embedding_model, prepared_trained_classifier in zip(
        prepared_trained_embedding_models, prepared_trained_classifiers
    ):
        prepared_trained_embedding_model.eval()
        prepared_trained_classifier.eval()

    total_test_predicted = 0
    total_test_correct = 0

    test_progress_bar = tqdm(prepared_test_dataloader)
    for test_batch_idx, (images, labels) in enumerate(test_progress_bar):
        with accelerator.autocast():
            test_progress_bar.set_description(
                f"Testing SISPA framework, testing batch {test_batch_idx + 1}/{len(prepared_test_dataloader)}"
            )
            shard_pred_probabilities = [
                prepared_trained_classifier(prepared_trained_embedding_model(images))
                for prepared_trained_embedding_model, prepared_trained_classifier in zip(
                    prepared_trained_embedding_models, prepared_trained_classifiers
                )
            ]
            class_predictions = majority_vote_class_indices(
                shard_pred_probabilities, device=accelerator.device
            )
            print(f"class_predictions.shape: {class_predictions.shape}")
            print(f"labels.shape: {labels.shape}")
            num_predicted, num_correct = compute_prediction_statistics(
                class_predictions,
                labels,
            )
            total_test_predicted += num_predicted
            total_test_correct += num_correct

            test_progress_bar.set_postfix(
                {
                    "testing_accuracy": total_test_correct / total_test_predicted,
                }
            )

    test_accuracy = (
        100 * total_test_correct / total_test_predicted
        if total_test_predicted > 0
        else 0.0
    )
    test_metrics = {
        "testing/testing_accuracy": test_accuracy,
    }
    wandb_run.log(test_metrics)
