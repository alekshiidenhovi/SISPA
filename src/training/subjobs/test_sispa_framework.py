import torch
from common.tracking import init_wandb_run
from common.types import ACCELERATOR, AVAILABLE_DATASETS
from models.sispa import SISPAFramework
from torch.utils.data import DataLoader
from tqdm import tqdm
from training.utils import compute_prediction_statistics


def test_sispa_framework(
    accelerator: ACCELERATOR,
    prepared_sispa_framework: SISPAFramework,
    prepared_test_dataloader: DataLoader,
    experiment_group_name: str,
    dataset_name: AVAILABLE_DATASETS,
) -> None:
    """
    Test the SISPA framework on testing data.

    Args:
        accelerator : Accelerator
            Accelerator to use for testing
        prepared_sispa_framework : SISPAFramework
            Prepared SISPA framework
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
    prepared_sispa_framework.eval()
    total_test_predicted = 0
    total_test_correct = 0

    test_progress_bar = tqdm(prepared_test_dataloader)
    for test_batch_idx, (images, labels) in enumerate(test_progress_bar):
        with accelerator.autocast():
            test_progress_bar.set_description(
                f"Testing SISPA framework, testing batch {test_batch_idx + 1}/{len(prepared_test_dataloader)}"
            )
            outputs = prepared_sispa_framework(images)
            class_predictions = torch.argmax(outputs, dim=1)
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
