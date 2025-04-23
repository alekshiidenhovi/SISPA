from common.tracking import init_wandb_run
from common.types import ACCELERATOR, AVAILABLE_DATASETS
from models.sispa import SISPAFramework
from torch.utils.data import DataLoader
from tqdm import tqdm
from training.utils import compute_prediction_statistics


def test_sispa_framework(
    accelerator: ACCELERATOR,
    sispa_framework: SISPAFramework,
    test_dataloader: DataLoader,
    epochs: int,
    experiment_group_name: str,
    dataset_name: AVAILABLE_DATASETS,
) -> None:
    """
    Test the SISPA framework on testing data.

    Args:
        accelerator : Accelerator
            Accelerator to use for testing
        sispa_framework : SISPAFramework
            SISPA framework to use for testing
        test_dataloader : DataLoader
            Test dataloader
        epochs : int
            Number of epochs to train for
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
    sispa_framework.eval()
    total_test_predicted = 0
    total_test_correct = 0
    for epoch_idx in range(epochs):
        test_progress_bar = tqdm(test_dataloader)
        for test_batch_idx, (images, labels) in enumerate(test_progress_bar):
            with accelerator.autocast():
                test_progress_bar.set_description(
                    f"Testing SISPA framework, epoch {epoch_idx + 1}/{epochs}, testing batch {test_batch_idx + 1}/{len(test_dataloader)}"
                )
                outputs = sispa_framework(images)
                num_predicted, num_correct = compute_prediction_statistics(
                    outputs,
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
