import typing as T
import torch
import click
import copy
import datetime
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset, Dataset
from prefect import flow, task
from prefect.cache_policies import NO_CACHE
from common.types import (
    DATASET_SPLIT_STRATEGY_FUNCTION,
    DatasetSplitStrategy,
    AVAILABLE_DATASETS,
    ACCELERATOR,
)
from common.config import TrainingConfig
from common.tracking import init_wandb_run
from datasets.choose_dataset import choose_dataset
from datasets.choose_dataset_split_strategy import (
    choose_dataset_split_strategy,
    BaseDatasetSplitStrategyParams,
)
from models.resnet import ResNetEmbedding
from training.subjobs.train_shard import train_sharded_embedding_model
from training.subjobs.test_sisa_framework import test_sisa_framework
from training.utils import parse_int_list
from storage.dataset_splits import SISPADatasetSplitsStorage


@task(cache_policy=NO_CACHE)
def dataset_splits_task(
    dataset_split_strategy_function: DATASET_SPLIT_STRATEGY_FUNCTION,
    dataset_split_strategy_params: BaseDatasetSplitStrategyParams,
    dataset_splits_storage: SISPADatasetSplitsStorage,
):
    train_shard_indices, val_indices, test_indices = dataset_split_strategy_function(
        dataset_split_strategy_params
    )
    dataset_splits_storage.store_all_splits(
        train_shard_indices=train_shard_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )


@task(cache_policy=NO_CACHE)
def load_shard_dataloaders_task(
    dataset: Dataset,
    dataset_splits_storage: SISPADatasetSplitsStorage,
    train_batch_size: int,
    val_batch_size: int,
    test_batch_size: int,
    num_shards: int,
):
    train_shard_indices, train_indices, val_indices, test_indices = (
        dataset_splits_storage.retrieve_all_splits_indices()
    )
    full_train_dataloader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=train_batch_size,
        shuffle=True,
        pin_memory=True,
    )

    train_shard_dataloaders = [
        DataLoader(
            Subset(dataset, train_shard_indices[i]),
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=True,
        )
        for i in range(num_shards)
    ]
    val_dataloader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=val_batch_size,
        shuffle=True,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=test_batch_size,
        shuffle=True,
        pin_memory=True,
    )
    return (
        full_train_dataloader,
        train_shard_dataloaders,
        val_dataloader,
        test_dataloader,
    )


@task(cache_policy=NO_CACHE)
def train_shards_task(
    accelerator: Accelerator,
    train_shard_dataloaders: T.List[DataLoader],
    val_dataloader: DataLoader,
    val_check_interval_percentage: float,
    shard_optimizers: T.List[torch.optim.Optimizer],
    untrained_embedding_models: T.List[torch.nn.Module],
    classifiers: T.List[torch.nn.Module],
    loss_fn: torch.nn.Module,
    epochs: int,
    experiment_group_name: str,
    dataset_name: AVAILABLE_DATASETS,
):
    trained_embedding_models = []
    trained_classifiers = []
    for shard_idx, (
        untrained_embedding_model,
        classifier,
        shard_optimizer,
        train_shard_dataloader,
    ) in enumerate(
        zip(
            untrained_embedding_models,
            classifiers,
            shard_optimizers,
            train_shard_dataloaders,
        )
    ):
        (
            prepared_embedding_model,
            prepared_classifier,
            prepared_optimizer,
            prepared_train_dataloader,
            prepared_val_dataloader,
        ) = accelerator.prepare(
            untrained_embedding_model,
            classifier,
            shard_optimizer,
            train_shard_dataloader,
            val_dataloader,
        )

        trained_embedding_model, trained_classifier = train_sharded_embedding_model(
            accelerator=accelerator,
            dataset_name=dataset_name,
            prepared_embedding_model=prepared_embedding_model,
            prepared_classifier=prepared_classifier,
            prepared_optimizer=prepared_optimizer,
            prepared_train_dataloader=prepared_train_dataloader,
            prepared_val_dataloader=prepared_val_dataloader,
            loss_fn=loss_fn,
            shard_idx=shard_idx,
            val_check_interval_percentage=val_check_interval_percentage,
            epochs=epochs,
            experiment_group_name=experiment_group_name,
        )
        trained_embedding_models.append(trained_embedding_model)
        trained_classifiers.append(trained_classifier)

    return trained_embedding_models, trained_classifiers


@task(cache_policy=NO_CACHE)
def test_sisa_framework_task(
    accelerator: Accelerator,
    trained_embedding_models: T.List[torch.nn.Module],
    trained_classifiers: T.List[torch.nn.Module],
    test_dataloader: DataLoader,
    experiment_group_name: str,
    dataset_name: AVAILABLE_DATASETS,
):
    (
        prepared_trained_embedding_models,
        prepared_trained_classifiers,
        prepared_test_dataloader,
    ) = accelerator.prepare(
        trained_embedding_models,
        trained_classifiers,
        test_dataloader,
    )

    test_sisa_framework(
        accelerator=accelerator,
        prepared_trained_embedding_models=prepared_trained_embedding_models,
        prepared_trained_classifiers=prepared_trained_classifiers,
        prepared_test_dataloader=prepared_test_dataloader,
        experiment_group_name=experiment_group_name,
        dataset_name=dataset_name,
    )

    (
        prepared_trained_embedding_models,
        prepared_trained_classifiers,
        prepared_test_dataloader,
    ) = accelerator.clear(
        prepared_trained_embedding_models,
        prepared_trained_classifiers,
        prepared_test_dataloader,
    )


@click.command()
@click.option("--dataset-name", type=str, default=None)
@click.option("--train-batch-size", type=int, default=None)
@click.option("--val-batch-size", type=int, default=None)
@click.option("--test-batch-size", type=int, default=None)
@click.option("--num-workers", type=int, default=None)
@click.option(
    "--train-val-test-split",
    type=click.Tuple([float, float, float]),
    default=None,
)
@click.option("--num-shards", type=int, default=None)
@click.option("--class-informed-strategy-sampling-ratio", type=float, default=None)
@click.option("--resnet-block-dims", callback=parse_int_list, default=None)
@click.option("--resnet-num-modules-per-block", type=int, default=None)
@click.option("--accelerator", type=ACCELERATOR, default=None)
@click.option("--val-check-interval-percentage", type=float, default=None)
@click.option("--epochs", type=int, default=None)
@click.option("--devices", type=T.List[int], default=None)
@click.option("--accumulate-grad-batches", type=int, default=None)
@click.option("--optimizer-weight-decay", type=float, default=None)
@click.option("--optimizer-adam-beta1", type=float, default=None)
@click.option("--optimizer-adam-beta2", type=float, default=None)
@click.option("--optimizer-learning-rate", type=float, default=None)
@click.option("--seed", type=int, default=None)
@click.option("--storage-path", type=str, default=None)
@click.option(
    "--dataset-split-strategy",
    type=str,
    default=None,
)
@flow
def sisa_full_training(**kwargs):
    valid_fields = set(TrainingConfig.model_fields.keys())
    config_kwargs = {
        k: v for k, v in kwargs.items() if v is not None and k in valid_fields
    }
    training_config = TrainingConfig(**config_kwargs)
    model_config = training_config.get_model_config()
    dataset_config = training_config.get_dataset_config()
    optimizer_config = training_config.get_optimizer_config()
    finetuning_config = training_config.get_finetuning_config()

    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_group_name = f"sisa_full-{current_datetime}-{finetuning_config.epochs}_epochs-{model_config.resnet_block_dims}_rb_dims-{model_config.resnet_num_modules_per_block}_num_mod_block-{optimizer_config.optimizer_learning_rate}_lr-{optimizer_config.optimizer_weight_decay}_wd-{dataset_config.dataset_split_strategy}"

    try:
        dataset_split_strategy_enum = DatasetSplitStrategy(
            dataset_config.dataset_split_strategy
        )
    except ValueError:
        raise ValueError(
            f"Invalid dataset split strategy: {dataset_config.dataset_split_strategy}"
        )

    wandb_run = init_wandb_run(
        dataset_name=dataset_config.dataset_name,
        experiment_group_name=experiment_group_name,
        experiment_name="Experiment config",
        reinit=False,
    )
    wandb_run.config.update(training_config.model_dump())

    accelerator = Accelerator(
        gradient_accumulation_steps=training_config.accumulate_grad_batches
    )
    dataset_splits_storage = SISPADatasetSplitsStorage(
        storage_path=training_config.storage_path
    )

    embedding_dim = model_config.resnet_block_dims[-1]

    raw_dataset, num_channels, num_classes = choose_dataset(dataset_config.dataset_name)

    backbone_embedding_model = ResNetEmbedding(
        num_modules_per_block=model_config.resnet_num_modules_per_block,
        block_dims=model_config.resnet_block_dims,
        in_channels=num_channels,
    )
    backbone_classifier = torch.nn.Linear(embedding_dim, num_classes)
    untrained_embedding_models = [
        copy.deepcopy(backbone_embedding_model)
        for _ in range(dataset_config.num_shards)
    ]
    classifiers = [
        copy.deepcopy(backbone_classifier) for _ in range(dataset_config.num_shards)
    ]
    loss_fn = torch.nn.CrossEntropyLoss()
    shard_optimizers = [
        torch.optim.AdamW(
            list(embedding_model.parameters()) + list(classifier.parameters()),
            lr=optimizer_config.optimizer_learning_rate,
            betas=(
                optimizer_config.optimizer_adam_beta1,
                optimizer_config.optimizer_adam_beta2,
            ),
            weight_decay=optimizer_config.optimizer_weight_decay,
        )
        for embedding_model, classifier in zip(untrained_embedding_models, classifiers)
    ]

    dataset_split_strategy_function, dataset_split_strategy_params = (
        choose_dataset_split_strategy(
            dataset_split_strategy_enum,
            {**training_config.model_dump(), "dataset": raw_dataset},
        )
    )

    dataset_splits_task(
        dataset_split_strategy_function=dataset_split_strategy_function,
        dataset_split_strategy_params=dataset_split_strategy_params,
        dataset_splits_storage=dataset_splits_storage,
    )

    _, train_shard_dataloaders, val_dataloader, test_dataloader = (
        load_shard_dataloaders_task(
            dataset=raw_dataset,
            dataset_splits_storage=dataset_splits_storage,
            train_batch_size=training_config.train_batch_size,
            val_batch_size=training_config.val_batch_size,
            test_batch_size=training_config.test_batch_size,
            num_shards=dataset_config.num_shards,
        )
    )

    trained_embedding_models, trained_classifiers = train_shards_task(
        accelerator=accelerator,
        train_shard_dataloaders=train_shard_dataloaders,
        val_dataloader=val_dataloader,
        val_check_interval_percentage=finetuning_config.val_check_interval_percentage,
        shard_optimizers=shard_optimizers,
        untrained_embedding_models=untrained_embedding_models,
        classifiers=classifiers,
        loss_fn=loss_fn,
        epochs=finetuning_config.epochs,
        experiment_group_name=experiment_group_name,
        dataset_name=dataset_config.dataset_name,
    )

    test_sisa_framework_task(
        accelerator=accelerator,
        trained_embedding_models=trained_embedding_models,
        trained_classifiers=trained_classifiers,
        test_dataloader=test_dataloader,
        num_shards=dataset_config.num_shards,
        epochs=finetuning_config.epochs,
        experiment_group_name=experiment_group_name,
        dataset_name=dataset_config.dataset_name,
    )


if __name__ == "__main__":
    sisa_full_training()
