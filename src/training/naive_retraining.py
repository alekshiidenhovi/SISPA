import typing as T
import torch
import uuid
import click
import copy
import datetime
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset, Dataset
from prefect import flow, task
from prefect.cache_policies import NO_CACHE
from common.types import (
    TrainingStep,
    DATASET_SPLIT_STRATEGY_FUNCTION,
    DatasetSplitStrategy,
    AVAILABLE_DATASETS,
)
from common.config import TrainingConfig
from common.tracking import init_wandb_run
from datasets.choose_dataset import choose_dataset
from datasets.choose_dataset_split_strategy import (
    choose_dataset_split_strategy,
    BaseDatasetSplitStrategyParams,
)
from models.resnet import ResNet
from models.sispa import (
    SISPAShardedEmbeddings,
    SISPAEmbeddingAggregator,
)
from training.subjobs.precompute_embeddings import precompute_embeddings
from training.subjobs.train_aggregation import train_aggregation_classifier
from training.subjobs.train_shard import train_sharded_embedding_model
from training.subjobs.load_aggregation_dataloader import load_aggregation_dataloader
from training.subjobs.collect_labels import collect_labels
from storage.dataset_splits import SISPADatasetSplitsStorage
from storage.embeddings import SISPAEmbeddingStorage
from storage.models import SISPAModelStorage


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
    model_storage: SISPAModelStorage,
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

        trained_embedding_model = train_sharded_embedding_model(
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

        model_storage.save_sharded_model(
            sharded_model=trained_embedding_model,
            shard_id=f"shard_{shard_idx}",
            experiment_group_name=experiment_group_name,
            dataset_name=dataset_name,
        )

        (
            prepared_embedding_model,
            prepared_classifier,
            prepared_optimizer,
            prepared_train_dataloader,
            prepared_val_dataloader,
        ) = accelerator.clear(
            prepared_embedding_model,
            prepared_classifier,
            prepared_optimizer,
            prepared_train_dataloader,
            prepared_val_dataloader,
        )


@task(cache_policy=NO_CACHE)
def precompute_embeddings_task(
    accelerator: Accelerator,
    embedding_model: torch.nn.Module,
    model_storage: SISPAModelStorage,
    embedding_storage: SISPAEmbeddingStorage,
    full_train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_shards: int,
):
    trained_sispa_models = []
    for shard_idx in range(num_shards):
        sispa_embedding_model = model_storage.load_sharded_model_local(
            sharded_model=embedding_model,
            shard_id=f"shard_{shard_idx}",
        )
        trained_sispa_models.append(sispa_embedding_model)
    trained_sispa_sharded_embedding_models = SISPAShardedEmbeddings(
        trained_sispa_models
    )

    all_training_labels = collect_labels(full_train_dataloader)
    all_validation_labels = collect_labels(val_dataloader)

    training_datapoint_ids = [str(uuid.uuid4()) for _ in all_training_labels]
    validation_datapoint_ids = [str(uuid.uuid4()) for _ in all_validation_labels]

    embedding_storage.initialize_datapoint_ids(
        training_step=TrainingStep.TRAINING,
        datapoint_ids=training_datapoint_ids,
    )
    embedding_storage.initialize_datapoint_ids(
        training_step=TrainingStep.VALIDATION,
        datapoint_ids=validation_datapoint_ids,
    )

    for shard_idx, sispa_embedding_model in enumerate(
        trained_sispa_sharded_embedding_models
    ):
        training_embeddings, training_labels = precompute_embeddings(
            accelerator=accelerator,
            trained_model=sispa_embedding_model,
            dataloader=full_train_dataloader,
            shard_idx=shard_idx,
            stage=TrainingStep.TRAINING,
        )

        validation_embeddings, validation_labels = precompute_embeddings(
            accelerator=accelerator,
            trained_model=sispa_embedding_model,
            dataloader=val_dataloader,
            shard_idx=shard_idx,
            stage=TrainingStep.VALIDATION,
        )

        embedding_storage.store_shard_embeddings(
            training_step=TrainingStep.TRAINING,
            shard_idx=shard_idx,
            labels=training_labels,
            embeddings=training_embeddings,
        )

        embedding_storage.store_shard_embeddings(
            training_step=TrainingStep.VALIDATION,
            shard_idx=shard_idx,
            labels=validation_labels,
            embeddings=validation_embeddings,
        )


@task(cache_policy=NO_CACHE)
def load_aggregation_dataloaders_task(
    embedding_storage: SISPAEmbeddingStorage,
    train_batch_size: int,
    val_batch_size: int,
):
    training_embeddings_dataloader = load_aggregation_dataloader(
        embedding_storage=embedding_storage,
        batch_size=train_batch_size,
        training_step=TrainingStep.TRAINING,
    )
    validation_embeddings_dataloader = load_aggregation_dataloader(
        embedding_storage=embedding_storage,
        batch_size=val_batch_size,
        training_step=TrainingStep.VALIDATION,
    )
    return training_embeddings_dataloader, validation_embeddings_dataloader


@task(cache_policy=NO_CACHE)
def train_aggregator_task(
    accelerator: Accelerator,
    aggregator: SISPAEmbeddingAggregator,
    aggregator_optimizer: torch.optim.Optimizer,
    training_embeddings_dataloader: DataLoader,
    validation_embeddings_dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    val_check_interval_percentage: float,
    epochs: int,
    model_storage: SISPAModelStorage,
    experiment_group_name: str,
    dataset_name: AVAILABLE_DATASETS,
):
    (
        prepared_aggregator,
        prepared_aggregator_optimizer,
        prepared_training_embeddings_dataloader,
        prepared_validation_embeddings_dataloader,
    ) = accelerator.prepare(
        aggregator,
        aggregator_optimizer,
        training_embeddings_dataloader,
        validation_embeddings_dataloader,
    )

    trained_aggregator = train_aggregation_classifier(
        accelerator=accelerator,
        prepared_aggregator=prepared_aggregator,
        prepared_optimizer=prepared_aggregator_optimizer,
        prepared_training_embeddings_dataloader=prepared_training_embeddings_dataloader,
        prepared_validation_embeddings_dataloader=prepared_validation_embeddings_dataloader,
        loss_fn=loss_fn,
        val_check_interval_percentage=val_check_interval_percentage,
        epochs=epochs,
        experiment_group_name=experiment_group_name,
        dataset_name=dataset_name,
    )

    model_storage.save_aggregator_model(
        aggregator_model=trained_aggregator,
        experiment_group_name=experiment_group_name,
        dataset_name=dataset_name,
    )

    (
        prepared_aggregator,
        prepared_aggregator_optimizer,
        prepared_training_embeddings_dataloader,
        prepared_validation_embeddings_dataloader,
    ) = accelerator.clear(
        prepared_aggregator,
        prepared_aggregator_optimizer,
        prepared_training_embeddings_dataloader,
        prepared_validation_embeddings_dataloader,
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
@click.option("--backbone-embedding-dim", type=int, default=None)
@click.option("--resnet-num-blocks", type=int, default=None)
@click.option("--aggregator-hidden-dim", type=int, default=None)
@click.option("--accelerator", type=str, default=None)
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
def naive_retraining(**kwargs):
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
    experiment_group_name = f"naive-rt-{current_datetime}-{dataset_config.num_shards}_shards-{finetuning_config.epochs}_epochs-{model_config.backbone_embedding_dim}_embed_dim-{model_config.resnet_num_blocks}_num_blocks-{model_config.aggregator_hidden_dim}_hidden_dim-{optimizer_config.optimizer_learning_rate}_lr-{optimizer_config.optimizer_weight_decay}_wd"

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
    model_storage = SISPAModelStorage(storage_path=training_config.storage_path)
    dataset_splits_storage = SISPADatasetSplitsStorage(
        storage_path=training_config.storage_path
    )
    embedding_storage = SISPAEmbeddingStorage(
        storage_path=training_config.storage_path,
        embedding_dim=model_config.backbone_embedding_dim,
    )

    raw_dataset, num_channels, num_classes = choose_dataset(dataset_config.dataset_name)

    backbone_embedding_model = ResNet(
        num_blocks=model_config.resnet_num_blocks,
        embedding_dim=model_config.backbone_embedding_dim,
        in_channels=num_channels,
    )
    backbone_classifier = torch.nn.Linear(
        model_config.backbone_embedding_dim, num_classes
    )
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

    aggregator = SISPAEmbeddingAggregator(
        embedding_dim=model_config.backbone_embedding_dim,
        hidden_dim=model_config.aggregator_hidden_dim,
        num_shards=dataset_config.num_shards,
        num_classes=num_classes,
    )
    aggregator_optimizer = torch.optim.AdamW(
        list(aggregator.parameters()),
        lr=optimizer_config.optimizer_learning_rate,
        betas=(
            optimizer_config.optimizer_adam_beta1,
            optimizer_config.optimizer_adam_beta2,
        ),
        weight_decay=optimizer_config.optimizer_weight_decay,
    )

    try:
        dataset_split_strategy_enum = DatasetSplitStrategy(
            dataset_config.dataset_split_strategy
        )
    except ValueError:
        raise ValueError(
            f"Invalid dataset split strategy: {dataset_config.dataset_split_strategy}"
        )

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

    full_train_dataloader, train_shard_dataloaders, val_dataloader, test_dataloader = (
        load_shard_dataloaders_task(
            dataset=raw_dataset,
            dataset_splits_storage=dataset_splits_storage,
            train_batch_size=training_config.train_batch_size,
            val_batch_size=training_config.val_batch_size,
            test_batch_size=training_config.test_batch_size,
            num_shards=dataset_config.num_shards,
        )
    )

    train_shards_task(
        accelerator=accelerator,
        model_storage=model_storage,
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

    precompute_embeddings_task(
        accelerator=accelerator,
        embedding_model=backbone_embedding_model,
        model_storage=model_storage,
        embedding_storage=embedding_storage,
        full_train_dataloader=full_train_dataloader,
        val_dataloader=val_dataloader,
        num_shards=dataset_config.num_shards,
    )

    training_embeddings_dataloader, validation_embeddings_dataloader = (
        load_aggregation_dataloaders_task(
            embedding_storage=embedding_storage,
            train_batch_size=training_config.train_batch_size,
            val_batch_size=training_config.val_batch_size,
        )
    )

    train_aggregator_task(
        accelerator=accelerator,
        aggregator=aggregator,
        aggregator_optimizer=aggregator_optimizer,
        training_embeddings_dataloader=training_embeddings_dataloader,
        validation_embeddings_dataloader=validation_embeddings_dataloader,
        loss_fn=loss_fn,
        val_check_interval_percentage=finetuning_config.val_check_interval_percentage,
        epochs=finetuning_config.epochs,
        model_storage=model_storage,
        experiment_group_name=experiment_group_name,
        dataset_name=dataset_config.dataset_name,
    )


if __name__ == "__main__":
    naive_retraining()
