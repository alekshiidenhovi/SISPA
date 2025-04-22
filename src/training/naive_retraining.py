import typing as T
import torch
import wandb
import uuid
import click
import copy
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from prefect import flow, task
from prefect.cache_policies import NO_CACHE
import wandb.wandb_run
from common.tracking import init_wandb_run
from common.types import TrainingStep
from models.resnet import ResNet
from models.sispa import (
    SISPAShardedEmbeddings,
    SISPAEmbeddingAggregator,
)
from training.subjobs.precompute_embeddings import precompute_embeddings
from training.subjobs.train_aggregation import train_aggregation_classifier
from training.subjobs.train_shard import train_sharded_embedding_model
from training.subjobs.create_dataset_splits import create_dataset_splits
from training.subjobs.load_aggregation_dataloader import load_aggregation_dataloader
from training.subjobs.collect_labels import collect_labels
from storage.dataset_splits import SISPADatasetSplitsStorage
from storage.embeddings import SISPAEmbeddingStorage
from storage.models import SISPAModelStorage


@task(cache_policy=NO_CACHE)
def dataset_splits_task(
    dataset: Dataset,
    dataset_splits_storage: SISPADatasetSplitsStorage,
    train_val_test_split: T.Tuple[float, float, float],
    sampling_ratio: float,
    seed: int,
):
    train_shard_indices, val_indices, test_indices = create_dataset_splits(
        dataset=dataset,
        train_val_test_split=train_val_test_split,
        sampling_ratio=sampling_ratio,
        seed=seed,
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
    val_batch_interval: int,
    shard_optimizers: T.List[torch.optim.Optimizer],
    untrained_embedding_models: T.List[torch.nn.Module],
    classifiers: T.List[torch.nn.Module],
    loss_fn: torch.nn.Module,
    epochs: int,
    wandb_run: wandb.wandb_run.Run,
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
            prepared_embedding_model=prepared_embedding_model,
            prepared_classifier=prepared_classifier,
            prepared_optimizer=prepared_optimizer,
            prepared_train_dataloader=prepared_train_dataloader,
            prepared_val_dataloader=prepared_val_dataloader,
            loss_fn=loss_fn,
            shard_idx=shard_idx,
            val_batch_interval=val_batch_interval,
            epochs=epochs,
            wandb_run=wandb_run,
        )

        model_storage.save_sharded_model(
            sharded_model=trained_embedding_model,
            shard_id=f"shard_{shard_idx}",
            wandb_run=wandb_run,
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
    val_batch_interval: int,
    epochs: int,
    model_storage: SISPAModelStorage,
    wandb_run: wandb.wandb_run.Run,
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
        val_batch_interval=val_batch_interval,
        epochs=epochs,
    )

    model_storage.save_aggregator_model(
        aggregator_model=trained_aggregator,
        wandb_run=wandb_run,
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
@click.option("--epochs", type=int, default=1)
@click.option("--num-shards", type=int, default=10)
@click.option(
    "--train-val-test-split",
    type=click.Tuple([float, float, float]),
    default=(0.8, 0.1, 0.1),
)
@click.option("--sampling-ratio", type=float, default=0.5)
@click.option("--seed", type=int, default=42)
@click.option("--train-batch-size", type=int, default=64)
@click.option("--val-batch-size", type=int, default=64)
@click.option("--test-batch-size", type=int, default=64)
@click.option("--embedding-dim", type=int, default=64)
@click.option("--hidden-dim", type=int, default=64)
@click.option("--num-blocks", type=int, default=3)
@click.option("--learning-rate", type=float, default=1e-4)
@click.option("--weight-decay", type=float, default=1e-4)
@click.option("--storage-path", type=str)
@click.option("--val-batch-interval", type=int, default=50)
@flow
def naive_retraining(
    epochs: int,
    num_shards: int,
    train_val_test_split: T.Tuple[float, float, float],
    sampling_ratio: float,
    seed: int,
    train_batch_size: int,
    val_batch_size: int,
    test_batch_size: int,
    embedding_dim: int,
    hidden_dim: int,
    num_blocks: int,
    learning_rate: float,
    weight_decay: float,
    storage_path: str,
    val_batch_interval: int,
):
    wandb_run = init_wandb_run()

    accelerator = Accelerator()
    model_storage = SISPAModelStorage(storage_path=storage_path)
    dataset_splits_storage = SISPADatasetSplitsStorage(storage_path=storage_path)
    embedding_storage = SISPAEmbeddingStorage(
        storage_path=storage_path, embedding_dim=embedding_dim
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    raw_dataset = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    num_classes = len(raw_dataset.targets.unique())

    embedding_model = ResNet(num_blocks=num_blocks, embedding_dim=embedding_dim)
    classifier = torch.nn.Linear(embedding_dim, num_classes)
    untrained_embedding_models = [
        copy.deepcopy(embedding_model) for _ in range(num_shards)
    ]
    classifiers = [copy.deepcopy(classifier) for _ in range(num_shards)]
    loss_fn = torch.nn.CrossEntropyLoss()
    shard_optimizers = [
        torch.optim.AdamW(
            list(embedding_model.parameters()) + list(classifier.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        for embedding_model, classifier in zip(untrained_embedding_models, classifiers)
    ]

    aggregator = SISPAEmbeddingAggregator(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_shards=num_shards,
        num_classes=num_classes,
    )
    aggregator_optimizer = torch.optim.AdamW(
        list(aggregator.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    dataset_splits_task(
        dataset=raw_dataset,
        dataset_splits_storage=dataset_splits_storage,
        train_val_test_split=train_val_test_split,
        sampling_ratio=sampling_ratio,
        seed=seed,
    )

    full_train_dataloader, train_shard_dataloaders, val_dataloader, test_dataloader = (
        load_shard_dataloaders_task(
            dataset=raw_dataset,
            dataset_splits_storage=dataset_splits_storage,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            num_shards=num_shards,
        )
    )

    train_shards_task(
        accelerator=accelerator,
        model_storage=model_storage,
        train_shard_dataloaders=train_shard_dataloaders,
        val_dataloader=val_dataloader,
        val_batch_interval=val_batch_interval,
        shard_optimizers=shard_optimizers,
        untrained_embedding_models=untrained_embedding_models,
        classifiers=classifiers,
        loss_fn=loss_fn,
        epochs=epochs,
        wandb_run=wandb_run,
    )

    precompute_embeddings_task(
        accelerator=accelerator,
        embedding_model=embedding_model,
        model_storage=model_storage,
        embedding_storage=embedding_storage,
        full_train_dataloader=full_train_dataloader,
        val_dataloader=val_dataloader,
        num_shards=num_shards,
    )

    training_embeddings_dataloader, validation_embeddings_dataloader = (
        load_aggregation_dataloaders_task(
            embedding_storage=embedding_storage,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
        )
    )

    train_aggregator_task(
        accelerator=accelerator,
        aggregator=aggregator,
        aggregator_optimizer=aggregator_optimizer,
        training_embeddings_dataloader=training_embeddings_dataloader,
        validation_embeddings_dataloader=validation_embeddings_dataloader,
        loss_fn=loss_fn,
        val_batch_interval=val_batch_interval,
        epochs=epochs,
        model_storage=model_storage,
        wandb_run=wandb_run,
    )


if __name__ == "__main__":
    naive_retraining()
