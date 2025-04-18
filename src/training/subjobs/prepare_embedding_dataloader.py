import torch
from torch.utils.data import DataLoader, TensorDataset
from storage.embeddings import SISPAEmbeddingStorage
from models.sispa import SISPAShardedEmbeddings
from common.logger import logger
from accelerate import Accelerator
from tqdm import tqdm


def prepare_aggregation_training_dataloader(
    embeddings_storage: SISPAEmbeddingStorage,
    num_shards: int,
    batch_size: int,
    device: torch.device,
) -> DataLoader:
    all_embeddings = torch.tensor([], device=device, dtype=torch.float32)
    all_labels = torch.tensor([], device=device, dtype=torch.long)

    for shard_idx in range(num_shards):
        shard_embeddings_dict = embeddings_storage.retrieve_shard(shard_idx)
        if shard_embeddings_dict is None:
            logger.warning(f"No embeddings found for shard {shard_idx}")
            continue

        for _, embedding_data in shard_embeddings_dict.items():
            all_embeddings = torch.cat(
                (all_embeddings, embedding_data.embedding), dim=0
            )
            all_labels = torch.cat((all_labels, embedding_data.label), dim=0)

    embeddings_dataset = TensorDataset(all_embeddings, all_labels)
    embeddings_dataloader = DataLoader(
        embeddings_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    return embeddings_dataloader


def prepare_aggregation_validation_dataloader(
    accelerator: Accelerator,
    prepared_sispa_embedding_models: SISPAShardedEmbeddings,
    prepared_validation_dataloader: DataLoader,
) -> DataLoader:
    """Prepare a validation dataloader for the aggregation step by computing embeddings
    from all shards and collecting them into a dataloader with respective labels.

    Parameters
    ----------
    accelerator : Accelerator
        Accelerator to use for computing embeddings
    prepared_sispa_embedding_models : SISPAShardedEmbeddings
        Prepared embedding models for all shards
    prepared_validation_dataloader : DataLoader
        Prepared validation dataloader containing images and labels

    Returns
    -------
    DataLoader
        Dataloader containing concatenated embeddings and labels for validation
    """
    all_embeddings = torch.tensor([], device=accelerator.device, dtype=torch.float32)
    all_labels = torch.tensor([], device=accelerator.device, dtype=torch.long)

    for model in prepared_sispa_embedding_models:
        model.eval()

    validation_progress_bar = tqdm(
        prepared_validation_dataloader,
        desc="Computing validation embeddings for aggregation ",
    )
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(validation_progress_bar):
            validation_progress_bar.set_description(
                f"Computing validation embeddings for shard {batch_idx}"
            )

            shard_embeddings = torch.tensor(
                [], device=accelerator.device, dtype=torch.float32
            )
            for shard_idx in range(len(prepared_sispa_embedding_models)):
                with accelerator.autocast():
                    embedding = prepared_sispa_embedding_models[shard_idx](images)
                    shard_embeddings = torch.cat((shard_embeddings, embedding), dim=0)

            all_embeddings = torch.cat((all_embeddings, shard_embeddings), dim=0)
            all_labels = torch.cat((all_labels, labels), dim=0)

    validation_dataset = TensorDataset(all_embeddings, all_labels)
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=prepared_validation_dataloader.batch_size,
        shuffle=False,
    )

    return validation_dataloader
