import torch
from torch.utils.data import DataLoader, TensorDataset
from storage.embeddings import SISPAEmbeddingStorage
from common.logger import logger
from common.types import TrainingStep


def load_aggregation_dataloader(
    embedding_storage: SISPAEmbeddingStorage,
    batch_size: int,
    training_step: TrainingStep,
) -> DataLoader:
    """
    Load a dataloader for the aggregation classifier from the embedding storage.

    Concatenates embedding across shards into a single wide embedding.
    """
    all_embeddings = torch.tensor([], dtype=torch.float32)
    all_labels = torch.tensor([], dtype=torch.int64)

    shards = embedding_storage.get_shards_for_step(training_step)
    if shards is None:
        raise ValueError(f"No shards found for training step {training_step}")

    for shard_idx in shards:
        embeddings, labels, _ = embedding_storage.retrieve_shard_embeddings(
            training_step=training_step, shard_idx=shard_idx
        )
        if embeddings is None:
            logger.warning(f"No embeddings found for shard {shard_idx}")
            continue

        if len(all_embeddings) == 0:
            all_embeddings = torch.tensor(embeddings, dtype=torch.float32)
            all_labels = torch.tensor(labels, dtype=torch.int64)
        else:
            all_embeddings = torch.cat(
                (all_embeddings, torch.tensor(embeddings)), dim=1
            )

    embeddings_dataset = TensorDataset(all_embeddings, all_labels)
    embeddings_dataloader = DataLoader(
        embeddings_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    return embeddings_dataloader
