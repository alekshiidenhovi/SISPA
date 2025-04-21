import torch
from torch.utils.data import DataLoader, TensorDataset
from storage.embeddings import SISPAEmbeddingStorage
from common.logger import logger
from common.types import TrainingStep


def load_aggregation_dataloader(
    embedding_storage: SISPAEmbeddingStorage,
    num_shards: int,
    batch_size: int,
    training_step: TrainingStep,
) -> DataLoader:
    all_embeddings = torch.tensor([], dtype=torch.float32)
    all_labels = torch.tensor([], dtype=torch.int64)

    for shard_idx in range(num_shards):
        embeddings, labels, _ = embedding_storage.retrieve_shard_embeddings(
            training_step=training_step, shard_idx=shard_idx
        )
        if embeddings is None:
            logger.warning(f"No embeddings found for shard {shard_idx}")
            continue

        all_embeddings = torch.cat((all_embeddings, torch.tensor(embeddings)), dim=0)
        all_labels = torch.cat((all_labels, torch.tensor(labels)), dim=0)

    embeddings_dataset = TensorDataset(all_embeddings, all_labels)
    embeddings_dataloader = DataLoader(
        embeddings_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    return embeddings_dataloader
