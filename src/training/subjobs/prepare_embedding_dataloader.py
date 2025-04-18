import torch
from torch.utils.data import DataLoader, TensorDataset
from storage.embeddings import SISPAEmbeddingStorage
from common.logger import logger


def prepare_aggregation_dataloader(
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
