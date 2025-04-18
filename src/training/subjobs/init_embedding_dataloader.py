import torch
from torch.utils.data import DataLoader, TensorDataset
from storage.embeddings import SISPAEmbeddingStorage
from common.logger import logger


def init_embedding_dataloader(
    embeddings_storage: SISPAEmbeddingStorage,
    num_shards: int,
    batch_size: int,
    device: torch.device,
):
    all_embeddings = []
    all_labels = []

    for shard_idx in range(num_shards):
        shard_embeddings_dict = embeddings_storage.retrieve_shard(shard_idx)
        if shard_embeddings_dict is None:
            logger.warning(f"No embeddings found for shard {shard_idx}")
            continue

        for _, embedding_data in shard_embeddings_dict.items():
            all_embeddings.append(embedding_data.embedding)
            all_labels.append(embedding_data.label)

    embeddings_tensor = torch.stack(all_embeddings).to(device)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long).to(device)

    embeddings_dataset = TensorDataset(embeddings_tensor, labels_tensor)
    embeddings_dataloader = DataLoader(
        embeddings_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    return embeddings_dataloader
