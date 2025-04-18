import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingAggregator(nn.Module):
    """Neural network for aggregating embeddings from SISA submodels."""

    def __init__(
        self, embedding_dim: int, hidden_dim: int, num_shards: int, num_classes: int
    ):
        super().__init__()
        self.num_shards = num_shards
        self.aggregate = nn.Linear(embedding_dim * num_shards, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, embeddings: torch.Tensor):
        """Forward pass of the embedding aggregator.

        Args:
            embeddings : torch.Tensor
                Embeddings to aggregate. Shape is (N, embedding_dim * num_shards) where N is the number of samples in the batch, embedding_dim is the dimension of the embeddings produced by each shard, and num_shards is the number of shards.


        Returns:
            torch.Tensor
                Logits for the aggregated embeddings. Shape is (N, num_classes) where N is the number of samples in the batch and num_classes is the number of classes.
        """
        aggregated_embeddings = F.relu(self.aggregate(embeddings))
        logits = self.classifier(aggregated_embeddings)
        return logits
