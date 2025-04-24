import torch
import torch.nn as nn
from models.sispa.sispa_embedding_aggregator import SISPAEmbeddingAggregator
from models.sispa.sispa_sharded_embeddings import SISPAShardedEmbeddings


class SISPAFramework(nn.Module):
    """SISPA with pre-computed embedding aggregation layer.

    A neural network that combines multiple backbone models and aggregates their embeddings to make predictions.

    Args:
        embedding_models : nn.ModuleList
            List of backbone embedding models that generate embeddings
        embedding_aggregator : EmbeddingAggregator
            Neural network that aggregates embeddings from the embedding models
    """

    def __init__(
        self,
        sispa_sharded_embeddings: SISPAShardedEmbeddings,
        sispa_embedding_aggregator: SISPAEmbeddingAggregator,
    ):
        super().__init__()
        self.sispa_sharded_embeddings = sispa_sharded_embeddings
        self.sispa_embedding_aggregator = sispa_embedding_aggregator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SISPA model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width)

        Returns
        -------
        torch.Tensor
            Logits tensor of shape (batch_size, num_classes)
        """
        embeddings = [model(x) for model in self.sispa_sharded_embeddings]
        concatenated_embeddings = torch.cat(embeddings, dim=1)
        aggregated_embeddings = self.sispa_embedding_aggregator(concatenated_embeddings)
        return aggregated_embeddings
