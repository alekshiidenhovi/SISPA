import torch
import torch.nn as nn
from models.embedding_aggregator import EmbeddingAggregator


class SISPA(nn.Module):
    """SISA with pre-computed embedding aggregation layer.

    A neural network that combines multiple backbone models and aggregates their embeddings to make predictions.

    Args:
        embedding_models : nn.ModuleList
            List of backbone embedding models that generate embeddings
        embedding_aggregator : EmbeddingAggregator
            Neural network that aggregates embeddings from the embedding models
    """

    def __init__(
        self,
        embedding_models: nn.ModuleList,
        embedding_aggregator: EmbeddingAggregator,
    ):
        super().__init__()
        self.embedding_models = embedding_models
        self.embedding_aggregator = embedding_aggregator

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
        embeddings = [model(x) for model in self.embedding_models]
        concated_embeddings = torch.cat(embeddings, dim=1)
        return self.embedding_aggregator(concated_embeddings)
