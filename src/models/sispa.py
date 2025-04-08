import torch
import torch.nn as nn
from models.embedding_aggregator import EmbeddingAggregator


class SISPA(nn.Module):
    """SISA with pre-computed embedding aggregation layer.

    A neural network that combines multiple backbone models and aggregates their embeddings to make predictions.

    Parameters
    ----------
    backbone_models : nn.ModuleList
        List of backbone neural networks that generate embeddings
    embedding_dim : int
        Dimension of embeddings produced by each backbone model
    hidden_dim : int
        Hidden dimension used in the embedding aggregator
    num_shards : int
        Number of shards/backbone models being aggregated
    num_classes : int
        Number of output classes for classification

    Attributes
    ----------
    backbone_models : nn.ModuleList
        Stores the backbone models that generate embeddings
    embedding_aggregator : EmbeddingAggregator
        Neural network that aggregates embeddings from backbone models
    """

    def __init__(
        self,
        backbone_models: nn.ModuleList,
        embedding_dim: int,
        hidden_dim: int,
        num_shards: int,
        num_classes: int,
    ):
        super().__init__()
        self.backbone_models = backbone_models
        self.embedding_aggregator = EmbeddingAggregator(
            embedding_dim, hidden_dim, num_shards, num_classes
        )

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
        embeddings = [model(x) for model in self.backbone_models]
        concated_embeddings = torch.cat(embeddings, dim=1)
        return self.embedding_aggregator(concated_embeddings)
