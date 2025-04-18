import torch
import torch.nn as nn
import typing as T


class SISPAShardedEmbeddings(nn.Module):
    """Manages multiple embedding models for the SISPA framework.

    This module encapsulates multiple embedding models (shards) and provides
    a unified interface for computing embeddings across all shards.

    Parameters
    ----------
    embedding_models : list[nn.Module]
        List of embedding models, one for each shard
    """

    def __init__(self, embedding_models: T.List[nn.Module]):
        super().__init__()
        self.embedding_models = nn.ModuleList(embedding_models)
        self.num_shards = len(embedding_models)

    def __getitem__(self, idx: int) -> nn.Module:
        """Get a specific embedding model by index.

        Parameters
        ----------
        idx : int
            Index of the embedding model to retrieve

        Returns
        -------
        nn.Module
            The requested embedding model
        """
        if idx < 0 or idx >= self.num_shards:
            raise IndexError(f"Index {idx} out of range for {self.num_shards} shards")
        return self.embedding_models[idx]

    def __iter__(self):
        """Iterate over all embedding models.

        Returns
        -------
        iterator
            Iterator over embedding models
        """
        return iter(self.embedding_models)

    def __len__(self) -> int:
        """Get the number of embedding models.

        Returns
        -------
        int
            Number of embedding models
        """
        return self.num_shards

    def forward(self, x: torch.Tensor) -> T.List[torch.Tensor]:
        """Compute embeddings for input data using all shards.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width)

        Returns
        -------
        list[torch.Tensor]
            List of embedding tensors, one from each shard,
            each of shape (batch_size, embedding_dim)
        """
        return [model(x) for model in self.embedding_models]

    def compute_embeddings_by_shard(
        self, x: torch.Tensor, shard_idx: int
    ) -> torch.Tensor:
        """Compute embeddings for input data using a specific shard.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width)
        shard_idx : int
            Index of the shard to use for computing embeddings

        Returns
        -------
        torch.Tensor
            Embedding tensor of shape (batch_size, embedding_dim)
        """
        if shard_idx < 0 or shard_idx >= self.num_shards:
            raise IndexError(
                f"Shard index {shard_idx} out of range for {self.num_shards} shards"
            )
        return self.embedding_models[shard_idx](x)
