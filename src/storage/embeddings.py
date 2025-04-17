import h5py
import torch
import numpy as np
import typing as T


class SISPAEmbeddingStorage:
    """Storage interface for SISPA embeddings.

    Manages storage and retrieval of embeddings using HDF5.

    Parameters
    ----------
    storage_path : str
        Path to the HDF5 file
    embedding_dim : int
        Dimension of embeddings
    create_new : bool, optional
        If True, creates a new storage file, by default False
    """

    def __init__(self, storage_path: str, embedding_dim: int):
        self.storage_path = storage_path
        self.embedding_dim = embedding_dim
        self.base_dir = "embeddings"
        self.file = h5py.File(storage_path, "a")

        if self.base_dir not in self.file:
            self.file.create_group(self.base_dir)

    def __del__(self):
        """Close the file when the object is deleted."""
        if hasattr(self, "file"):
            self.file.close()

    def _get_shard_group_name(self, shard_idx: int) -> str:
        """Internal helper method to get the name of a shard group."""
        return f"{self.base_dir}/shard_{shard_idx}"

    def _get_datapoint_path(self, shard_idx: int, datapoint_id: str) -> str:
        """Internal helper method to get the path of a datapoint."""
        return f"{self._get_shard_group_name(shard_idx)}/{datapoint_id}"

    def store_embeddings(
        self,
        shard_idx: int,
        datapoint_ids: T.List[str],
        embeddings: torch.Tensor,
    ):
        """Store embeddings for a specific shard and datapoints.

        Parameters
        ----------
        shard_idx : int
            Index of the shard
        datapoint_ids : list[str]
            ID of the datapoint
        embedding : torch.Tensor
            Embedding tensor to store
        """
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embeddings.shape[1]}"
            )

        if embeddings.shape[0] != len(datapoint_ids):
            raise ValueError(
                f"Number of datapoints mismatch: expected {len(datapoint_ids)}, got {embeddings.shape[0]}"
            )

        shard_group = self._get_shard_group_name(shard_idx)
        if shard_group not in self.file:
            self.file.create_group(shard_group)

        for idx in range(len(datapoint_ids)):
            datapoint_id = datapoint_ids[idx]
            embedding = embeddings[idx]
            datapoint_path = self._get_datapoint_path(shard_idx, datapoint_id)
            if datapoint_path in self.file:
                del self.file[datapoint_path]
            embedding_np = embedding.detach().cpu().numpy()
            self.file.create_dataset(
                datapoint_path,
                shape=embedding_np.shape,
                data=embedding_np,
                dtype=np.float32,
            )

    def retrieve_shard(self, shard_idx: int) -> T.Optional[T.Dict[int, torch.Tensor]]:
        """Retrieve all embeddings for a specific shard.

        Parameters
        ----------
        shard_idx : int
            Index of the shard

        Returns
        -------

        Dict[int, torch.Tensor]
            Dictionary mapping datapoint IDs to embeddings
        """
        shard_group = self._get_shard_group_name(shard_idx)
        if shard_group not in self.file:
            return None

        embedding_dict: T.Dict[int, torch.Tensor] = {}
        for datapoint_id in self.file[shard_group]:
            embedding_np = self.file[self._get_datapoint_path(shard_idx, datapoint_id)][
                ()
            ]
            embedding_dict[int(datapoint_id)] = torch.tensor(embedding_np)

        return embedding_dict

    def retrieve_embedding_by_id(
        self, shard_idx: int, datapoint_id: str
    ) -> torch.Tensor:
        """Retrieve an embedding by its ID.

        Parameters
        ----------
        shard_idx : int
            Index of the shard
        datapoint_id : str
            ID of the datapoint

        Returns
        -------
        torch.Tensor
            Embedding tensor
        """
        embedding_np = self.file[self._get_datapoint_path(shard_idx, datapoint_id)][()]
        return torch.tensor(embedding_np)

    def remove_shard(self, shard_idx: int) -> bool:
        """Remove a shard from storage.

        Parameters
        ----------
        shard_idx : int
            Index of the shard

        Returns
        -------
        bool
            True if removed successfully, False if not found
        """
        shard_group = self._get_shard_group_name(shard_idx)
        if shard_group in self.file:
            del self.file[shard_group]
            return True
        return False

    def remove_embedding(self, shard_idx: int, datapoint_id: str) -> bool:
        """Remove an embedding from storage.

        Parameters
        ----------
        shard_idx : int
            Index of the shard
        datapoint_id : str
            ID of the datapoint

        Returns
        -------
        bool
            True if removed successfully, False if not found
        """
        datapoint_path = self._get_datapoint_path(shard_idx, datapoint_id)
        if datapoint_path in self.file:
            del self.file[datapoint_path]
            return True
        return False
