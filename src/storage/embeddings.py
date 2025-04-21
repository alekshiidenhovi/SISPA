import h5py
import torch
import numpy as np
import typing as T
import os
from pydantic import BaseModel, Field
from common.types import TrainingStep


class EmbeddingData(BaseModel):
    label: int = Field(description="Label of the datapoint")
    embedding: T.List[float] = Field(
        description="Embedding of the datapoint in list format"
    )


class SISPAEmbeddingStorage:
    """Storage interface for SISPA embeddings.

    Manages storage and retrieval of embeddings using HDF5.

    Parameters
    ----------
    storage_path : str
        Path to the HDF5 file
    embedding_dim : int
        Dimension of embeddings
    """

    def __init__(self, storage_path: str, embedding_dim: int):
        self.storage_path = storage_path
        self.embedding_dim = embedding_dim
        self.base_file = "embeddings.hdf5"
        self.file = h5py.File(os.path.join(storage_path, self.base_file), "a")

    def __del__(self):
        """Close the file when the object is deleted."""
        if hasattr(self, "file"):
            self.file.close()

    def _get_shard_group_name(self, training_step: TrainingStep, shard_idx: int) -> str:
        """Internal helper method to get the name of a shard group."""
        return f"{training_step.value}/shard_{shard_idx}"

    def _get_shard_datapoint_path(
        self, training_step: TrainingStep, shard_idx: int, datapoint_id: str
    ) -> str:
        """Internal helper method to get the path of a datapoint."""
        return f"{self._get_shard_group_name(training_step, shard_idx)}/{datapoint_id}"

    def store_embeddings(
        self,
        training_step: TrainingStep,
        shard_idx: int,
        datapoint_ids: T.List[str],
        labels: torch.Tensor,
        embeddings: torch.Tensor,
    ):
        """Store embeddings for a specific shard and datapoints.

        Parameters
        ----------
        training_step : TrainingStep
            Training step
        shard_idx : int
            Index of the shard
        datapoint_ids : list[str]
            ID of the datapoint
        labels: torch.Tensor
            Labels of the datapoint
        embeddings : torch.Tensor
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

        if len(labels) != len(datapoint_ids):
            raise ValueError(
                f"Number of labels mismatch: expected {len(datapoint_ids)}, got {len(labels)}"
            )

        shard_group = self._get_shard_group_name(training_step, shard_idx)
        if shard_group not in self.file:
            self.file.create_group(shard_group)

        for idx in range(len(datapoint_ids)):
            datapoint_id = datapoint_ids[idx]
            label = int(labels[idx].detach().cpu().item())
            embedding_np = embeddings[idx].detach().cpu().numpy()
            datapoint_path = self._get_shard_datapoint_path(
                training_step, shard_idx, datapoint_id
            )
            if datapoint_path in self.file:
                del self.file[datapoint_path]

            dataset = self.file.create_dataset(
                datapoint_path,
                shape=embedding_np.shape,
                data=embedding_np,
                dtype=np.float32,
            )
            dataset.attrs["label"] = label

    def retrieve_shard_embeddings(
        self, training_step: TrainingStep, shard_idx: int
    ) -> T.Optional[T.Dict[str, EmbeddingData]]:
        """Retrieve all embeddings for a specific shard.

        Parameters
        ----------
        shard_idx : int
            Index of the shard

        Returns
        -------

        Dict[str, EmbeddingData]
            Dictionary mapping datapoint IDs to embeddings
        """
        shard_group = self._get_shard_group_name(training_step, shard_idx)
        if shard_group not in self.file:
            return None

        embedding_dict: T.Dict[str, EmbeddingData] = {}
        for datapoint_id in self.file[shard_group]:
            datapoint_path = self._get_shard_datapoint_path(
                training_step, shard_idx, datapoint_id
            )
            label = self.file[datapoint_path].attrs["label"]
            embedding_np = self.file[datapoint_path][()]
            embedding_dict[datapoint_id] = EmbeddingData(
                label=int(label[0]),
                embedding=embedding_np.tolist(),
            )

        return embedding_dict

    def remove_shard(self, training_step: TrainingStep, shard_idx: int) -> bool:
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
        shard_group = self._get_shard_group_name(training_step, shard_idx)
        if shard_group in self.file:
            del self.file[shard_group]
            return True
        return False

    def remove_embedding(
        self, training_step: TrainingStep, shard_idx: int, datapoint_id: str
    ) -> bool:
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
        datapoint_path = self._get_shard_datapoint_path(
            training_step, shard_idx, datapoint_id
        )
        if datapoint_path in self.file:
            del self.file[datapoint_path]
            return True
        return False
