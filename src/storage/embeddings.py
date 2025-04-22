import h5py
import torch
import numpy as np
import typing as T
import os
from common.types import TrainingStep
from common.logger import logger


class SISPAEmbeddingStorage:
    """Storage for SISPA embeddings with single embedding removal capability.

    Manages storage and retrieval using HDF5, storing data in bulk per shard.
    Supports marking individual embeddings for removal across all shards via a
    validity mask ("lazy deletion"). Includes optional compaction.

    Parameters:
    ----------
    storage_path : str
        Path to the directory where the HDF5 file will be stored.
    embedding_dim : int
        Dimension of embeddings.
    file_name : str, optional
        Name of the HDF5 file, by default "embeddings_modifiable.hdf5".
    compression : str or None, optional
        Compression filter (e.g., 'gzip'). Default 'gzip'.
    compression_opts : int or None, optional
        Compression level (e.g., 4 for gzip). Default 4.
    chunk_size : int, optional
        Chunk size for the first dimension of datasets. Default 128.
    """

    def __init__(
        self,
        storage_path: str,
        embedding_dim: int,
        file_name: str = "embeddings.hdf5",
        compression: T.Optional[str] = "gzip",
        compression_opts: T.Optional[int] = 4,
        chunk_size: int = 128,
    ):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)

        self.embedding_dim = embedding_dim
        self.base_file = file_name
        self.file_path = os.path.join(storage_path, self.base_file)
        self.file = h5py.File(self.file_path, "a")
        self.compression = compression
        self.compression_opts = compression_opts
        self.chunk_shape_embeddings = (chunk_size, self.embedding_dim)
        self.chunk_shape_vector = (chunk_size,)

        self.required_shard_datasets = ["embeddings", "labels"]
        self.ids_dataset_name = "ids"
        self.mask_dataset_name = "valid_mask"

    def __del__(self):
        """Close the file when the object is deleted."""
        self.file.close()

    def _get_split_group_path(self, training_step: TrainingStep) -> str:
        """Get the HDF5 path for a split group."""
        return f"{training_step.value}"

    def _get_shard_group_path(self, training_step: TrainingStep, shard_idx: int) -> str:
        """Get the HDF5 path for a shard group."""
        return f"{self._get_split_group_path(training_step)}/{shard_idx}"

    def _get_ids_path_for_step(self, training_step: TrainingStep) -> str:
        """Get the HDF5 path for the IDs dataset for a specific TrainingStep."""
        return f"{self._get_split_group_path(training_step)}/{self.ids_dataset_name}"

    def _get_mask_path_for_step(self, training_step: TrainingStep) -> str:
        """Get the HDF5 path for the validity mask dataset for a specific TrainingStep."""
        return f"{self._get_split_group_path(training_step)}/{self.mask_dataset_name}"

    def initialize_datapoint_ids(
        self, training_step: TrainingStep, datapoint_ids: T.List[str]
    ):
        """Initialize stepwise ID and mask datasets.

        If the datasets already exist, they are deleted and recreated.

        Args:
            training_step : TrainingStep
                Training step to initialize
            datapoint_ids : List[str]
                List of datapoint IDs
        """
        split_group_path = self._get_split_group_path(training_step)
        split_group = self.file.require_group(split_group_path)

        num_datapoints = len(datapoint_ids)
        logger.info(
            f"Initializing store for TrainingStep '{training_step.value}' with {num_datapoints} datapoint IDs."
        )

        ids_np = np.array(datapoint_ids, dtype=h5py.string_dtype(encoding="utf-8"))
        valid_mask_np = np.ones(num_datapoints, dtype=bool)

        if self.ids_dataset_name in split_group:
            del split_group[self.ids_dataset_name]
        if self.mask_dataset_name in split_group:
            del split_group[self.mask_dataset_name]

        split_group.create_dataset(
            self.ids_dataset_name,
            data=ids_np,
            chunks=self.chunk_shape_vector,
            compression=self.compression,
            compression_opts=self.compression_opts,
        )
        split_group.create_dataset(
            self.mask_dataset_name,
            data=valid_mask_np,
            chunks=self.chunk_shape_vector,
            compression=self.compression,
            compression_opts=self.compression_opts,
        )
        self.file.flush()

    def _get_split_datapoint_ids(
        self, training_step: TrainingStep
    ) -> T.Optional[T.List[str]]:
        """Retrieve the split datapoint IDs."""
        ids_path = self._get_ids_path_for_step(training_step)
        if ids_path in self.file:
            ids_bytes = self.file[ids_path][()]
            return [id_b.decode("utf-8") for id_b in ids_bytes]
        return None

    def _get_step_mask(self, training_step: TrainingStep) -> T.Optional[np.ndarray]:
        """Retrieve the global validity mask."""
        mask_path = self._get_mask_path_for_step(training_step)
        if mask_path in self.file:
            return self.file[mask_path][()]
        return None

    def _get_all_training_steps(self) -> T.Optional[T.List[TrainingStep]]:
        steps = []
        if not self.file:
            return None
        for key in self.file.keys():
            if isinstance(self.file[key], h5py.Group):
                try:
                    step = TrainingStep(key)
                    steps.append(step)
                except ValueError:
                    logger.warning(
                        f"Found group key '{key}' in HDF5 root which is not a valid TrainingStep enum member."
                    )
        return steps

    def get_shards_for_step(
        self, training_step: TrainingStep
    ) -> T.Optional[T.List[int]]:
        """List all shard indices available for a given training step."""
        split_group_path = self._get_split_group_path(training_step)
        if split_group_path not in self.file:
            logger.warning(f"No shards found for TrainingStep '{training_step.value}'.")
            return None

        split_group = self.file[split_group_path]
        shard_indices = []
        for group_name in split_group:
            try:
                shard_idx = int(group_name)
                shard_path = self._get_shard_group_path(training_step, shard_idx)
                if isinstance(self.file.get(shard_path), h5py.Group):
                    shard_indices.append(shard_idx)
            except Exception:
                logger.warning(
                    f"Found potentially non-standard group name: {group_name} under {split_group_path}"
                )
        return sorted(shard_indices)

    def store_shard_embeddings(
        self,
        training_step: TrainingStep,
        shard_idx: int,
        labels: torch.Tensor,
        embeddings: torch.Tensor,
    ):
        """Store embeddings and labels for a shard. Assumes order matches global IDs.

        If global IDs are not initialized (first call), `global_datapoint_ids_for_init`
        MUST be provided to set up the global store. Subsequent calls MUST NOT provide it.

        OVERWRITES existing data for the specific shard (training_step, shard_idx).
        """
        split_group_path = self._get_split_group_path(training_step)
        ids_path = self._get_ids_path_for_step(training_step)
        split_group = self.file.require_group(split_group_path)
        if self.ids_dataset_name not in split_group:
            raise RuntimeError(
                f"IDs not initialized for TrainingStep '{training_step.value}'."
            )

        num_input_datapoints, input_embedding_dim = embeddings.shape
        num_step_datapoints = len(self.file[ids_path])
        if num_input_datapoints != num_step_datapoints:
            raise ValueError(
                f"Number of embeddings ({num_input_datapoints}) does not match "
                f"the number of global datapoint IDs ({num_step_datapoints}). "
                f"Ensure data corresponds to the globally stored order."
            )
        if labels.shape[0] != num_step_datapoints:
            raise ValueError(
                f"Number of labels ({labels.shape[0]}) does not match global count ({num_step_datapoints})."
            )
        if input_embedding_dim != self.embedding_dim:
            raise ValueError(
                f"Incorrect embedding dimension: expected {self.embedding_dim}, got {input_embedding_dim}"
            )

        embeddings_np = embeddings.detach().cpu().numpy().astype(np.float32)
        labels_np = labels.detach().cpu().numpy().astype(np.int64)

        shard_group_path = self._get_shard_group_path(training_step, shard_idx)
        if shard_group_path in self.file:
            del self.file[shard_group_path]
        group = self.file.create_group(shard_group_path)

        group.create_dataset(
            "embeddings",
            data=embeddings_np,
            chunks=self.chunk_shape_embeddings,
            compression=self.compression,
            compression_opts=self.compression_opts,
        )
        group.create_dataset(
            "labels",
            data=labels_np,
            chunks=self.chunk_shape_vector,
            compression=self.compression,
            compression_opts=self.compression_opts,
        )

        self.file.flush()
        return True

    def retrieve_shard_embeddings(
        self,
        training_step: TrainingStep,
        shard_idx: int,
    ) -> T.Optional[T.Tuple[np.ndarray, np.ndarray, T.List[str]]]:
        """Retrieve embeddings, labels for a shard, using global IDs and mask.

        Args:
            training_step : TrainingStep identifier.
            shard_idx : int Index of the shard.

        Returns:
            Tuple containing (filtered_embeddings_np, filtered_labels_np, filtered_ids_list)
            or None if shard/global data is missing or inconsistent.
        """
        ids_path = self._get_ids_path_for_step(training_step)
        mask_path = self._get_mask_path_for_step(training_step)

        if ids_path not in self.file or mask_path not in self.file:
            logger.error(
                f"IDs or Mask not found in the store for TrainingStep '{training_step.value}'."
            )
            return None

        step_ids_bytes = self.file[ids_path][()]
        step_valid_mask = self.file[mask_path][()]

        shard_group_path = self._get_shard_group_path(training_step, shard_idx)
        if shard_group_path not in self.file:
            logger.debug(f"Shard {shard_group_path} not found.")
            return None

        shard_group = self.file[shard_group_path]
        if not all(dname in shard_group for dname in self.required_shard_datasets):
            logger.warning(
                f"Shard group {shard_group_path} is incomplete. Skipping retrieval."
            )
            return None

        if not np.any(step_valid_mask):
            return None

        shard_embeddings = shard_group["embeddings"][()]
        shard_labels = shard_group["labels"][()]

        num_step_datapoints = len(step_ids_bytes)
        if (
            shard_embeddings.shape[0] != num_step_datapoints
            or shard_labels.shape[0] != num_step_datapoints
        ):
            logger.error(
                f"Data length mismatch in shard {shard_group_path}! "
                f"Expected {num_step_datapoints}, got {shard_embeddings.shape[0]} embeddings, "
                f"{shard_labels.shape[0]} labels. Data may be corrupt or was stored incorrectly."
            )
            return None

        filtered_embeddings = shard_embeddings[step_valid_mask]
        filtered_labels = shard_labels[step_valid_mask]
        filtered_ids_bytes = step_ids_bytes[step_valid_mask]
        filtered_ids_str = [id_b.decode("utf-8") for id_b in filtered_ids_bytes]

        return filtered_embeddings, filtered_labels, filtered_ids_str

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
        shard_group_path = self._get_shard_group_path(training_step, shard_idx)
        if shard_group_path in self.file:
            try:
                del self.file[shard_group_path]
                self.file.flush()
                logger.info(f"Removed shard group: {shard_group_path}")
                return True
            except Exception as e:
                logger.error(f"Error removing shard group {shard_group_path}: {e}")
                return False
        return False

    def remove_embedding(self, training_step: TrainingStep, datapoint_id: str) -> bool:
        """Mark a datapoint_id as invalid in the global mask.

        This performs "lazy deletion". Run compact_file() to physically remove data.

        Args:
            datapoint_id: The ID to mark as invalid.

        Returns:
            True if the ID was found and marked invalid, False otherwise.
        """
        if not self.file:
            raise RuntimeError("Storage file is closed.")

        ids_path = self._get_ids_path_for_step(training_step)
        mask_path = self._get_mask_path_for_step(training_step)

        if ids_path not in self.file or mask_path not in self.file:
            logger.error(
                f"IDs or Mask not initialized for TrainingStep '{training_step.value}'. Cannot remove embedding."
            )
            return False

        datapoint_id_bytes = datapoint_id.encode("utf-8")
        step_ids_dataset = self.file[ids_path]
        step_mask_dataset = self.file[mask_path]

        try:
            step_ids_bytes = step_ids_dataset[()]
            indices = np.where(step_ids_bytes == datapoint_id_bytes)[0]

            if len(indices) == 0:
                logger.warning(
                    f"Datapoint ID '{datapoint_id}' not found in global ID list."
                )
                return False

            step_index = indices[0]
            if len(indices) > 1:
                logger.warning(
                    f"Datapoint ID '{datapoint_id}' found multiple times in global list (indices: {indices}). Using first occurrence ({step_index})."
                )

            step_mask_dataset[step_index] = False
            self.file.flush()
            return True

        except Exception as e:
            logger.error(
                f"Error accessing global datasets during removal of '{datapoint_id}': {e}"
            )
            return False
