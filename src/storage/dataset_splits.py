import h5py
import numpy as np
import os
import typing as T
from common.types import TrainingStep, Operation


class SISPADatasetSplitsStorage:
    """Storage interface for dataset splits.

    Manages storage and retrieval of dataset splits using HDF5.

    Parameters
    ----------
    storage_path : str
        Path to the HDF5 file
    """

    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.base_file = "dataset_splits.hdf5"
        self.file = h5py.File(os.path.join(storage_path, self.base_file), "a")

    def __del__(self):
        """Close the file when the object is deleted."""
        if hasattr(self, "file"):
            self.file.close()

    def _get_split_dir(self, training_step: TrainingStep):
        return f"{training_step.value}"

    def _get_split_indices_path(self, training_step: TrainingStep):
        return f"{self._get_split_dir(training_step)}/indices"

    def _get_shards_dir(self):
        return f"{self._get_split_dir(TrainingStep.TRAINING)}/shards"

    def _get_shard_dir(self, shard_idx: int) -> str:
        return f"{self._get_shards_dir()}/{shard_idx}"

    def _create_split_dir(self, training_step: TrainingStep):
        split_dir = self._get_split_dir(training_step)
        if split_dir not in self.file:
            self.file.create_group(split_dir)

    def _update_shard_indices_metadata(self, shard_idx: int, operation: Operation):
        """Update the shard indices metadata in the training dataset.

        Parameters
        ----------
        shard_idx : int
            Index of the shard to add or remove
        operation : Operation
            Operation to perform on the shard index
        """
        train_path = self._get_split_indices_path(TrainingStep.TRAINING)
        if train_path not in self.file:
            self.file.create_dataset(train_path, shape=(0,), dtype=np.int64)

        shard_indices = self.file[train_path].attrs.get("shard_indices", [])

        if operation == Operation.ADD and shard_idx not in shard_indices:
            shard_indices.append(shard_idx)
            shard_indices.sort()
        elif operation == Operation.REMOVE and shard_idx in shard_indices:
            shard_indices.remove(shard_idx)

        # Store updated shard indices
        self.file[train_path].attrs["shard_indices"] = shard_indices

    def store_split(self, training_step: TrainingStep, indices: T.List[int]):
        """Store split indices.

        Parameters
        ----------
        indices : list[int]
            Indices of samples in the training split
        """
        self._create_split_dir(training_step)

        split_path = self._get_split_indices_path(training_step)
        if split_path in self.file:
            del self.file[split_path]

        indices_np = np.array(indices, dtype=np.int64)
        self.file.create_dataset(
            split_path,
            shape=indices_np.shape,
            data=indices_np,
            dtype=np.int64,
        )

    def store_shard(self, shard_idx: int, indices: T.List[int]):
        """Store a dataset shard under the training directory.

        Parameters
        ----------
        shard_idx : int
            Index of the shard
        indices : list[int]
            Indices of samples in the shard
        """
        shards_dir = self._get_shards_dir()
        if shards_dir not in self.file:
            self.file.create_group(shards_dir)

        shard_path = self._get_shard_dir(shard_idx)
        if shard_path in self.file:
            del self.file[shard_path]

        indices_np = np.array(indices, dtype=np.int64)
        self.file.create_dataset(
            shard_path,
            shape=indices_np.shape,
            data=indices_np,
            dtype=np.int64,
        )

        self._update_shard_indices_metadata(shard_idx, operation=Operation.ADD)

    def store_all_splits(
        self,
        train_shard_indices: T.List[T.List[int]],
        val_indices: T.List[int],
        test_indices: T.List[int],
    ):
        """Store all dataset splits at once.

        Parameters
        ----------
        train_shard_indices : list[list[int]]
            List of indices for each training shard
        val_indices : list[int]
            Indices for validation set
        test_indices : list[int]
            Indices for test set
        """

        self.store_split(TrainingStep.VALIDATION, val_indices)
        self.store_split(TrainingStep.TESTING, test_indices)

        all_train_indices = [idx for shard in train_shard_indices for idx in shard]
        self.store_split(TrainingStep.TRAINING, all_train_indices)

        for shard_idx, shard_indices in enumerate(train_shard_indices):
            self.store_shard(shard_idx, shard_indices)

    def retrieve_split(self, training_step: TrainingStep) -> T.Optional[T.List[int]]:
        """Retrieve the split indices.

        Returns
        -------
        list[int] or None
            Indices of the split, or None if not found
        """
        split_path = self._get_split_indices_path(training_step)
        if split_path not in self.file:
            return None

        indices_np = self.file[split_path][()]
        return indices_np.tolist()

    def retrieve_shard(self, shard_idx: int) -> T.Optional[T.List[int]]:
        """Retrieve a dataset shard.

        Parameters
        ----------
        shard_idx : int
            Index of the shard to retrieve

        Returns
        -------
        list[int] or None
            Indices of the shard, or None if not found
        """
        shard_path = self._get_shard_path(shard_idx)
        if shard_path not in self.file:
            return None

        indices_np = self.file[shard_path][()]
        return indices_np.tolist()

    def retrieve_all_splits(
        self,
    ) -> T.Tuple[T.List[T.List[int]], T.List[int], T.List[int], T.List[int]]:
        """Retrieve all dataset splits.

        Returns
        -------
        tuple
            Tuple of (train_shard_indices, val_indices, test_indices)
        """
        train_indices = self.retrieve_split(TrainingStep.TRAINING)
        val_indices = self.retrieve_split(TrainingStep.VALIDATION)
        test_indices = self.retrieve_split(TrainingStep.TESTING)
        shard_indices = self.list_shards()

        if train_indices is None:
            raise ValueError("Training split not found")
        if val_indices is None:
            raise ValueError("Validation split not found")
        if test_indices is None:
            raise ValueError("Test split not found")
        if shard_indices is None:
            raise ValueError("Shard indices not found")

        train_shard_indices = []
        for shard_idx in shard_indices:
            shard = self.retrieve_shard(shard_idx)
            if shard is None:
                raise ValueError(f"Shard {shard_idx} not found")
            train_shard_indices.append(shard.indices)

        return train_shard_indices, train_indices, val_indices, test_indices

    def remove_split(self, training_step: TrainingStep) -> bool:
        """Remove the split from storage.

        Returns
        -------
        bool
            True if removed successfully, False if not found
        """
        split_path = self._get_split_indices_path(training_step)
        if split_path in self.file:
            del self.file[split_path]
            return True
        return False

    def remove_shard(self, shard_idx: int) -> bool:
        """Remove a shard from storage.

        Parameters
        ----------
        shard_idx : int
            Index of the shard to remove

        Returns
        -------
        bool
            True if removed successfully, False if not found
        """
        shard_path = self._get_shard_path(shard_idx)
        if shard_path in self.file:
            del self.file[shard_path]
            self._update_shard_indices_metadata(shard_idx, operation=Operation.REMOVE)
            return True
        return False

    def list_shards(self) -> T.List[int]:
        """List all available shards.

        Returns
        -------
        list[int]
            List of shard indices
        """
        train_path = self._get_split_indices_path(TrainingStep.TRAINING)
        if train_path not in self.file:
            return []

        shard_indices = self.file[train_path].attrs["shard_indices"]
        return sorted(shard_indices)
