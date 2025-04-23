import torch
import wandb
import os
import typing as T
from pathlib import Path
from safetensors.torch import load_file, save_file
from common.tracking import init_wandb_run
from common.types import AVAILABLE_DATASETS


class SISPAModelStorage:
    """
    Storage interface for SISPA models.

    This class provides methods for saving and loading models to and from local storage and W&B.
    """

    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.base_dir = "models"
        self.sharded_model_dir = "sharded"
        self.aggregator_model_dir = "aggregator"

        os.makedirs(os.path.join(storage_path, self.base_dir), exist_ok=True)
        os.makedirs(
            os.path.join(storage_path, self.base_dir, self.sharded_model_dir),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(storage_path, self.base_dir, self.aggregator_model_dir),
            exist_ok=True,
        )

    def _get_sharded_model_name(self, shard_id: str) -> str:
        """
        Get the name of the model name for a given shard.

        Args:
            shard_id: Identifier for the SISA shard

        Returns:
            Name of the model name
        """
        return f"sisa_shard_{shard_id}"

    def _get_shard_model_path(self, shard_id: str) -> str:
        """
        Get the path to the model directory for a given shard.

        Args:
            shard_id: Identifier for the SISA shard

        Returns:
            Path to the model directory
        """
        return os.path.join(self.storage_path, self.base_dir, self.sharded_model_dir)

    def _get_sharded_model_filename(self, shard_id: str) -> str:
        """
        Get the name of the model file for a given shard.

        Args:
            shard_id: Identifier for the SISA shard

        Returns:
            Name of the model file
        """
        return f"{self._get_shard_model_path(shard_id)}/{self._get_sharded_model_name(shard_id)}.safetensors"

    def _get_sharded_model_artifact_name(self, shard_id: str) -> str:
        """
        Get the name of the model artifact for a given shard.

        Args:
            shard_id: Identifier for the SISA shard

        Returns:
            Name of the model artifact
        """
        return f"{self._get_sharded_model_name(shard_id)}-latest"

    def _get_aggregator_model_path(self) -> str:
        """
        Get the path to the model directory for the aggregator model.

        Returns:
            Path to the model directory
        """
        return os.path.join(self.storage_path, self.base_dir, self.aggregator_model_dir)

    def _get_aggregator_model_filename(self) -> str:
        """
        Get the name of the model file for the aggregator model.

        Returns:
            Name of the model file
        """
        return f"{self._get_aggregator_model_path()}/aggregator_model.safetensors"

    def _get_aggregator_model_artifact_name(self) -> str:
        """
        Get the name of the model artifact for the aggregator model.

        Returns:
            Name of the model artifact
        """
        return "aggregator_model-latest"

    def save_sharded_model(
        self,
        sharded_model: torch.nn.Module,
        shard_id: str,
        experiment_group_name: str,
        dataset_name: AVAILABLE_DATASETS,
    ) -> str:
        """
        Save a PyTorch model trained on a SISA shard to both local storage and W&B.

        Args:
            sharded_model: The PyTorch model to save
            shard_id: Identifier for the SISA shard
            experiment_group_name: Name of the experiment group
            dataset_name: Name of the dataset
        Returns:
            Path to the locally saved model file
        """

        save_path = self._get_sharded_model_filename(shard_id)
        state_dict = sharded_model.state_dict()

        metadata = {"shard_id": shard_id}
        save_file(state_dict, save_path, metadata=metadata)

        artifact = wandb.Artifact(
            name=self._get_sharded_model_artifact_name(shard_id),
            type="model",
            metadata=metadata,
        )
        artifact.add_file(save_path)
        wandb_run = init_wandb_run(
            dataset_name=dataset_name,
            experiment_group_name=experiment_group_name,
            reinit=False,
        )
        wandb_run.log_artifact(artifact)

        return save_path

    def load_sharded_model_local(
        self,
        sharded_model: torch.nn.Module,
        shard_id: str,
    ) -> torch.nn.Module:
        """
        Load a model for a specific SISA shard from local storage.

        Args:
            sharded_model: An initialized model instance to load weights into
            shard_id: Identifier for the SISA shard

        Returns:
            The loaded model
        """

        load_path = self._get_sharded_model_filename(shard_id)

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found at {load_path}")

        state_dict = load_file(load_path)
        sharded_model.load_state_dict(state_dict)

        return sharded_model

    def load_sharded_model_wandb(
        self,
        sharded_model: torch.nn.Module,
        shard_id: str,
        wandb_artifact_name: T.Optional[str] = None,
    ) -> torch.nn.Module:
        """
        Load a model for a specific SISA shard from W&B.

        Args:
            sharded_model: An initialized model instance to load weights into
            shard_id: Identifier for the SISA shard
            wandb_run: W&B run to use for loading
            wandb_artifact_name: Name of the W&B artifact to load (defaults to latest version)

        Returns:
            The loaded model
        """
        if wandb_artifact_name is None:
            wandb_artifact_name = self._get_sharded_model_artifact_name(shard_id)

        artifact: wandb.Artifact = wandb.use_artifact(wandb_artifact_name)
        artifact_dir = artifact.download()

        safetensors_files = list(Path(artifact_dir).glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(
                f"No safetensors file found in W&B artifact {wandb_artifact_name}"
            )

        load_path = str(safetensors_files[0])
        state_dict = load_file(load_path)
        sharded_model.load_state_dict(state_dict)

        return sharded_model

    def get_sharded_model_local_metadata(self, shard_id: str) -> T.Dict:
        """
        Get metadata for a specific SISA shard model from local storage.

        Args:
            shard_id: Identifier for the SISA shard

        Returns:
            Dictionary of metadata
        """
        model_filename = self._get_sharded_model_filename(shard_id)
        load_path = os.path.join(self.sharded_model_dir, model_filename)

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found at {load_path}")

        metadata = load_file(load_path, metadata_only=True)
        return metadata

    def get_sharded_model_wandb_metadata(self, shard_id: str) -> T.Dict:
        """
        Get metadata for a specific SISA shard model from W&B.

        Args:
            shard_id: Identifier for the SISA shard

        Returns:
            Dictionary of metadata
        """
        model_artifact_name = self._get_sharded_model_artifact_name(shard_id)
        artifact = wandb.use_artifact(model_artifact_name)
        return artifact.metadata

    def save_aggregator_model(
        self,
        aggregator_model: torch.nn.Module,
        experiment_group_name: str,
        dataset_name: AVAILABLE_DATASETS,
    ) -> str:
        """
        Save a PyTorch aggregator model to both local storage and W&B.

        Args:
            aggregator_model: The PyTorch aggregator model to save
            experiment_group_name: Name of the experiment group
            dataset_name: Name of the dataset
        Returns:
            Path to the locally saved model file
        """
        save_path = self._get_aggregator_model_filename()
        state_dict = aggregator_model.state_dict()

        metadata = {"model_type": "aggregator"}
        save_file(state_dict, save_path, metadata=metadata)

        artifact: wandb.Artifact = wandb.Artifact(
            name=self._get_aggregator_model_artifact_name(),
            type="model",
            metadata=metadata,
        )
        artifact.add_file(save_path)
        wandb_run = init_wandb_run(
            dataset_name=dataset_name,
            experiment_group_name=experiment_group_name,
            reinit=False,
        )
        wandb_run.log_artifact(artifact)

        return save_path

    def load_aggregator_model_local(
        self,
        aggregator_model: torch.nn.Module,
    ) -> torch.nn.Module:
        """
        Load an aggregator model from local storage.

        Args:
            aggregator_model: An initialized model instance to load weights into

        Returns:
            The loaded model
        """
        load_path = self._get_aggregator_model_filename()

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Aggregator model file not found at {load_path}")

        state_dict = load_file(load_path)
        aggregator_model.load_state_dict(state_dict)

        return aggregator_model

    def load_aggregator_model_wandb(
        self,
        aggregator_model: torch.nn.Module,
        wandb_artifact_name: T.Optional[str] = None,
    ) -> torch.nn.Module:
        """
        Load an aggregator model from W&B.

        Args:
            aggregator_model: An initialized model instance to load weights into
            wandb_run: W&B run to use for loading
            wandb_artifact_name: Name of the W&B artifact to load (defaults to latest version)

        Returns:
            The loaded model
        """
        if wandb_artifact_name is None:
            wandb_artifact_name = self._get_aggregator_model_artifact_name()

        artifact: wandb.Artifact = wandb.use_artifact(wandb_artifact_name)
        artifact_dir = artifact.download()

        safetensors_files = list(Path(artifact_dir).glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(
                f"No safetensors file found in W&B artifact {wandb_artifact_name}"
            )

        load_path = str(safetensors_files[0])
        state_dict = load_file(load_path)
        aggregator_model.load_state_dict(state_dict)

        return aggregator_model

    def get_aggregator_model_local_metadata(self) -> T.Dict:
        """
        Get metadata for the aggregator model from local storage.

        Returns:
            Dictionary of metadata
        """
        load_path = self._get_aggregator_model_filename()

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Aggregator model file not found at {load_path}")

        metadata = load_file(load_path, metadata_only=True)
        return metadata

    def get_aggregator_model_wandb_metadata(self) -> T.Dict:
        """
        Get metadata for the aggregator model from W&B.

        Args:
            wandb_run: W&B run to use for loading

        Returns:
            Dictionary of metadata
        """
        artifact: wandb.Artifact = wandb.use_artifact(
            self._get_aggregator_model_artifact_name()
        )
        return artifact.metadata
