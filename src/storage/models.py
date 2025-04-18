import torch
import wandb
import os
import typing as T
from pathlib import Path
from safetensors.torch import load_file, save_file
import wandb.wandb_run


class SISPAModelStorage:
    """
    Storage interface for SISPA models.

    This class provides methods for saving and loading models to and from local storage and W&B.
    """

    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.base_dir = "models"
        os.makedirs(os.path.join(storage_path, self.base_dir), exist_ok=True)

    def _get_model_name(self, shard_id: str) -> str:
        """
        Get the name of the model name for a given shard.

        Args:
            shard_id: Identifier for the SISA shard

        Returns:
            Name of the model name
        """
        return f"sisa_shard_{shard_id}"

    def _get_model_filename(self, shard_id: str) -> str:
        """
        Get the name of the model file for a given shard.

        Args:
            shard_id: Identifier for the SISA shard

        Returns:
            Name of the model file
        """
        return f"{self._get_model_name(shard_id)}.safetensors"

    def _get_model_artifact_name(self, shard_id: str) -> str:
        """
        Get the name of the model artifact for a given shard.

        Args:
            shard_id: Identifier for the SISA shard

        Returns:
            Name of the model artifact
        """
        return f"{self._get_model_name(shard_id)}:latest"

    def save_model(
        self,
        model: torch.nn.Module,
        shard_id: str,
        wandb_run: wandb.wandb_run.Run,
    ) -> str:
        """
        Save a PyTorch model trained on a SISA shard to both local storage and W&B.

        Args:
            model: The PyTorch model to save
            shard_id: Identifier for the SISA shard
            wandb_run: W&B run to use for uploading

        Returns:
            Path to the locally saved model file
        """

        model_filename = self._get_model_filename(shard_id)
        save_path = os.path.join(self.storage_path, self.base_dir, model_filename)
        state_dict = model.state_dict()

        metadata = {"shard_id": shard_id}
        save_file(state_dict, save_path, metadata=metadata)

        model_artifact_name = self._get_model_artifact_name(shard_id)
        artifact = wandb.Artifact(
            name=model_artifact_name, type="model", metadata=metadata
        )
        artifact.add_file(save_path)
        wandb_run.log_artifact(artifact)

        return save_path

    def load_model_local(
        self,
        model: torch.nn.Module,
        shard_id: str,
    ) -> torch.nn.Module:
        """
        Load a model for a specific SISA shard from local storage.

        Args:
            model: An initialized model instance to load weights into
            shard_id: Identifier for the SISA shard

        Returns:
            The loaded model
        """

        model_filename = self._get_model_filename(shard_id)
        load_path = os.path.join(self.storage_path, self.base_dir, model_filename)

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found at {load_path}")

        state_dict = load_file(load_path)
        model.load_state_dict(state_dict)

        return model

    def load_model_wandb(
        self,
        model: torch.nn.Module,
        shard_id: str,
        wandb_artifact_name: T.Optional[str] = None,
    ) -> torch.nn.Module:
        """
        Load a model for a specific SISA shard from W&B.

        Args:
            model: An initialized model instance to load weights into
            shard_id: Identifier for the SISA shard
            wandb_artifact_name: Name of the W&B artifact to load (defaults to latest version)

        Returns:
            The loaded model
        """
        if wandb_artifact_name is None:
            wandb_artifact_name = self._get_model_artifact_name(shard_id)

        artifact = wandb.use_artifact(wandb_artifact_name)
        artifact_dir = artifact.download()

        safetensors_files = list(Path(artifact_dir).glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(
                f"No safetensors file found in W&B artifact {wandb_artifact_name}"
            )

        load_path = str(safetensors_files[0])
        state_dict = load_file(load_path)
        model.load_state_dict(state_dict)

        return model

    def get_local_metadata(self, shard_id: str) -> T.Dict:
        """
        Get metadata for a specific SISA shard model from local storage.

        Args:
            shard_id: Identifier for the SISA shard

        Returns:
            Dictionary of metadata
        """
        model_filename = self._get_model_filename(shard_id)
        load_path = os.path.join(self.storage_path, self.base_dir, model_filename)

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found at {load_path}")

        metadata = load_file(load_path, metadata_only=True)
        return metadata

    def get_wandb_metadata(self, shard_id: str) -> T.Dict:
        """
        Get metadata for a specific SISA shard model from W&B.

        Args:
            shard_id: Identifier for the SISA shard

        Returns:
            Dictionary of metadata
        """
        model_artifact_name = self._get_model_artifact_name(shard_id)
        artifact = wandb.use_artifact(model_artifact_name)
        return artifact.metadata
