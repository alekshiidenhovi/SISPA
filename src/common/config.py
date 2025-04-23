from pydantic import BaseModel, Field, field_validator
import typing as T
import multiprocessing
from common.types import DatasetSplitStrategy, AVAILABLE_DATASETS, ACCELERATOR
from common.tracking import init_wandb_api_client


class DatasetConfig(BaseModel):
    """Configuration for dataset loading and preprocessing.

    Contains parameters for data loading, batch sizes, image dimensions,
    dataset splits, and sampling configurations for training, validation and testing.
    """

    dataset_name: AVAILABLE_DATASETS = Field(
        default="cifar10",
        description="Name of the dataset",
    )
    train_batch_size: int = Field(
        default=64,
        ge=1,
        description="Batch size for training",
    )
    val_batch_size: int = Field(
        default=64,
        ge=1,
        description="Batch size for validation",
    )
    test_batch_size: int = Field(
        default=64,
        ge=1,
        description="Batch size for testing",
    )
    num_workers: int = Field(
        default=multiprocessing.cpu_count() // 2,
        description="Number of workers for data loading",
        ge=1,
    )
    train_val_test_split: T.Tuple[float, float, float] = Field(
        default=(0.8, 0.1, 0.1),
        description="Train, validation and test split proportions",
    )
    num_shards: int = Field(
        default=10,
        ge=1,
        description="Number of shards for SISA training",
    )
    class_informed_strategy_sampling_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Proportion of samples from target class in each shard",
    )
    dataset_split_strategy: str = Field(
        default=DatasetSplitStrategy.EQUAL.value,
        description="Strategy to use for dataset splitting",
    )

    @field_validator("train_val_test_split")
    def validate_split_sum(cls, v):
        ratio = sum(v)
        if not abs(ratio - 1.0) < 1e-6:
            raise ValueError(
                "Train, validation and test split proportions must sum to 1"
            )
        return v


class ModelConfig(BaseModel):
    """Configuration for the model architecture."""

    resnet_block_dims: T.List[int] = Field(
        default=[32, 64, 128, 256],
        description="Dimensions of the Resnet blocks",
    )
    resnet_num_modules_per_block: int = Field(
        default=3, ge=1, description="Number of modules in each Resnet block"
    )
    aggregator_hidden_dim: int = Field(
        default=128, ge=1, description="Hidden dimension of the aggregation model"
    )


class FinetuningConfig(BaseModel):
    """Configuration for model fine-tuning.

    Contains training hyperparameters including learning rate schedules, validation intervals,
    checkpointing settings and LoRA-specific parameters.
    """

    accelerator: ACCELERATOR = Field(
        default="gpu",
        description="Compute accelerator to use for training",
    )
    val_check_interval_percentage: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="The percentage of training batches of an epoch after which to run validation",
    )
    epochs: int = Field(
        default=8,
        ge=1,
        description="Number of epochs to train for",
    )
    devices: T.List[int] = Field(
        default=[0],
        description="GPU devices to use for training",
    )
    accumulate_grad_batches: int = Field(
        default=1,
        ge=1,
        description="Number of batches to accumulate gradients before updating the model",
    )


class OptimizerConfig(BaseModel):
    """Configuration for the optimizer.

    Contains parameters for the optimizer including weight decay, beta1 and beta2 parameters for the Adam optimizer.
    """

    optimizer_weight_decay: float = Field(
        default=0.01,
        ge=0,
        description="Weight decay for the optimizer",
    )
    optimizer_adam_beta1: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Beta1 parameter for the Adam optimizer, used for the first moment estimate",
    )
    optimizer_adam_beta2: float = Field(
        default=0.999,
        ge=0.0,
        le=1.0,
        description="Beta2 parameter for the Adam optimizer, used for the second moment estimate",
    )
    optimizer_learning_rate: float = Field(
        default=1e-4, ge=0, description="Learning rate of the model"
    )


class TrainingConfig(DatasetConfig, ModelConfig, FinetuningConfig, OptimizerConfig):
    """Complete training configuration combining dataset, model and fine-tuning settings.

    Inherits from DatasetConfig, ModelConfig, FinetuningConfig and OptimizerConfig to provide a comprehensive configuration for the entire training pipeline.
    """

    seed: int = Field(
        default=42,
        description="Seed for training reproducibility",
    )
    storage_path: str = Field(
        description="Directory to save training artifacts, such as dataset splits, precomputed embeddings, and trained model weights",
    )

    def get_dataset_config(self) -> DatasetConfig:
        """Get dataset-specific configuration."""
        return DatasetConfig(
            **{
                k: v
                for k, v in self.model_dump().items()
                if k in DatasetConfig.model_fields
            }
        )

    def get_model_config(self) -> ModelConfig:
        """Get model architecture configuration."""
        return ModelConfig(
            **{
                k: v
                for k, v in self.model_dump().items()
                if k in ModelConfig.model_fields
            }
        )

    def get_finetuning_config(self) -> FinetuningConfig:
        """Get fine-tuning hyperparameters configuration."""
        return FinetuningConfig(
            **{
                k: v
                for k, v in self.model_dump().items()
                if k in FinetuningConfig.model_fields
            }
        )

    def get_optimizer_config(self) -> OptimizerConfig:
        """Get optimizer configuration."""
        return OptimizerConfig(
            **{
                k: v
                for k, v in self.model_dump().items()
                if k in OptimizerConfig.model_fields
            }
        )

    @classmethod
    def load_from_wandb(cls, run_id: str):
        """Load the configuration from a W&B run."""
        wandb_api = init_wandb_api_client()
        run = wandb_api.run(run_id)
        valid_config = {k: v for k, v in run.config.items() if k in cls.model_fields}
        return cls(**valid_config)
