import typing as T
from pydantic import BaseModel, Field
from torch.utils.data import Dataset


class BaseDatasetSplitStrategyParams(BaseModel):
    dataset: Dataset = Field(description="The dataset to split")
    train_val_test_split: T.Tuple[float, float, float] = Field(
        description="Tuple of (train_ratio, val_ratio, test_ratio) that sum to 1.0"
    )
    seed: int = Field(description="Random seed for reproducibility")

    class Config:
        arbitrary_types_allowed = True


class ClassInformedDatasetSplitStrategyParams(BaseDatasetSplitStrategyParams):
    sampling_ratio: float = Field(
        description="Proportion of samples that should go to each class's dedicated shard"
    )


class EqualDatasetSplitStrategyParams(BaseDatasetSplitStrategyParams):
    num_shards: int = Field(
        description="Number of equal shards to create from the training set"
    )
