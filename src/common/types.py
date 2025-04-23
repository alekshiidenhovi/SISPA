import typing as T
from enum import Enum
from torch.utils.data import Dataset


class TrainingStep(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"


class Operation(Enum):
    ADD = "add"
    REMOVE = "remove"


SHARDED_DATASET_SPLITS = T.Tuple[T.List[T.List[int]], T.List[int], T.List[int]]
DATASET_SPLIT_STRATEGY_FUNCTION = T.Callable[
    [Dataset, T.Tuple[float, float, float], int, int],
    SHARDED_DATASET_SPLITS,
]


class DatasetSplitStrategy(Enum):
    CLASS_INFORMED = "class_informed"
    EQUAL = "equal"
