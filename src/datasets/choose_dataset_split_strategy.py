import typing as T
from common.types import DatasetSplitStrategy, DATASET_SPLIT_STRATEGY_FUNCTION
from datasets.class_informed import create_class_informed_dataset_splits
from datasets.equal import create_equal_distribution_dataset_splits
from datasets.dataset_split_params import (
    BaseDatasetSplitStrategyParams,
    ClassInformedDatasetSplitStrategyParams,
    EqualDatasetSplitStrategyParams,
)


def get_valid_params(
    param_parser: BaseDatasetSplitStrategyParams,
    params: T.Dict[str, T.Any],
) -> T.Dict[str, T.Any]:
    valid_fields = set(param_parser.model_fields.keys())
    return {k: v for k, v in params.items() if k in valid_fields}


def choose_dataset_split_strategy(
    strategy: DatasetSplitStrategy, params: T.Dict[str, T.Any]
) -> T.Tuple[DATASET_SPLIT_STRATEGY_FUNCTION, BaseDatasetSplitStrategyParams]:
    """Chooses the dataset split strategy and params based on the strategy enum.

    Args:
        strategy: The dataset split strategy to choose

    Returns:
        Tuple containing:
        - The dataset split strategy function
        - The params for the chosen strategy
    """
    if strategy == DatasetSplitStrategy.CLASS_INFORMED:
        return (
            create_class_informed_dataset_splits,
            ClassInformedDatasetSplitStrategyParams(
                **get_valid_params(ClassInformedDatasetSplitStrategyParams, params)
            ),
        )
    elif strategy == DatasetSplitStrategy.EQUAL:
        return (
            create_equal_distribution_dataset_splits,
            EqualDatasetSplitStrategyParams(
                **get_valid_params(EqualDatasetSplitStrategyParams, params)
            ),
        )
    else:
        raise ValueError(f"Invalid dataset split strategy: {strategy}")
