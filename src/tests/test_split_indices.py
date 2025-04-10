import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from datasets.mnist.shard_splits import calculate_split_indices


def test_basic_functionality():
    class_labels = {0, 1, 2, 3}
    class_label = 2
    num_samples = 100
    sampling_ratio = 0.5

    splits = calculate_split_indices(
        class_labels, class_label, num_samples, sampling_ratio
    )

    assert len(splits) == len(class_labels) + 1
    assert splits[0] == 0
    assert splits[-1] == num_samples - 1


def test_binary_case():
    class_labels = {0, 1}
    class_label = 1
    num_samples = 100
    sampling_ratio = 0.7

    splits = calculate_split_indices(
        class_labels, class_label, num_samples, sampling_ratio
    )

    assert len(splits) == 3
    assert splits == [0, 30, 100]
