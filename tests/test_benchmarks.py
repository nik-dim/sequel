import sys
from pathlib import Path

sys.path.append(Path().parent.absolute().as_posix())
import pytest
import re
from sequel.benchmarks import (
    Benchmark,
    SplitCIFAR10,
    SplitCIFAR100,
    SplitMNIST,
    RotatedMNIST,
    PermutedMNIST,
    SplitTinyImagenet,
)


def test_benchmark():
    with pytest.raises(
        TypeError, match=re.escape("Benchmark.__init__() missing 1 required positional argument: 'batch_size'")
    ):
        benchmark = Benchmark(num_tasks=5)

    with pytest.raises(NotImplementedError):
        Benchmark(num_tasks=5, batch_size=1)


def test_splitmnist():
    with pytest.raises(
        TypeError, match=re.escape("Benchmark.__init__() missing 1 required positional argument: 'batch_size'")
    ):
        benchmark = SplitMNIST(num_tasks=5)

    bs = 12
    benchmark = SplitMNIST(num_tasks=5, batch_size=bs)
    assert benchmark.num_tasks == 5

    with pytest.raises(
        ValueError, match=re.escape("Split MNIST benchmark can have at most 5 tasks (i.e., 10 classes, 2 per task)")
    ):
        SplitMNIST(num_tasks=10, batch_size=bs)

    with pytest.raises(ValueError):
        SplitMNIST(num_tasks=7, batch_size=bs)

    with pytest.raises(
        ValueError, match=re.escape(f"Asked to load task {999} but the benchmark has {benchmark.num_tasks} tasks")
    ):
        benchmark.train_dataloader(task=999)
