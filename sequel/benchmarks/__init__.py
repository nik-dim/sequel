import os
from pathlib import Path

DEFAULT_DATASET_DIR = Path(Path.home(), "benchmarks", "data").absolute()
if not DEFAULT_DATASET_DIR.exists():
    DEFAULT_DATASET_DIR.mkdir(parents=True, exist_ok=True)

from sequel.benchmarks.base_benchmark import Benchmark
from sequel.benchmarks.mnist import SplitMNIST, PermutedMNIST, RotatedMNIST
from sequel.benchmarks.cifar import SplitCIFAR10, SplitCIFAR100
from sequel.benchmarks.tinyimagenet import SplitTinyImagenet

__all__ = [
    "DEFAULT_DATASET_DIR",
    "Benchmark",
    "SplitMNIST",
    "PermutedMNIST",
    "RotatedMNIST",
    "SplitCIFAR10",
    "SplitCIFAR100",
    "SplitTinyImagenet",
]


def select_benchmark(config, *args, **kwargs) -> Benchmark:
    benchmark_name = config.name.lower()

    if benchmark_name == "splitmnist":
        return SplitMNIST.from_config(config)
    elif benchmark_name == "permutedmnist":
        return PermutedMNIST.from_config(config)
    elif benchmark_name == "rotatedmnist":
        return RotatedMNIST.from_config(config)
    elif benchmark_name == "splitcifar10":
        return SplitCIFAR10.from_config(config)
    elif benchmark_name == "splitcifar100":
        return SplitCIFAR100.from_config(config)
    elif benchmark_name == "tinyimagenet":
        return SplitTinyImagenet.from_config(config)

    raise NotImplementedError(f"The benchmark {benchmark_name} is not implemented yet!")
