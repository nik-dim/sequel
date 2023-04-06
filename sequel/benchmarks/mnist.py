from typing import List, Optional
import warnings

import torch
import torchvision
import torchvision.transforms as T
from torchvision.transforms.functional import rotate

from sequel.benchmarks.base_benchmark import Benchmark

from . import DEFAULT_DATASET_DIR
from .utils import ContinualVisionDataset, SplitDataset


class PermuteTransform:
    def __init__(self, permute_indices):
        self.permuted_indices = permute_indices

    def __call__(self, x: torch.Tensor):
        shape = x.shape
        return x.view(-1)[self.permuted_indices].view(shape)


class RotationTransform:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return rotate(x, self.angle, fill=(0,), interpolation=T.InterpolationMode.BILINEAR)


class ContinualMNIST(Benchmark):
    """Base class for (Permuted/Rotated/Split)-MNIST benchmarks."""

    @property
    def dimensions(self) -> List[int]:
        return [1, 28, 28]

    @property
    def num_classes(self) -> int:
        return 10

    MEAN = (0.1307,)
    STD = (0.3081,)


class SplitMNIST(ContinualMNIST):
    """Split MNIST benchmark.

    The benchmark can have at most 5 tasks, each a binary classification on MNIST digits.
    """

    @property
    def classes_per_task(self):
        if self.num_tasks not in [2, 5]:
            raise ValueError("Split MNIST benchmark can have at most 5 tasks (i.e., 10 classes, 2 per task)")
        return 10 // self.num_tasks

    def prepare_datasets(self):
        transform = T.Compose([T.ToTensor(), T.Normalize(self.MEAN, self.STD)])
        mnist_train = torchvision.datasets.MNIST(DEFAULT_DATASET_DIR, train=True, download=True, transform=transform)
        mnist_test = torchvision.datasets.MNIST(DEFAULT_DATASET_DIR, train=False, download=True, transform=transform)

        trains, tests = {}, {}
        for task in range(1, self.num_tasks + 1):
            trains[task] = SplitDataset(task, self.classes_per_task, mnist_train)
            tests[task] = SplitDataset(task, self.classes_per_task, mnist_test)

        return trains, tests

    @classmethod
    def from_config(cls, config):
        kwargs = cls.get_default_kwargs(config)
        return cls(**kwargs)

    def __repr__(self) -> str:
        return f"SplitMNIST(num_tasks={self.num_tasks}, batch_size={self.batch_size})"


class PermutedMNIST(ContinualMNIST):
    """Permuted MNIST benchmark."""

    classes_per_task = 10

    @classmethod
    def from_config(cls, config):
        kwargs = cls.get_default_kwargs(config)
        return cls(**kwargs)

    def __repr__(self) -> str:
        return f"PermutedMNIST(num_tasks={self.num_tasks}, batch_size={self.batch_size})"

    def prepare_datasets(self):
        mnist_train = torchvision.datasets.MNIST(DEFAULT_DATASET_DIR, train=True, download=True)
        mnist_test = torchvision.datasets.MNIST(DEFAULT_DATASET_DIR, train=False, download=True)
        transforms = self.get_transforms(self.num_tasks)
        trains, tests = {}, {}
        for task in range(1, self.num_tasks + 1):
            trains[task] = ContinualVisionDataset(task, mnist_train.data, mnist_train.targets, transforms[task - 1])
            tests[task] = ContinualVisionDataset(task, mnist_test.data, mnist_test.targets, transforms[task - 1])
        return trains, tests

    def get_transforms(self, num_tasks):
        transforms = []
        for task in range(1, num_tasks + 1):
            transform = [T.ToTensor()]
            if task > 1:
                transform.append(PermuteTransform(torch.randperm(28 * 28)))
            transform.append(T.Normalize(self.MEAN, self.STD))
            transforms.append(T.Compose(transform))
        return transforms


class RotatedMNIST(ContinualMNIST):
    """Rotated MNIST benchmark."""

    classes_per_task = 10

    def __init__(self, num_tasks: int, per_task_rotation: Optional[float] = None, *args, **kwargs):
        self.per_task_rotation = per_task_rotation
        super().__init__(num_tasks=num_tasks, *args, **kwargs)

    @classmethod
    def from_config(cls, config):
        kwargs = cls.get_default_kwargs(config)
        kwargs["per_task_rotation"] = config.per_task_rotation
        return cls(**kwargs)

    def __repr__(self) -> str:
        return f"RotatedMNIST(num_tasks={self.num_tasks}, per_task_rotation={self.per_task_rotation}, batch_size={self.batch_size})"

    def get_transforms(self, num_tasks: int, per_task_rotation: float = None):
        warnings.warn(
            "The RotatedMNIST benchmark currently supports fixed rotations of `per_task_rotation` degrees. "
            "Randomly sampling degrees will be added."
        )
        if not per_task_rotation:
            per_task_rotation = 180.0 / num_tasks
        transforms = []
        for t in range(1, num_tasks + 1):
            rotation_degree = (t - 1) * per_task_rotation
            transform = T.Compose([RotationTransform(rotation_degree), T.ToTensor(), T.Normalize(self.MEAN, self.STD)])
            transforms.append(transform)
        return transforms

    def prepare_datasets(self):
        trains, tests = {}, {}
        mnist_train = torchvision.datasets.MNIST(DEFAULT_DATASET_DIR, train=True, download=True)
        mnist_test = torchvision.datasets.MNIST(DEFAULT_DATASET_DIR, train=False, download=True)
        transforms = self.get_transforms(self.num_tasks, self.per_task_rotation)
        for task in range(1, self.num_tasks + 1):
            trains[task] = ContinualVisionDataset(task, mnist_train.data, mnist_train.targets, transforms[task - 1])
            tests[task] = ContinualVisionDataset(task, mnist_test.data, mnist_test.targets, transforms[task - 1])
        return trains, tests
