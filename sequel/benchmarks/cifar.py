from typing import List, Optional, Tuple

import torch
import torchvision
import torchvision.transforms as T

from . import DEFAULT_DATASET_DIR
from .base_benchmark import Benchmark
from .utils import ContinualDataset, SplitDataset

CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN, CIFAR100_STD = (0.5071, 0.4867, 0.4408), (0.2673, 0.2564, 0.2762)


class SplitCIFAR(Benchmark):
    """SplitCIFAR benchmarks."""

    @property
    def num_classes(self) -> int:
        return 100 if self.is_cifar_100 else 10

    @property
    def MEAN(self):
        if self.is_cifar_100:
            return CIFAR100_MEAN
        else:
            return CIFAR10_MEAN

    @property
    def STD(self):
        if self.is_cifar_100:
            return CIFAR100_STD
        else:
            return CIFAR10_STD

    @property
    def dimensions(self) -> List[int]:
        return [3, 32, 32]

    @property
    def num_classes(self):
        num_classes = 100 if self.is_cifar_100 else 10
        return num_classes

    @property
    def classes_per_task(self):
        assert self.num_classes % self.num_tasks == 0
        return self.num_classes // self.num_tasks

    def __init__(
        self,
        num_tasks: int,
        batch_size: int,
        fixed_class_order: Optional[List[int]] = None,
        is_cifar_100: bool = True,
        eval_batch_size: int = None,
        num_workers: int = 0,
        pin_memory: bool = True,
        subset: Optional[int] = None,
    ):
        """Inits the SplitCIFAR100/100 class. The `is_cifar100` boolean flag denotes which dataset is instantiated.

        Args:
            num_tasks (int): the number of tasks in the benchmark. Usually 20 for SplitCIFAR100 and 5 for SplitCIFAR10.
                Must be divisible by the number of classes.
            batch_size (int, optional): The train dataloader batch size. Defaults to 256.
            fixed_class_order (Optional[List[int]], optional): A list of integers denoting a custom fixed_class_order.
                If None, the alphabetical order is used. Defaults to None.
            is_cifar_100 (bool, optional): Boolean denoting whether SplitCIFAR100 or SplitCIFAR10 is selected. Defaults
                to True.
            eval_batch_size (int, optional): The validation dataloader batch size. If None, `eval_batch_size` is set to
                `batch_size`. Defaults to None.
            num_workers (int, optional): Dataloader number of workers. Defaults to 0.
            pin_memory (bool, optional): pin_memory argument for dataloaders. Defaults to True.
        """

        self.is_cifar_100 = is_cifar_100

        if fixed_class_order is None:
            fixed_class_order = list(range(self.num_classes))
        assert (
            torch.tensor(fixed_class_order).sort()[0] == torch.arange(0, self.num_classes)
        ).all(), "The fixed_class_order argument muct contain exactly once all integers from 0 to (num_classes-1)."

        self.fixed_class_order = fixed_class_order
        super().__init__(
            num_tasks=num_tasks,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            subset=subset,
        )

    def prepare_datasets(self) -> Tuple[ContinualDataset, ContinualDataset]:
        transform = T.Compose([T.ToTensor(), T.Normalize(self.MEAN, self.STD)])
        CIFAR_dataset = torchvision.datasets.CIFAR100 if self.is_cifar_100 else torchvision.datasets.CIFAR10
        self.cifar_train = CIFAR_dataset(DEFAULT_DATASET_DIR, train=True, download=True, transform=transform)
        self.cifar_test = CIFAR_dataset(DEFAULT_DATASET_DIR, train=False, download=True, transform=transform)

        self.trains, self.tests = {}, {}
        for t in range(1, self.num_tasks + 1):
            self.trains[t] = SplitDataset(t, self.classes_per_task, self.cifar_train, self.fixed_class_order)
            self.tests[t] = SplitDataset(t, self.classes_per_task, self.cifar_test, self.fixed_class_order)

        return self.trains, self.tests

    @classmethod
    def from_config(cls, config):
        # breakpoint()
        kwargs = cls.get_default_kwargs(config)
        kwargs["is_cifar_100"] = "100" in config.name
        kwargs["fixed_class_order"] = getattr(config, "fixed_class_order", None)
        return cls(**kwargs)

    def __repr__(self) -> str:
        return f"SplitCIFAR{self.num_classes}(num_tasks={self.num_tasks}, batch_size={self.batch_size})"


class SplitCIFAR10(SplitCIFAR):
    def __init__(self, is_cifar_100=False, *args, **kwargs):
        """Helper class for SplitCIFAR10. Inherits from SplitCIFAR. Look at the parent class for a description of the
        class arguments.

        Args:
            is_cifar_100 (bool, optional): Set to False.
        """
        super().__init__(is_cifar_100=is_cifar_100, *args, **kwargs)


class SplitCIFAR100(SplitCIFAR):
    def __init__(self, is_cifar_100=True, *args, **kwargs):
        """Helper class for SplitCIFAR100. Inherits from SplitCIFAR. Look at the parent class for a description of the
        class arguments.

        Args:
            is_cifar_100 (bool, optional): Set to True.
        """
        super().__init__(is_cifar_100=is_cifar_100, *args, **kwargs)
