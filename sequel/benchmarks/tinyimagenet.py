import logging
from typing import Optional

import torchvision
import torchvision.transforms as T

from sequel.benchmarks.base_benchmark import Benchmark

from .utils import SplitDataset

_default_input_transform = T.Compose([T.ToTensor()])

from . import DEFAULT_DATASET_DIR


class SplitTinyImagenet(Benchmark):

    root = DEFAULT_DATASET_DIR

    @property
    def num_classes(self) -> int:
        return 200

    def __init__(
        self,
        num_tasks: int = 10,
        task_input_transforms: Optional[list] = _default_input_transform,
        task_target_transforms: Optional[list] = None,
    ):
        """Inits the SplitTinyImagenet class. The number of `classes_per_task` is equal to the ratio of 200 and
        `num_tasks`.

        Args:
            num_tasks (int, optional): The number of tasks. Defaults to 10.
            task_input_transforms (Optional[list], optional): If set, the benchmark will use the
                provided torchvision input transform. Defaults to _default_input_transform.
            task_target_transforms (Optional[list], optional): If set, the benchmark will use the
                provided torchvision target transform. Defaults to None.

        Raises:
            ValueError: The number of tasks must be divisible by the number of classes (200).
        """
        if self.num_classes % num_tasks != 0:
            raise ValueError("The number of tasks must be divisible by the number of classes (200).")
        self.classes_per_task = self.num_classes // num_tasks

        super().__init__(
            num_tasks=num_tasks,
            task_input_transforms=task_input_transforms,
            task_target_transforms=task_target_transforms,
        )

        logging.info(f"Classes_per_task={self.classes_per_task}")
        logging.info(f"num_tasks={self.num_tasks}")

    @classmethod
    def from_config(cls, config):
        num_tasks = config.benchmark.num_tasks
        return cls(num_tasks=num_tasks)

    def prepare_datasets(self):
        trains, tests = {}, {}
        self.__load_tinyimagenet()
        for task in range(1, self.num_tasks + 1):
            trains[task] = SplitDataset(task, self.classes_per_task, self.tiny_train)
            tests[task] = SplitDataset(task, self.classes_per_task, self.tiny_test)

        return trains, tests

    def __load_tinyimagenet(self):
        """Loads the tinyimagenet dataset.

        The original dataset does not have labels for the test dataset. For this reason, the validation dataset is
        used.
        """
        self.tiny_train = torchvision.datasets.ImageFolder(self.root + "train", transform=self.task_input_transforms)
        tiny_val = torchvision.datasets.ImageFolder(self.root + "val", transform=self.task_input_transforms)
        self.tiny_test = tiny_val

    def __repr__(self) -> str:
        return f"SplitTinyImageNet(num_tasks={self.num_tasks}, batch_size={self.batch_size})"
