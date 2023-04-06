import logging
from typing import TYPE_CHECKING, List

import numpy as np
from torch.utils.data import random_split

from .utils import ContinualDataset

if TYPE_CHECKING:
    from sequel.algos.base_algo import BaseAlgorithm


class MemoryMechanism:
    """Implements the memory handling/manipulation for continual learning algorithms."""

    def __init__(self, per_task_memory_samples: int, groupby: str = "class"):
        logging.info("Initializing MemoryCallback")
        self.per_task_memory_samples = per_task_memory_samples

        if groupby not in ("task", "class"):
            raise ValueError("Only class and task are supported as options for groupby argument.")

        self.groupby = groupby

    def update_memory(self, algo: "BaseAlgorithm"):
        """Updates the memory by selecting `per_task_memory_samples` samples from the current dataset. The selection
        process is defined by the `groupby` instance attribute.

        Args:
            algo (BaseAlgorithm): the algorithm instance.
        """
        logging.info("Setting memory indices for task {}".format(algo.task_counter))
        task = algo.task_counter
        dataset = algo.benchmark.get_train_dataset(task)
        if self.groupby == "class":
            memory_indices = MemoryMechanism.sample_uniform_class_indices(dataset, self.per_task_memory_samples)
        else:
            memory_indices = MemoryMechanism.sample_uniform_task_indices(dataset, self.per_task_memory_samples)
        algo.benchmark.set_memory_indices(task, memory_indices)

    def update_memory_(self, benchmark, task):
        """Updates the memory by selecting `per_task_memory_samples` samples from the current dataset. The selection
        process is defined by the `groupby` instance attribute.
        """
        logging.info("Setting memory indices for task {}".format(task))
        dataset = benchmark.get_train_dataset(task)
        if self.groupby == "class":
            memory_indices = MemoryMechanism.sample_uniform_class_indices(dataset, self.per_task_memory_samples)
        else:
            memory_indices = MemoryMechanism.sample_uniform_task_indices(dataset, self.per_task_memory_samples)
        benchmark.set_memory_indices(task, memory_indices)

    @staticmethod
    def sample_uniform_task_indices(dataset: ContinualDataset, num_samples: int) -> List[int]:
        """Selects a specified number of indices uniformly at random.

        Args:
            dataset (ContinualDataset): The dataset that is sampled.
            num_samples (int): the number of samples to draw.

        Returns:
            List[int]: The selected dataset indices.
        """
        to_remove = len(dataset) - num_samples
        dataset, _ = random_split(dataset, [num_samples, to_remove])
        return dataset.indices

    @staticmethod
    def sample_uniform_class_indices(dataset: ContinualDataset, num_samples: int) -> List[int]:
        """Selects an approximately equal (ties broken arbitrarily) number of indices corresponding to each class from
        the input dataset. Each dataset yields ~num_samples // num_classes samples.

        Args:
            dataset (ContinualDataset): The dataset that is sampled.
            num_samples (int): the number of samples to draw.

        Returns:
            List[int]: The selected dataset indices.
        """
        target_classes = dataset.targets.clone().detach().numpy()
        classes = np.unique(target_classes).tolist()
        num_classes = len(classes)
        num_examples_per_class = MemoryMechanism.pack_bins_uniformly(num_samples * num_classes, num_classes)
        class_indices = []

        for class_id, cls_number in enumerate(classes):
            candidates = np.array([i for i, t in enumerate(target_classes) if t == cls_number])
            np.random.shuffle(candidates)

            selected_indices = candidates[: num_examples_per_class[class_id]]
            class_indices += list(selected_indices)
        return class_indices

    @staticmethod
    def pack_bins_uniformly(num_samples: int, num_categories: int) -> List[int]:
        """Splits an integer to a specified number of bins so that bins have approximately the same size. If
        `num_categories` is not a divisor of `num_samples`, the reminder is split an equal number of bins selectrd
        uniformly at random.

        Args:
            num_samples (int): The number of items.
            num_categories (int): the number of bins.

        Returns:
            List[int]: a list containing the number of items corresponding to each bin.
        """
        num_samples_per_cat = np.ones(num_categories) * num_samples // num_categories
        remaining = num_samples % num_categories
        correction_vector = np.array([0] * (num_categories - remaining) + [1] * remaining)
        np.random.shuffle(correction_vector)
        num_samples_per_cat += correction_vector
        return num_samples_per_cat.astype("int").tolist()
