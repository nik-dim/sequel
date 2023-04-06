import logging
import random
from typing import List, Optional, Tuple

import omegaconf
import torch
from numpy.random import randint
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from .utils import ContinualConcatDataset, ContinualDataset, ContinualSubsetDataset


class Benchmark:
    """Base class for Continual Learning datasets (called benchmarks).

    All benchmarks (e.g. PermutedMNIST, SplitCifar100 etc) inherit from this class. It implements basic dataset and
    memory handling, such as splitting the original dataset into task datasets (train+val), constructing dataloaders
    which include one or multiple task datasets, dataloaders only for memory samples, dataloaders for one or multiple
    task datasets augmented with memory samples and more!
    """

    @staticmethod
    def get_default_kwargs(config: omegaconf.ListConfig) -> dict:
        """Utility function that covers the standard arguments for the construction of a benchmark. Used implicilty by
        benchmark selectors.

        Args:
            config (omegaconf.ListConfig): The user-specified experiment configuration.

        Returns:
            dict: a dictionary with argument key and value pairs for the construction of the benchmark.
        """
        kwargs = {}
        kwargs["num_tasks"] = config.num_tasks
        kwargs["batch_size"] = config.batch_size
        kwargs["eval_batch_size"] = getattr(config, "eval_batch_size", None)
        kwargs["num_workers"] = getattr(config, "num_workers", 2)
        kwargs["pin_memory"] = getattr(config, "pin_memory", True)
        kwargs["subset"] = getattr(config, "subset", None)

        return kwargs

    @property
    def dimensions(self) -> List[int]:
        raise NotImplementedError

    @property
    def num_classes(self) -> int:
        raise NotImplementedError

    def __init__(
        self,
        num_tasks: int,
        batch_size: int,
        eval_batch_size: int = None,
        num_workers: int = 0,
        pin_memory: bool = True,
        subset: Optional[int] = None,
    ):
        """Inits the base Benchmark class.

        Args:
            num_tasks (int): the number of tasks in the benchmark.
            batch_size (int, optional): The train dataloader batch size. Defaults to 256.
            eval_batch_size (int, optional): The validation dataloader batch size. If None, `eval_batch_size` is set to
                `batch_size`. Defaults to None.
            num_workers (int, optional): Dataloader number of workers. Defaults to 0.
            pin_memory (bool, optional): pin_memory argument for dataloaders. Defaults to True.
        """

        self.num_tasks = num_tasks

        # dataloader arguments
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.subset = subset
        self.dl_kwargs = dict(pin_memory=pin_memory, num_workers=num_workers)

        # set up
        self.trains, self.tests = self.prepare_datasets()
        self.memory_indices = {}

        if subset is not None:
            logging.info("Setting up the subset indices.")
            assert isinstance(subset, int) and subset > 0
            self.subset_indices = {}
            for k, v in self.trains.items():
                indices = list(range(len(v)))
                random.shuffle(indices)
                self.subset_indices[k] = indices[:subset]

    def __check_valid_task__(self, task: int):
        if task > self.num_tasks:
            raise ValueError(f"Asked to load task {task} but the benchmark has {self.num_tasks} tasks")

    @classmethod
    def from_config(cls, config: omegaconf.OmegaConf, *args, **kwargs):
        raise NotImplementedError

    def prepare_datasets(self) -> Tuple[ContinualDataset, ContinualDataset]:
        raise NotImplementedError

    def get_memory_indices(self, task: int) -> torch.Tensor:
        return self.memory_indices[task]

    def set_memory_indices(self, task: int, indices: List[int]) -> None:
        self.memory_indices[task] = indices

    def get_train_dataset(self, task: int) -> ContinualDataset:
        return self.trains[task]

    def get_test_dataset(self, task: int) -> ContinualDataset:
        return self.tests[task]

    def get_memories(self, task: int) -> ContinualConcatDataset:
        """Returns a `ContinualConcatDataset` containing all the memory samples for tasks up to `task`.

        Args:
            task (int): The current task. The final dataset consists of all memory samples up to the specified id.

        Returns:
            ContinualConcatDataset: The constructed concatenated dataset.
        """
        self.__check_valid_task__(task)
        memories = [
            ContinualSubsetDataset(self.get_train_dataset(t), self.memory_indices[t]) for t in range(1, task + 1)
        ]
        return ContinualConcatDataset(memories)

    def train_dataloader(self, task: int, batch_size: Optional[int] = None) -> DataLoader:
        """Constructs the train dataloader for the current task.

        Args:
            task (int): the current task id
            batch_size (Optional[int], optional): The batch size used for both dataloaders. If set to None, the
                benchmark batch size is used. Defaults to None.

        Returns:
            DataLoader: the constructed DataLoader
        """
        self.__check_valid_task__(task)

        if batch_size is None:
            batch_size = self.batch_size

        dataset = self.get_train_dataset(task)
        if self.subset:
            indices = self.subset_indices[task]
            logging.info(f"Extracting subset [{len(indices)}/{len(dataset)} samples] for train dataset of task {task}")
            dataset = ContinualSubsetDataset(dataset, indices)
        return DataLoader(dataset, batch_size, shuffle=True, **self.dl_kwargs)

    def val_dataloader(self, task: int, batch_size: Optional[int] = None) -> DataLoader:
        """Constructs the val dataloader for the current task.

        Args:
            task (int): the current task id
            batch_size (Optional[int], optional): The batch size used for both dataloaders. If set to None, the
                benchmark batch size is used. Defaults to None.

        Returns:
            DataLoader: the constructed DataLoader
        """
        self.__check_valid_task__(task)

        if batch_size is None:
            batch_size = self.eval_batch_size

        dataset = self.get_test_dataset(task)
        return DataLoader(dataset, batch_size, **self.dl_kwargs)

    def train_dataloader_subset(
        self, task: int, subset_size: Optional[int] = None, batch_size: Optional[int] = None
    ) -> DataLoader:
        """Constructs a dataloader containing a random subset from the dataset indexed by id `task`.

        Args:
            task (int): the dataset task id.
            batch_size (Optional[int], optional): The batch size used for both dataloaders. If set to None, the
                benchmark batch size is used. Defaults to None.

        Returns:
            DataLoader: the constructed dataloader.
        """

        self.__check_valid_task__(task)

        if batch_size is None:
            batch_size = self.batch_size

        if subset_size is None:
            assert self.subset is not None
            subset_size = self.subset

        train_dataset = self.get_train_dataset(task)
        indices = torch.randperm(len(train_dataset))[:subset_size]
        # sampler = RandomSampler(train_dataset, replacement=True, num_samples=subset_size)
        train_dataset = ContinualSubsetDataset(train_dataset, indices=indices)
        return DataLoader(train_dataset, batch_size, **self.dl_kwargs)

    def train_dataloader_joint(self, task: int, batch_size: Optional[int] = None) -> DataLoader:
        """Constructs the train dataloader for the current task.

        Args:
            task (int): the current task id
            batch_size (Optional[int], optional): The batch size used for both dataloaders. If set to None, the
                benchmark batch size is used. Defaults to None.

        Returns:
            DataLoader: the constructed DataLoader
        """
        self.__check_valid_task__(task)

        if batch_size is None:
            batch_size = self.batch_size

        dataset = ContinualConcatDataset([self.get_train_dataset(t) for t in range(1, task + 1)])
        return DataLoader(dataset, batch_size, shuffle=True, **self.dl_kwargs)

    def memory_dataloader(
        self, task: int, batch_size: Optional[int] = None, return_infinite_stream: bool = True
    ) -> DataLoader:
        self.__check_valid_task__(task)
        dataset = self.get_memories(task)

        if batch_size is None:
            batch_size = self.batch_size

        if batch_size > len(dataset):
            batch_size = len(dataset)

        if return_infinite_stream:
            sampler = RandomSampler(dataset, replacement=True, num_samples=100**100)
            return DataLoader(dataset, batch_size, shuffle=False, sampler=sampler, **self.dl_kwargs)
        else:
            return DataLoader(dataset, batch_size, shuffle=True, **self.dl_kwargs)

    def train_dataloader_with_memory(
        self, task: int, batch_size: Optional[int] = None, verbose: bool = False
    ) -> DataLoader:
        """Constructs a dataloader consisting of samples coming from the current task as well as the memory samples for
        all previous tasks.

        Args:
            task (int): the current task id
            batch_size (Optional[int], optional): The dataloader batch size. If set to None, the benchmark batch size is used. Defaults to None.
            verbose (bool, optional): boolean indicating if the method will print additional information to the console. Defaults to False.

        Returns:
            Dataloader: the constructed PyTorch dataloader.
        """
        self.__check_valid_task__(task)

        current_train_dataset = self.get_train_dataset(task)
        memory = self.get_memories(task)
        dataset = ContinualConcatDataset([current_train_dataset, memory])
        if verbose:
            logging.info("Samples of train dataset:\t{}".format(len(current_train_dataset)))
            logging.info("Samples of memory:\t{}".format(len(memory)))
            logging.info("Samples of overall dataset:\t{}".format(len(dataset)))

        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(dataset, batch_size, **self.dl_kwargs)
