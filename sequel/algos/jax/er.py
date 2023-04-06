import logging
from typing import Literal

import numpy as np
from .jax_base_algo import JaxBaseAlgorithm
from sequel.benchmarks.memory import MemoryMechanism


class ER(JaxBaseAlgorithm):
    def __init__(
        self,
        per_task_memory_samples: int,
        memory_batch_size: int,
        memory_group_by: Literal["task", "class"],
        *args,
        **kwargs
    ):

        super().__init__(*args, **kwargs)
        self.memory = MemoryMechanism(per_task_memory_samples=per_task_memory_samples, groupby=memory_group_by)
        self.per_task_memory_samples = per_task_memory_samples
        self.memory_batch_size = memory_batch_size

    def on_after_training_task(self, *args, **kwargs):
        self.memory.update_memory(self)
        self.update_episodic_memory()
        logging.info("The episodic memory now stores {} samples".format(len(self.episodic_memory_loader.dataset)))

    def update_episodic_memory(self):
        logging.info("Updating episodic memory for task {}".format(self.task_counter))
        self.episodic_memory_loader = self.benchmark.memory_dataloader(self.task_counter, self.memory_batch_size)
        self.episodic_memory_iter = iter(self.episodic_memory_loader)

    def sample_batch_from_memory(self):
        try:
            return next(self.episodic_memory_iter)
        except StopIteration:
            # makes the dataloader an infinite stream
            self.episodic_memory_iter = iter(self.episodic_memory_loader)
            return next(self.episodic_memory_iter)

    def on_before_training_step(self, *args, **kwargs):
        if self.task_counter > 1:
            self.orig_x, self.orig_y, self.orig_t, self.orig_bs = self.x, self.y, self.t, self.bs

            mem_batch = self.sample_batch_from_memory()
            self.mem_x, self.mem_y, self.mem_t = self.unpack_batch_functional(mem_batch)

            self.x = np.concatenate([self.x, self.mem_x])
            self.y = np.concatenate([self.y, self.mem_y])
            self.t = np.concatenate([self.t, self.mem_t])
            self.bs = len(self.y)
