import logging
from typing import Literal

import torch

from sequel.algos.pytorch.pytorch_base_algo import PytorchBaseAlgorithm
from sequel.benchmarks.memory import MemoryMechanism

from .utils.weight_gradient_manipulation import get_grads, set_grads


class AGEM(PytorchBaseAlgorithm):
    """A-GEM: Averaged-Gradient Episodic Memory. Maintains a memory of samples from past tasks.
    The gradients for the current batch are projected to the convex hull of the task gradients
    produced by the the aforementioned memory. Inherits from BaseAlgorithm.

    The equivalent JAX implementation is [`A-GEM in JAX`][sequel.algos.jax.agem.AGEM].

    References:
        [1] Chaudhry, A., Ranzato, M., Rohrbach, M. & Elhoseiny, M. Efficient Lifelong Learning with A-GEM. in 7th
            International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019.
    """

    def __init__(
        self,
        per_task_memory_samples: int,
        memory_batch_size: int,
        memory_group_by: Literal["task", "class"],
        *args,
        **kwargs,
    ):
        """Inits the AGEM algorithm class.

        Args:
            per_task_memory_samples (int): number of exemplars per experience in the memory.
        """
        super().__init__(*args, **kwargs)
        self.memory = MemoryMechanism(per_task_memory_samples=per_task_memory_samples, groupby=memory_group_by)
        self.per_task_memory_samples = per_task_memory_samples
        self.memory_batch_size = memory_batch_size

    def __repr__(self) -> str:
        return (
            f"AGEM(memory_batch_size={self.memory_batch_size}, per_task_memory_samples={self.per_task_memory_samples})"
        )

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

    def on_before_optimizer_step(self, *args, **kwargs):
        if self.task_counter == 1:
            return

        # save gradients from current task and flush optimizer gradients
        old_grads = get_grads(self.backbone).detach().clone()
        self.optimizer_zero_grad()

        # sample from memory and compute corresponding gradients.
        x, y, t = self.sample_batch_from_memory()
        x, y = x.to(self.device), y.to(self.device)
        y_hat = self.backbone(x, t)
        loss = self.compute_loss(y_hat, y, t)
        loss.backward()

        # gradients from memory
        mem_grads = get_grads(self.backbone).detach().clone()

        assert old_grads.shape == mem_grads.shape, "Different model parameters in AGEM projection"

        dotg = torch.dot(old_grads, mem_grads)
        if dotg < 0:
            # if current task and memory gradients have negative angle (negative cosine similarity),
            # perform the A-GEM projection.
            alpha2 = dotg / torch.dot(mem_grads, mem_grads)
            new_grads = old_grads - mem_grads * alpha2

            self.backbone = set_grads(self.backbone, new_grads)
        else:
            self.backbone = set_grads(self.backbone, old_grads)

        return super().on_before_optimizer_step(*args, **kwargs)
