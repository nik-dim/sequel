import logging
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from sequel.benchmarks.memory import MemoryMechanism

from .jax_base_algo import JaxBaseAlgorithm


@jax.jit
def dot_product(tree_a, tree_b):
    tree = jax.tree_util.tree_map(lambda a, b: a * b, tree_a, tree_b)
    return sum(jnp.sum(x) for x in jax.tree_util.tree_leaves(tree))


class AGEM(JaxBaseAlgorithm):
    """A-GEM: Averaged-Gradient Episodic Memory. Maintains a memory of samples from past tasks.
    The gradients for the current batch are projected to the convex hull of the task gradients
    produced by the the aforementioned memory. Inherits from BaseAlgorithm.

    The equivalent PyTorch implementation is [`A-GEM in Pytorch`][sequel.algos.pytorch.agem.AGEM].

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
        """Inits the A-GEM algorithm class.

        Args:
            per_task_memory_samples (int): number of exemplars per experience in the memory.
            memory_batch_size (int): the batch size of the memory samples used to modify the gradient update.
            memory_group_by (Literal["task", "class"]): Determines the selection process of samples for the memory.
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
        self.episodic_memory_loader = self.benchmark.memory_dataloader(
            self.task_counter,
            self.memory_batch_size,
            return_infinite_stream=True,
        )
        self.episodic_memory_iter = iter(self.episodic_memory_loader)

    def sample_batch_from_memory(self):
        try:
            return next(self.episodic_memory_iter)
        except StopIteration:
            # makes the dataloader an infinite stream
            # The exception is only reached if the argument `return_infinite_stream` is set to False in
            # [`memory_dataloader`][sequel.benchmarks.base_benchmark.return_infinite_stream] set in
            # [`update_episodic_memory`][sequel.algos.jax.agem.update_episodic_memory].
            self.episodic_memory_iter = iter(self.episodic_memory_loader)
            return next(self.episodic_memory_iter)

    def training_step(self):
        if self.task_counter == 1:
            super().training_step()
        else:
            self.batch_outputs = self.agem_training_step(
                self.state, self.x, self.y, self.t, self.mem_x, self.mem_y, self.mem_t, self.step_counter
            )
            self.register_batch_outputs(self.batch_outputs)

    def on_before_training_step(self, *args, **kwargs):
        if self.task_counter > 1:
            batch = self.sample_batch_from_memory()
            x, y, t = self.unpack_batch_functional(batch)
            self.mem_x, self.mem_y, self.mem_t = x, y, t

    @partial(jax.jit, static_argnums=(0,))
    def agem_training_step(self, state: TrainState, x, y, t, mem_x, mem_y, mem_t, step):
        """The A-GEM training step that uses the memory samples to modify the gradient.

        Note:
            this implementation is suboptimal since it computes mem_norm and performs the tree_map operation even if not
            needed (case of dotg nonnegative). However, it has been implemented in this way in order to jit in a single
            function the gradient updates.
        """
        grad_fn = jax.value_and_grad(self.cross_entropy, has_aux=True, allow_int=True)

        (loss, logits), old_grads = grad_fn(state.params, x, y, t, self.is_training, step=step)
        # 1000000 is added so that steps are different. This applie for the rng of some modules, e.g. dropout
        _, mem_grads = grad_fn(state.params, mem_x, mem_y, mem_t, self.is_training, step=step + 1000000)
        dotg = jnp.minimum(dot_product(old_grads, mem_grads), 0)
        mem_norm = dot_product(mem_grads, mem_grads)

        alpha = dotg / mem_norm
        grads = jax.tree_util.tree_map(lambda o, m: o - m * alpha, old_grads, mem_grads)

        state = state.apply_gradients(grads=grads)
        return dict(state=state, logits=logits, loss=loss, grads=grads)
