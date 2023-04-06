import copy
from functools import partial
from typing import Dict, Literal

import jax
import numpy as np
import optax
from flax.training.train_state import TrainState
from jax.tree_util import tree_map
from tqdm import tqdm

from sequel.benchmarks.memory import MemoryMechanism

from .jax_base_algo import JaxBaseAlgorithm


class MCSGD(JaxBaseAlgorithm):
    """MC-SGD: Mode Connectivity-Stochastic Gradient Descent. Inherits from BaseAlgorithm.

    The equivalent PyTorch implementation is [`MCSGD in Pytorch`][sequel.algos.pytorch.mcsgd.MCSGD].

    References:
        [1] Mirzadeh, S.-I., Farajtabar, M., Görür, D., Pascanu, R. & Ghasemzadeh, H. Linear Mode Connectivity in
            Multitask and Continual Learning. in 9th International Conference on Learning Representations, ICLR 2021.
    """

    state: TrainState

    def __init__(
        self,
        per_task_memory_samples: int = 100,
        memory_group_by: Literal["task", "class"] = "task",
        lmc_policy="offline",
        lmc_interpolation="linear",
        lmc_lr=0.05,
        lmc_momentum=0.8,
        lmc_batch_size=64,
        lmc_init_position=0.1,
        lmc_line_samples=10,
        lmc_epochs=1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.memory = MemoryMechanism(per_task_memory_samples=per_task_memory_samples, groupby=memory_group_by)
        self.w_bar_prev = None
        self.w_hat_curr = None

        # parse init arguments
        self.per_task_memory_samples = per_task_memory_samples
        self.lmc_policy = lmc_policy
        self.lmc_interpolation = lmc_interpolation
        self.lmc_lr = lmc_lr
        self.lmc_momentum = lmc_momentum
        self.lmc_batch_size = lmc_batch_size
        self.lmc_init_position = lmc_init_position
        self.lmc_line_samples = lmc_line_samples
        self.lmc_epochs = lmc_epochs

    def __repr__(self) -> str:
        return (
            "MCSGD("
            + f"per_task_memory_samples={self.per_task_memory_samples}, "
            + f"policy={self.lmc_policy}, "
            + f"interpolation={self.lmc_interpolation}, "
            + f"lr={self.lmc_lr}, "
            + f"momentum={self.lmc_momentum}, "
            + f"batch_size={self.lmc_batch_size}, "
            + f"init_position={self.lmc_init_position}, "
            + f"line_samples={self.lmc_line_samples}, "
            + f"epochs={self.lmc_epochs}"
            + ")"
        )

    def calculate_line_loss(self, w_start, w_end, loader):
        line_samples = np.arange(0.0, 1.01, 1.0 / float(self.lmc_line_samples))
        grads = tree_map(lambda x: 0 * x, w_start)
        for t in tqdm(line_samples, desc="Line samples"):
            params = tree_map(lambda a, b: a + (b - a) * t, w_start, w_end)
            g = self.calculate_point_loss(params, loader)
            grads = tree_map(lambda a, b: a + b, grads, g)
        return grads

    @partial(jax.jit, static_argnums=(0,))
    def simple_training_step(self, params, x, y, t, step):
        grad_fn = jax.value_and_grad(self.cross_entropy, has_aux=True, allow_int=True)
        (loss, logits), grads = grad_fn(params, x, y, t, self.is_training, step=step)
        return grads

    def calculate_point_loss(self, params, loader):
        total_count = 0.0
        grads = tree_map(lambda x: 0 * x, params)
        for batch in loader:
            self.unpack_batch(batch)
            g = self.simple_training_step(params, self.x, self.y, self.t, self.step_counter)
            grads = tree_map(lambda a, b: a + b, grads, g)
            total_count += self.bs

        return tree_map(lambda a: a / total_count, grads)

    def find_connected_minima(self, task):
        bs = self.lmc_batch_size
        loader_curr = self.benchmark.train_dataloader_subset(
            task, batch_size=bs, subset_size=self.per_task_memory_samples
        )
        loader_prev = self.benchmark.memory_dataloader(task, batch_size=bs, return_infinite_stream=False)

        params = tree_map(lambda a, b: a + (b - a) * self.lmc_init_position, self.w_bar_prev, self.w_hat_curr)
        tx = optax.sgd(learning_rate=self.lmc_lr, momentum=self.lmc_momentum)
        state = TrainState.create(apply_fn=self.apply_fn, params=params, tx=tx)

        grads_prev = self.calculate_line_loss(self.w_bar_prev, state.params, loader_prev)
        grads_curr = self.calculate_line_loss(self.w_hat_curr, state.params, loader_curr)

        grads = tree_map(lambda a, b: a + b, grads_prev, grads_curr)
        state = state.apply_gradients(grads=grads)
        return state

    def on_after_training_epoch(self, *args, **kwargs):
        self.w_hat_curr = copy.deepcopy(self.state.params)
        self.old_state = copy.deepcopy(self.state)

    def validate_algorithm_on_all_tasks(self) -> Dict[str, float]:
        if self.task_counter == 1:
            super().validate_algorithm_on_all_tasks()

    def on_after_validating_algorithm_on_all_tasks_callbacks(self):
        if self.task_counter == 1:
            return super().on_after_validating_algorithm_on_all_tasks_callbacks()

    def on_after_training_task(self, *args, **kwargs):
        self.memory.update_memory(self)
        if self.task_counter > 1:
            # save the backbone obtained from the mode-connectivity updates
            # as the Multi-Task approximate solution

            self.w_bar_prev = self.find_connected_minima(self.task_counter).params
            # perform the validation with the weights obtained after the mode-connectivity updates
            self.state = self.state.replace(params=self.w_bar_prev)
            super().on_before_validating_algorithm_on_all_tasks_callbacks()
            super().validate_algorithm_on_all_tasks()
            super().on_after_validating_algorithm_on_all_tasks_callbacks()
        else:
            self.w_bar_prev = copy.deepcopy(self.state.params)

        # revert the weights of the backbone to the Continual Learning solution
        self.state = copy.deepcopy(self.old_state)
