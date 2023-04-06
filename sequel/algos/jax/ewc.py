import copy
import logging
from functools import partial

import jax
from jax.tree_util import tree_map

from .jax_base_algo import JaxRegularizationBaseAlgorithm


class EWC(JaxRegularizationBaseAlgorithm):
    """The Elastic Weight Consolidation algorithm.

    The equivalent PyTorch implementation is [`EWC in Pytorch`][sequel.algos.pytorch.ewc.EWC].

    References:
        [1] Kirkpatrick, J. et al. Overcoming catastrophic forgetting in neural networks. PNAS 114, 3521-3526 (2017).
    """

    def __init__(self, ewc_lambda: float, *args, **kwargs) -> None:
        super().__init__(regularization_coefficient=ewc_lambda, *args, **kwargs)

    def __repr__(self) -> str:
        return f"EWC(ewc_lambda={self.regularization_coefficient})"

    @partial(jax.jit, static_argnums=(0,))
    def fisher_training_step(self, state, x, y, t, step):
        grad_fn = jax.value_and_grad(self.cross_entropy, has_aux=True, allow_int=True)
        (loss, logits), grads = grad_fn(state.params, x, y, t, training=True, step=step)
        return grads

    def on_after_training_task(self, *args, **kwargs):
        self.train_loader = self.benchmark.train_dataloader(self.task_counter)
        # initialize fisher diagonals to zero
        fisher_diagonals = tree_map(lambda x: 0 * x, self.state.params)
        num_samples = 0
        for self.batch_idx, batch in enumerate(self.train_loader):
            self.unpack_batch(batch)
            num_samples += self.bs
            grads = self.fisher_training_step(self.state, self.x, self.y, self.t, self.step_counter)
            fisher_diagonals = tree_map(lambda a, b: a**2 + b, grads, fisher_diagonals)

        self.importance = tree_map(lambda x: x / num_samples, fisher_diagonals)
        self.old_params = copy.deepcopy(self.state.params)
