import copy

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

from .jax_base_algo import JaxRegularizationBaseAlgorithm


class MAS(JaxRegularizationBaseAlgorithm):
    """Memory Aware Synapses. Algorithm Class. Inherits from BaseAlgorithm.

    The equivalent PyTorch implementation is [`MAS in Pytorch`][sequel.algos.pytorch.mas.MAS].

    References:
        [1] Aljundi, R., Babiloni, F., Elhoseiny, M., Rohrbach, M. & Tuytelaars, T. Memory Aware Synapses: Learning
            What (not) to Forget. in Computer Vision - ECCV 2018.
    """

    def __init__(self, mas_lambda: float = 1.0, *args, **kwargs):
        super().__init__(regularization_coefficient=mas_lambda, *args, **kwargs)
        self.w = tree_map(lambda x: 0 * x, self.state.params)

    def __repr__(self) -> str:
        return f"MAS(mas_lambda={self.regularization_coefficient})"

    def calculate_parameter_importance(self):
        if self.task_counter == 1:
            importance = tree_map(lambda x: 0 * x, self.state.params)
        else:
            importance = self.importance
        importance = tree_map(lambda i, w: i + w, importance, self.w)
        self.w = tree_map(lambda x: 0 * x, self.state.params)
        return importance

    def on_before_training_step(self, *args, **kwargs):
        self.old_params = copy.deepcopy(self.state.params)

    def on_after_training_step(self, *args, **kwargs):
        @jax.jit
        def secondary_loss(params, x, t, training=True):
            logits = self.apply_fn(params, x, t, training=training)
            loss = jnp.mean(jnp.square(logits))
            return loss, logits

        grad_fn = jax.value_and_grad(secondary_loss, has_aux=True, allow_int=True)
        _, grads = grad_fn(self.state.params, self.x, self.t, self.is_training)
        self.w = tree_map(lambda w, g: w + jnp.abs(g) / len(self.y), self.w, grads)

    def on_after_training_task(self, *args, **kwargs):
        self.old_params = copy.deepcopy(self.state.params)
        self.importance = self.calculate_parameter_importance()
