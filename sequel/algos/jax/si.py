import copy

import jax
from jax.tree_util import tree_map

from .jax_base_algo import JaxRegularizationBaseAlgorithm


class SI(JaxRegularizationBaseAlgorithm):
    """Synaptic Intelligence Algorithm.

    The equivalent PyTorch implementation is [`SI in Pytorch`][sequel.algos.pytorch.si.SI].

    References:
        [1] Zenke, F., Poole, B. & Ganguli, S. Continual Learning Through Synaptic Intelligence. in Proceedings of the
            34th International Conference on Machine Learning, ICML 2017.
    """

    def __init__(self, si_lambda: float = 1.0, xi: float = 0.1, *args, **kwargs):
        super().__init__(regularization_coefficient=si_lambda, *args, **kwargs)
        self.xi = xi
        self.w = tree_map(lambda a: 0 * a, self.state.params)

    def __repr__(self) -> str:
        return f"SI(si_lambda={self.regularization_coefficient}, xi={self.xi})"

    def calculate_parameter_importance(self):
        if self.task_counter == 1:
            importance = tree_map(lambda x: 0 * x, self.state.params)
        else:
            importance = self.importance
        delta = tree_map(lambda w_cur, w_old: w_cur - w_old, self.state.params, self.old_params)
        importance = tree_map(lambda i, w, dt: i + w / (dt**2 + self.xi), importance, self.w, delta)
        self.w = tree_map(lambda x: 0 * x, self.state.params)
        return importance

    def on_before_training_step(self, *args, **kwargs):
        self.prev_params = copy.deepcopy(self.state.params)

    # @partial(jax.jit, static_argnums=(0,))
    def on_after_training_step(self, *args, **kwargs):
        grads = self.batch_outputs["grads"]
        delta = tree_map(lambda w_cur, w_old: w_cur - w_old, self.state.params, self.prev_params)
        self.w = tree_map(lambda w, g, d: w - g * d, self.w, grads, delta)

    def on_after_training_task(self, *args, **kwargs):
        self.old_params = copy.deepcopy(self.state.params)
        self.importance = self.calculate_parameter_importance()
