import copy
from functools import partial

import jax
import jax.numpy as jnp
from .jax_base_algo import JaxBaseAlgorithm, cross_entropy_loss
from flax.training.train_state import TrainState


class LFL(JaxBaseAlgorithm):
    """Less-Forgetting Learning implementation in JAX.

    The equivalent PyTorch implementation is [`LFL in Pytorch`][sequel.algos.pytorch.lfl.LFL].

    References:
        [1] Jung, H., Ju, J., Jung, M. & Kim, J. Less-forgetful learning for domain expansion in deep neural
            networks. Proceedings of the AAAI Conference on Artificial Intelligence 32, (2018).
    """

    def __init__(self, lfl_lambda: float, *args, **kwargs):
        """Inits the LFL class.

        Args:
            lfl_lambda (float): the regularization coefficient.
        """
        super().__init__(*args, **kwargs)
        self.regularization_coefficient = lfl_lambda

    def __repr__(self) -> str:
        return f"LFL(regularization_coefficient={self.regularization_coefficient})"

    def on_after_training_task(self, *args, **kwargs):
        # freeze previous model
        # assert isinstance
        self.prev_params = copy.deepcopy(self.state.params)

    @partial(jax.jit, static_argnums=(0,))
    def lfl_loss(self, params, x, y, t):
        dropout_train_key = jax.random.fold_in(key=self.dropout_key, data=self.state.step)

        logits = self.apply_fn(params, x=x, training=self.is_training, rngs={"dropout": dropout_train_key})
        loss = cross_entropy_loss(logits=logits, labels=y)

        # disable dropout, etc for features. Equivalent to model.eval() in PyTorch
        features = self.apply_fn(
            params,
            x,
            training=False,
            method=lambda module, x, training: module.encoder(x, training),
        )

        old_features = self.apply_fn(
            self.prev_params,
            x,
            training=False,
            method=lambda module, x, training: module.encoder(x, training),
        )

        lfl_loss = jnp.mean((old_features - features) ** 2)
        loss += self.regularization_coefficient * lfl_loss
        return loss, logits

    @partial(jax.jit, static_argnums=(0,))
    def lfl_training_step(self, state: TrainState, x, y, t):
        """Train for a single step."""
        grad_fn = jax.value_and_grad(self.lfl_loss, has_aux=True, allow_int=True)
        (loss, logits), grads = grad_fn(state.params, x=x, y=y, t=t)
        state = state.apply_gradients(grads=grads)
        return dict(state=state, logits=logits, loss=loss, grads=grads)

    def training_step(self):
        if self.task_counter == 1:
            self.batch_outputs = self.base_training_step(self.state, self.x, self.y, self.t)
        else:
            self.batch_outputs = self.lfl_training_step(self.state, self.x, self.y, self.t)
        self.register_batch_outputs(self.batch_outputs)
