from functools import partial
from typing import Optional
import jax
import torch

from .jax_base_algo import JaxBaseAlgorithm, cross_entropy_loss
from sequel.benchmarks.buffer import Buffer
from flax.training.train_state import TrainState
from jax.tree_util import tree_map, tree_reduce
import jax.numpy as jnp
import optax


class DER(JaxBaseAlgorithm):
    """Dark Experience Replay algorithm implemented in JAX.

    The equivalent PyTorch implementation is [`DER in Pytorch`][sequel.algos.pytorch.der.DER].

    References:
        [1] Buzzega, P., Boschini, M., Porrello, A., Abati, D. & Calderara, S. Dark experience for general continual
            learning: a strong, simple baseline. in Advances in neural information processing systems 2020.
    """

    def __init__(self, memory_size: int, alpha: float, beta: Optional[float] = None, *args, **kwargs):
        """Inits the DER class. Implements the Dark Experience Replay algorithm.

        Args:
            memory_size (int): The size of the memory.
            alpha (float): The regularization coefficient for the DER objective.
            beta (Optional[float], optional): The regulrization coefficent for the DER++ objective. If set to None or
                zero, the algorithm corresponds to DER. Defaults to None.
        """

        super().__init__(*args, **kwargs)
        self.buffer = Buffer(memory_size=memory_size, return_logits=True)
        self.memory_size = memory_size
        self.alpha = alpha

        # Beta is used for DER++
        self.beta = beta

    def __repr__(self) -> str:
        if self.beta is None:
            return f"DER(memory_size={self.memory_size}, alpha={self.alpha})"
        else:
            return f"DER++(memory_size={self.memory_size}, alpha={self.alpha}, beta={self.beta})"

    @partial(jax.jit, static_argnums=(0,))
    def der_loss(self, params, x, y, t):
        # TODO: add task id support
        dropout_train_key = jax.random.fold_in(key=self.dropout_key, data=self.state.step)
        logits = self.apply_fn(params, x=x, training=self.is_training, rngs={"dropout": dropout_train_key})
        loss = cross_entropy_loss(logits=logits, labels=y)
        # DER LOSS
        dropout_key = jax.random.fold_in(key=dropout_train_key, data=self.state.step)
        mem_y_hat = self.apply_fn(params, x=x, training=self.is_training, rngs={"dropout": dropout_key})
        der_loss = jnp.mean((self.mem_logits - mem_y_hat) ** 2)

        loss += self.alpha * der_loss
        return loss, logits

    @partial(jax.jit, static_argnums=(0,))
    def derpp_loss(self, params, x, y, t):
        # TODO: add task id support
        dropout_key = jax.random.fold_in(key=self.dropout_key, data=self.state.step)
        logits = self.apply_fn(params, x=x, t=t, training=self.is_training, rngs={"dropout": dropout_key})
        loss = cross_entropy_loss(logits=logits, labels=y)
        # DER LOSS
        dropout_key = jax.random.fold_in(key=dropout_key, data=self.state.step)
        mem_y_hat = self.apply_fn(params, x=self.mem_x, training=self.is_training, rngs={"dropout": dropout_key})
        der_loss = jnp.mean((self.mem_logits - mem_y_hat) ** 2)

        # DER++ LOSS
        dropout_key = jax.random.fold_in(key=dropout_key, data=self.state.step)
        mem_y_hat2 = self.apply_fn(params, x=x, training=self.is_training, rngs={"dropout": dropout_key})
        derpp_loss = cross_entropy_loss(logits=mem_y_hat2, labels=self.mem_y2)

        loss += self.alpha * der_loss + self.beta * derpp_loss
        return loss, logits

    @partial(jax.jit, static_argnums=(0, 5))
    def custom_training_step(self, state: TrainState, x, y, t, loss_fn):
        """Train for a single step."""
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True, allow_int=True)
        (loss, logits), grads = grad_fn(state.params, x=x, y=y, t=t)
        state = state.apply_gradients(grads=grads)
        return dict(state=state, logits=logits, loss=loss, grads=grads)

    def training_step(self, *args, **kwargs):
        if self.task_counter == 1:
            self.batch_outputs = self.base_training_step(self.state, self.x, self.y, self.t)
        else:
            x, y, t, logits = self.buffer.sample_from_buffer(batch_size=self.benchmark.batch_size)
            self.mem_x, self.mem_y, self.mem_t, self.mem_logits = x, y, t, logits
            if self.beta is None:
                self.batch_outputs = self.custom_training_step(self.state, self.x, self.y, self.t, self.der_loss)
            else:
                x, y, t, _ = self.buffer.sample_from_buffer(batch_size=self.benchmark.batch_size)
                self.mem_x2, self.mem_y2, self.mem_t2 = x, y, t
                self.batch_outputs = self.custom_training_step(self.state, self.x, self.y, self.t, self.derpp_loss)
        self.register_batch_outputs(self.batch_outputs)

    def on_after_training_step(self, *args, **kwargs):
        self.buffer.add_data(self.x, self.y, self.t, self.y_hat)
