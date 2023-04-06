import copy
import logging
from functools import partial
from typing import Dict, Iterable, Optional
import warnings
from chex import PRNGKey

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from jax.tree_util import tree_map, tree_reduce

from sequel.algos.base_algo import BaseAlgorithm
from sequel.backbones.jax import BaseBackbone
from sequel.backbones.jax.base_backbone import BackboneWrapper
from sequel.backbones.jax.base_backbone import BaseBackbone as JaxBaseBackbone
from sequel.benchmarks.base_benchmark import Benchmark
from sequel.utils.callbacks.base_callback import BaseCallback
from sequel.utils.loggers.base_logger import Logger


def cross_entropy_loss(*, logits, labels, num_classes):
    labels_onehot = jax.nn.one_hot(labels, num_classes=num_classes)
    return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()


class JaxBaseAlgorithm(BaseAlgorithm):
    """Base class for algorithms implemented in JAX."""

    def __init__(
        self,
        backbone: JaxBaseBackbone,
        benchmark: Benchmark,
        optimizer: optax.GradientTransformation,
        callbacks: Iterable[BaseCallback] = [],
        loggers: Optional[Iterable[Logger]] = None,
        lr_decay: Optional[float] = None,
        grad_clip: Optional[float] = None,
        reinit_optimizer: bool = True,
        seed=0,
    ) -> None:
        """Inits JaxBaseAlgorithm class.

        Args:
            backbone (JaxBaseBackbone): The backbone model, e.g., a CNN.
            benchmark (Benchmark): The benchmark, e.g., SplitMNIST.
            optimizer (optax.GradientTransformation):  The optimizer used to update the backbone weights.
            callbacks (Iterable[BaseCallback], optional): A list of callbacks. At least one instance of MetricCallback
                should be given. Defaults to [].
            loggers (Optional[Logger], optional): A list of logger, e.g. for Weights&Biases logging functionality.
                Defaults to None.
            lr_decay (Optional[float], optional): A learning rate decay used for every new task. Defaults to None.
            grad_clip (Optional[float], optional): The gradient clipping norm. Defaults to None.
            reinit_optimizer (bool): Indicates whether the optimizer state is reinitialized before fitting a new task.
                Defaults to True.
            seed (int, optional): The seed used by JAX. Sets the corresponding `PRNGKey`. Defaults to 0.

        Note:
            1. the `_configure_optimizers` method will be moved to a dedicated Callback.
        """

        assert isinstance(backbone, BaseBackbone)
        super().__init__(
            backbone=backbone,
            benchmark=benchmark,
            optimizer=optimizer,
            callbacks=callbacks,
            loggers=loggers,
            lr_decay=lr_decay,
            grad_clip=grad_clip,
            reinit_optimizer=reinit_optimizer,
        )
        print(">" * 100)
        print(self.benchmark.num_classes)
        print(">" * 100)
        self.seed = seed
        rng = jax.random.PRNGKey(seed)
        self.rng, init_rng = jax.random.split(rng)
        self.state: TrainState = self.create_train_state(self.backbone, init_rng, task=None)
        self.apply_fn = self.state.apply_fn
        self.original_optimizer = copy.deepcopy(self.optimizer)

    def create_train_state(self, model: nn.Module, rng: PRNGKey, task=None) -> TrainState:
        """Creates initial `TrainState`."""
        dims = self.benchmark.dimensions
        dimensions = [1] + dims[1:] + [dims[0]]
        params = model.init(rng, x=jnp.ones(dimensions), task_ids=None, training=False)
        tx = self.optimizer

        rng, self.dropout_key = jax.random.split(rng)
        del rng

        return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    def prepare_for_next_task(self, task: int):
        if self.reinit_optimizer:
            logging.info("Reinitializing optimizer for next task")
            params = self.state.params
            apply_fn = self.state.apply_fn
            tx: optax.GradientTransformation = copy.deepcopy(self.original_optimizer)
            if self.lr_decay is not None and task > 1:
                assert isinstance(self.lr_decay, float)
                assert self.lr_decay > 0 and self.lr_decay <= 1, "lr decay should be in the interval (0,1]"
                new_lr = self.state.opt_state.hyperparams["learning_rate"] * self.lr_decay
                logging.info(f"Decaying the learning rate by a factor of {self.lr_decay} to the next lr={new_lr}")
            else:
                new_lr = self.state.opt_state.hyperparams["learning_rate"]
            self.state = TrainState.create(apply_fn=apply_fn, params=params, tx=tx)
            print(self.state.opt_state.hyperparams)
            self.state.opt_state.hyperparams["learning_rate"] = new_lr
            print(self.state.opt_state.hyperparams)

    def count_parameters(self):
        dims = self.benchmark.dimensions
        dimensions = [1] + dims[1:] + [dims[0]]
        print(dimensions)
        rng = jax.random.PRNGKey(0)
        params = self.backbone.init(rng, jnp.ones(dimensions), task_ids=None, training=False)
        return sum(x.size for x in jax.tree_util.tree_leaves(params))

    def _configure_criterion(self, task_id=None):
        logging.debug("_configure_criterion should change?")

    def unpack_batch(self, batch):
        self.x, self.y, self.t = self.unpack_batch_functional(batch)
        self.bs = len(self.x)

    def unpack_batch_functional(self, batch):
        x, y, t = batch
        if x.dim() > 2:
            # in case of image datasets
            x = x.permute(0, 2, 3, 1)
        return np.array(x), np.array(y), np.array(t)

    def perform_gradient_clipping(self):
        warnings.warn("Gradient Clipping has not been implemented for JAX.")
        pass

    @partial(jax.jit, static_argnums=(0, 4))
    def forward(self, params, x, t, training, step):
        dropout_train_key = jax.random.fold_in(key=self.dropout_key, data=step)
        logits = self.apply_fn(
            params,
            x=x,
            task_ids=t,
            training=training,
            rngs={"dropout": dropout_train_key},
            # This applies to ResNet; BathcNorm are not updated for the moment.
            mutable=False,
        )
        return logits

    @partial(jax.jit, static_argnums=(0, 5))
    def cross_entropy(self, params, x, y, t, training, step=None):
        logits = self.forward(params, x, t, training, step=step)
        loss = cross_entropy_loss(logits=logits, labels=y, num_classes=self.benchmark.num_classes)
        return loss, logits

    @partial(jax.jit, static_argnums=(0,))
    def base_training_step(self, state: TrainState, x, y, t, step):
        """Train for a single step."""
        grad_fn = jax.value_and_grad(self.cross_entropy, has_aux=True, allow_int=True)
        (loss, logits), grads = grad_fn(state.params, x=x, y=y, t=t, training=True, step=step)
        state = state.apply_gradients(grads=grads)
        return dict(state=state, logits=logits, loss=loss, grads=grads)

    def register_batch_outputs(self, batch_outputs):
        self.state = batch_outputs["state"]
        self.loss = batch_outputs["loss"]
        self.y_hat = batch_outputs["logits"]
        self.grads = batch_outputs["grads"]

    def training_step(self):
        self.batch_outputs = self.base_training_step(self.state, self.x, self.y, self.t, step=self.step_counter)
        self.register_batch_outputs(self.batch_outputs)

    @partial(jax.jit, static_argnums=(0,))
    def base_eval_step(self, state: TrainState, x, t):
        return state.apply_fn(state.params, x, t, training=False)

    def valid_step(self):
        self.y_hat = self.base_eval_step(self.state, self.x, self.t)


class Naive(JaxBaseAlgorithm):
    pass


class JaxRegularizationBaseAlgorithm(JaxBaseAlgorithm):
    """JaxRegularizationBaseAlgorithm inherits from `JaxBaseAlgorithm` and implements a few utility functions that are
    used by all regularization-based algorithms such as calculating the regularization loss and computing the
    per-parameter importance.
    """

    def __init__(self, regularization_coefficient: float, *args, **kwargs) -> None:
        """Base class for regularization-based algorithms implemented in JAX, such as EWC and SI

        Args:
            regularization_coefficient (float): the coefficient used to weigh the regularization loss.
        """
        super().__init__(*args, **kwargs)
        self.regularization_coefficient = regularization_coefficient
        self.old_params = None
        self.importance = None

    @partial(jax.jit, static_argnums=(0,))
    def calculate_regularization_loss(self, params):
        assert self.task_counter > 1
        return tree_reduce(
            lambda x, y: jnp.sum(x) + jnp.sum(y),
            tree_map(
                lambda a, b, w: jnp.sum(w * (a - b) ** 2.0),
                params,
                self.old_params,
                self.importance,
            ),
        )

    @partial(jax.jit, static_argnums=(0,))
    def compute_overall_loss(self, params, x, y, t, step):
        ewc_loss = self.calculate_regularization_loss(params)
        loss, logits = self.cross_entropy(params, x, y, t, training=True, step=step)
        loss += self.regularization_coefficient * ewc_loss
        return loss, logits

    @partial(jax.jit, static_argnums=(0,))
    def regularization_training_step(self, state: TrainState, x, y, t, step):
        grad_fn = jax.value_and_grad(self.compute_overall_loss, has_aux=True, allow_int=True)
        (loss, logits), grads = grad_fn(state.params, x, y, t, step=step)
        state = state.apply_gradients(grads=grads)
        return dict(state=state, logits=logits, loss=loss, grads=grads)

    def training_step(self, *args, **kwargs):
        if self.task_counter == 1:
            return super().training_step()
        else:
            self.batch_outputs = self.regularization_training_step(
                self.state, self.x, self.y, self.t, step=self.step_counter
            )
            self.register_batch_outputs(self.batch_outputs)
