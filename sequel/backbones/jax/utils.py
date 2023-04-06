import warnings
from typing import Any, Optional

import omegaconf
import optax
from omegaconf import OmegaConf
from optax._src import combine

from .base_backbone import BaseBackbone
from .cnn import CNN
from .mlp import MLP
from .resnet import ResNet18Thin


def features(apply_fn, params, *args, **kwargs):
    # Assumes that the backbone has the architecture of an encoder succeeded by a classifier (usually a Linear layer)
    return apply_fn(params, *args, **kwargs, method=lambda module, *args, **kwargs: module.encoder(*args, **kwargs))


def without(d, key):
    new_d = d.copy()
    new_d.pop(key)
    return new_d


def select_backbone(config: omegaconf.ListConfig, *args, **kwargs) -> BaseBackbone:
    model_type = config.backbone.type.lower()
    model_kwargs = without(dict(config.backbone), "type")
    if model_type == "mlp":
        return MLP(**model_kwargs)
    elif model_type == "cnn":
        return CNN(**model_kwargs)
    elif model_type == "resnet18thin":
        return ResNet18Thin(**model_kwargs)
    else:
        raise NotImplementedError("Only MLP, CNN and ResNet18thin backbones are supported.")


def sgdw(
    learning_rate: optax.ScalarOrSchedule,
    momentum: Optional[float] = None,
    nesterov: bool = False,
    accumulator_dtype: Optional[Any] = None,
    weight_decay: float = 0.0,
) -> optax.GradientTransformation:
    """A canonical Stochastic Gradient Descent optimizer with weight decay. Adapted from optax sourc code.

    Returns:
      A `GradientTransformation`.
    """
    warnings.warn("SGD with weight decay still an experimental feature in JAX.")

    return combine.chain(
        (
            optax.add_decayed_weights(weight_decay)
            if weight_decay is not None and weight_decay > 0
            else optax.identity()
        ),
        optax.sgd(learning_rate, momentum, nesterov, accumulator_dtype),
    )


def select_optimizer(config: OmegaConf, *args, **kwargs):
    assert "optimizer" in config
    cfg = config.optimizer
    weight_decay = getattr(cfg, "weight_decay", 0)
    if cfg.type.lower() == "sgd":
        momentum = getattr(cfg, "momentum", None)
        if weight_decay > 0:
            # assert weight_decay == 0, "Weight decay in conjuction with lr_decay not yet supported for JAX."
            # The problem is that, at the moment, lr_decay is handled by exposing the optimizer hyperparmeters

            return optax.inject_hyperparams(sgdw)(learning_rate=cfg.lr, momentum=momentum, weight_decay=weight_decay)

        else:
            return optax.inject_hyperparams(optax.sgd)(learning_rate=cfg.lr, momentum=momentum)

    if cfg.type.lower() == "adam":
        return optax.inject_hyperparams(optax.adamw)(learning_rate=cfg.lr, weight_decay=weight_decay)

    raise NotImplementedError("Only SGD and Adam are currently supported.")
