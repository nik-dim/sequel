import copy

import omegaconf
import torch

from . import model_factory
from .base_backbone import BackboneWrapper, BaseBackbone
from .cnn import CNN
from .mlp import MLP
from .resnet import ResNet18Thin


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
    elif model_type in model_factory.__all__:
        return model_factory[model_type](**model_kwargs)
    else:
        backbones = ["mlp", "cnn", "resnet18thin"]
        raise NotImplementedError(
            f"The backbone '{model_type}' is not implemented. The supported backbone templates are {backbones}. "
            f"The custom-made backbones (look at sequel.backbones.pytorch.model_factory) are {model_factory.__all__}."
        )


def select_optimizer(config: omegaconf.OmegaConf, model: torch.nn.Module):
    assert "optimizer" in config, "The Hydra config should include an optimizer."
    cfg = copy.deepcopy(config.optimizer)

    weight_decay = getattr(cfg, "weight_decay", 0)
    if cfg.type.lower() == "sgd":
        momentum = getattr(cfg, "momentum", 0)
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=momentum, weight_decay=weight_decay)
    if cfg.type.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=weight_decay)
    raise NotImplementedError("Unknown optimizer. Only SGD and Adam are currently supported.")
