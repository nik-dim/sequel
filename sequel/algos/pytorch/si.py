import logging

import torch
from torch import Tensor

from .pytorch_base_algo import PytorchRegularizationBaseAlgorithm


class SI(PytorchRegularizationBaseAlgorithm):
    """Synaptic Intelligence Algorithm Class. Inherits from PytorchBaseAlgorithm.

    The equivalent JAX implementation is [`SI in JAX`][sequel.algos.jax.si.SI].

    References:
        [1] Zenke, F., Poole, B. & Ganguli, S. Continual Learning Through Synaptic Intelligence. in Proceedings of the
            34th International Conference on Machine Learning, ICML 2017.
    """

    def __init__(self, si_lambda: float = 1.0, xi: float = 0.1, *args, **kwargs):
        super().__init__(regularization_coefficient=si_lambda, *args, **kwargs)
        # hyperparameters
        self.xi = xi

        for name, param in self.backbone.named_parameters():
            name = name.replace(".", "_")
            self.backbone.register_buffer(f"{name}_w", torch.zeros_like(param))

    def __repr__(self) -> str:
        return f"SI(si_lambda={self.regularization_coefficient}, xi={self.xi})"

    def on_before_training_step(self, *args, **kwargs):
        for name, param in self.backbone.named_parameters():
            name = name.replace(".", "_")
            setattr(self.backbone, f"{name}_prev", param.data.clone())

    def on_after_training_step(self, *args, **kwargs):
        for name, param in self.backbone.named_parameters():
            name = name.replace(".", "_")
            if param.grad is not None:
                delta = param.clone().detach() - getattr(self.backbone, f"{name}_prev")
                w = getattr(self.backbone, f"{name}_w")
                setattr(self.backbone, f"{name}_w", w - w * delta)

    def calculate_parameter_importance(self):
        logging.info("Updating importance parameters for Synaptic Intelligence")
        importances = {}
        for (name, p) in self.backbone.named_parameters():
            name = name.replace(".", "_")
            old_importance = getattr(self.backbone, f"{name}_importance")
            omega: Tensor = getattr(self.backbone, f"{name}_w")
            delta: Tensor = p.detach() - getattr(self.backbone, f"{name}_old")

            # see Eq. 5 from paper.
            importances[name] = old_importance + omega / (delta.pow(2) + self.xi)

            # reset (small) omega for next task
            setattr(self.backbone, f"{name}_w", omega.clone().zero_())

        return importances
