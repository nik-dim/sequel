import torch
from .pytorch_base_algo import PytorchRegularizationBaseAlgorithm


class MAS(PytorchRegularizationBaseAlgorithm):
    """Memory Aware Synapses. Algorithm Class. Inherits from BaseAlgorithm.

    The equivalent JAX implementation is [`MAS in JAX`][sequel.algos.jax.mas.MAS].

    References:
        [1] Aljundi, R., Babiloni, F., Elhoseiny, M., Rohrbach, M. & Tuytelaars, T. Memory Aware Synapses: Learning
            What (not) to Forget. in Computer Vision - ECCV 2018.
    """

    def __init__(self, mas_lambda: float = 1.0, *args, **kwargs):
        """Inits the Memory Aware Synapses algorithm.

        Args:
            mas_lambda (float): The c coefficient of the algorithm.
        """
        super().__init__(regularization_coefficient=mas_lambda, *args, **kwargs)

        torch.autograd.set_detect_anomaly(True)
        for name, param in self.backbone.named_parameters():
            name = name.replace(".", "_")
            self.backbone.register_buffer(f"{name}_w", torch.zeros_like(param))

    def __repr__(self) -> str:
        return f"MAS(mas_lambda={self.regularization_coefficient})"

    def on_after_training_step(self, *args, **kwargs):
        # perform the forward pass once again with the new parameters.
        self.forward()
        self.optimizer_zero_grad()
        f_loss: torch.Tensor = self.y_hat.pow_(2).mean()
        f_loss.backward()
        for name, param in self.backbone.named_parameters():
            name = name.replace(".", "_")
            w = getattr(self.backbone, f"{name}_w")
            if param.grad is not None:
                setattr(self.backbone, f"{name}_w", w + param.grad.abs() / len(self.train_loader))
        return super().on_after_training_step(*args, **kwargs)

    def calculate_parameter_importance(self):
        importances = {}
        for name, param in self.backbone.named_parameters():
            name = name.replace(".", "_")
            importances[name] = getattr(self.backbone, f"{name}_w")

        return importances
