from .pytorch_base_algo import PytorchRegularizationBaseAlgorithm


class EWC(PytorchRegularizationBaseAlgorithm):
    """Elastic Weight Consolidation Algorithm Class. Inherits from BaseAlgorithm.

    The equivalent JAX implementation is [`EWC in JAX`][sequel.algos.jax.ewc.EWC].

    References:
        [1] Kirkpatrick, J. et al. Overcoming catastrophic forgetting in neural networks. PNAS 114, 3521-3526 (2017).
    """

    def __init__(self, ewc_lambda: float = 1.0, *args, **kwargs):
        """Inits the Elastic Weight Consolidation algorithm.

        Args:
            ewc_lambda (float): The lambda coefficient of EWC algorithm.
        """
        super().__init__(regularization_coefficient=ewc_lambda, *args, **kwargs)

    def __repr__(self) -> str:
        return f"EWC(ewc_lambda={self.regularization_coefficient})"

    def calculate_parameter_importance(self):
        train_loader = self.benchmark.train_dataloader(self.task_counter)
        self.backbone = self.backbone.to(self.device)

        importances = {}
        for ii, batch in enumerate(train_loader):
            self.unpack_batch(batch)
            outs = self.backbone(self.x, self.t)
            loss = super().compute_loss(outs, self.y, self.t)
            loss.backward()

            for (name, p) in self.backbone.named_parameters():
                name = name.replace(".", "_")
                if p.grad is not None:
                    if getattr(importances, name, None) is None:
                        importances[name] = p.grad.data.clone().pow(2) / len(train_loader)
                    else:
                        importances[name] += p.grad.data.clone().pow(2) / len(train_loader)

        return importances
