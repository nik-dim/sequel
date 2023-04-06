import logging
from collections import defaultdict
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from sequel.algos.base_algo import BaseAlgorithm
from sequel.backbones.pytorch import BaseBackbone
from sequel.backbones.pytorch.base_backbone import BackboneWrapper
from sequel.backbones.pytorch.base_backbone import BaseBackbone as PytorchBaseBackbone
from sequel.benchmarks.base_benchmark import Benchmark
from sequel.utils.callbacks.base_callback import BaseCallback
from sequel.utils.loggers.base_logger import Logger


class PytorchBaseAlgorithm(BaseAlgorithm):
    optimizer: torch.optim.Optimizer
    backbone: torch.nn.Module

    def __init__(
        self,
        backbone: PytorchBaseBackbone,
        benchmark: Benchmark,
        optimizer: torch.optim.Optimizer,
        callbacks: Iterable[BaseCallback] = [],
        loggers: Optional[Iterable[Logger]] = None,
        lr_decay: Optional[float] = None,
        grad_clip: Optional[float] = None,
        reinit_optimizer: bool = True,
        device="cuda:0",
        min_lr=0.00005,
    ) -> None:
        """Inits the PytorchBaseAlgorithm class.

        Args:
            backbone (PytorchBaseBackbone): The backbone model, e.g., a CNN.
            benchmark (Benchmark): The benchmark, e.g., SplitMNIST.
            optimizer (torch.optim.Optimizer):  The optimizer used to update the backbone weights.
            callbacks (Iterable[BaseCallback], optional): A list of callbacks. At least one instance of MetricCallback
                should be given. Defaults to [].
            loggers (Optional[Logger], optional): A list of logger, e.g. for Weights&Biases logging functionality.
                Defaults to None.
            lr_decay (Optional[float], optional): A learning rate decay used for every new task. Defaults to None.
            grad_clip (Optional[float], optional): The gradient clipping norm. Defaults to None.
            reinit_optimizer (bool): Indicates whether the optimizer state is reinitialized before fitting a new task.
                Defaults to True.
            device (str, optional): _description_. Defaults to "cuda:0".
            min_lr (float, optional): _description_. Defaults to 0.00005.

        Note:
            1. the `_configure_optimizers` method will be moved to a dedicated Callback.
        """
        self.device = device
        if not isinstance(backbone, BaseBackbone):
            backbone = BackboneWrapper(backbone)
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
        self.backbone = self.backbone.to(self.device)
        self.min_lr = min_lr

    def count_parameters(self):
        device = next(self.backbone.parameters()).device
        self.backbone(torch.ones(self.input_dimensions).unsqueeze(0).to(device), torch.ones((1)))
        return sum([p.numel() for p in self.backbone.parameters() if p.requires_grad])

    def _configure_optimizers(self, task):
        if self.task_counter == 1 or self.reinit_optimizer:
            assert len(self.optimizer.param_groups) == 1
            lr = self.optimizer.param_groups[0]["lr"]
            self.optimizer.state = defaultdict(dict)
            self.optimizer.param_groups[0]["params"] = list(self.backbone.parameters())
            if self.lr_decay is not None and task > 1:
                assert isinstance(self.lr_decay, float)
                assert self.lr_decay > 0 and self.lr_decay <= 1, "lr decay should be in the interval (0,1]"
                new_lr = max(lr * self.lr_decay, self.min_lr)
                self.optimizer.param_groups[0]["lr"] = new_lr
                logging.info(f"Decaying the learning rate by a factor of {self.lr_decay}")

            logging.info(self.optimizer)

    def _configure_criterion(self, task_id=None):
        return torch.nn.CrossEntropyLoss()

    def forward(self, *args, **kwargs):
        """Calls the forward function of the model."""
        outs = self.backbone(self.x, self.t)
        self.y_hat = outs
        return outs

    def unpack_batch(self, batch):
        device = self.device
        x, y, t = batch
        self.x, self.y, self.t = x.to(device), y.to(device), t.to(device)
        self.bs = len(x)

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()

    def backpropagate_loss(self):
        self.loss.backward()

    def optimizer_step(self):
        self.optimizer.step()

    def perform_gradient_clipping(self):
        if self.grad_clip is not None:
            assert self.grad_clip > 0
            torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), self.grad_clip)

    def valid_step(self, *args, **kwargs):
        """Performs the validation step. Callbacks are offered for each step of the process."""
        with torch.no_grad():
            y_hat = self.forward()
            self.loss = self.compute_loss(y_hat, self.y, self.t)

    def test_step(self, *args, **kwargs):
        """Performs the testing step. Callbacks are offered for each step of the process."""
        pass

    def prepare_for_next_task(self, task):
        self._configure_optimizers(task)

    def set_training_mode(self):
        self.backbone.train()
        super().set_training_mode()

    def set_evaluation_mode(self):
        self.backbone.eval()
        super().set_evaluation_mode()

    def fit(self, epochs):
        self.backbone = self.backbone.to(self.device)
        return super().fit(epochs=epochs)

    def compute_loss(self, predictions, targets, task_ids=None, *args, **kwargs) -> torch.Tensor:
        return F.cross_entropy(predictions, targets)


class Naive(PytorchBaseAlgorithm):
    pass


class PytorchRegularizationBaseAlgorithm(PytorchBaseAlgorithm):
    def __init__(self, regularization_coefficient, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.regularization_coefficient = regularization_coefficient
        self.w = {}
        for name, param in self.backbone.named_parameters():
            name = name.replace(".", "_")
            # register old parameters and importance weight
            self.backbone.register_buffer(f"{name}_old", torch.zeros_like(param))
            self.backbone.register_buffer(f"{name}_importance", torch.zeros_like(param))

    def calculate_regularization_loss(self):
        """Calculates the regularization loss:

        $$
        \\mathcal{L}_{\\textrm{reg}} = \\sum\\limits_{i} \\Omega_i(\\theta_i-\\theta_{i, \\textrm{old}})^2
        $$

        where $\\Omega_i$ is the importance of parameter $i$, $\\theta_i$ and $\\theta_{i, \\textrm{old}}$ are the current and previous task's parameters.

        The parameter importances $\\Omega_i$ are calculated by the method `calculate_parameter_importance`.

        """
        assert self.task_counter > 1
        # shouldn't be called for the first task
        # because we have not calculate_parameter_importanced anything yet
        losses = []
        for name, param in self.backbone.named_parameters():
            name = name.replace(".", "_")
            old_param = getattr(self.backbone, f"{name}_old")
            importance = getattr(self.backbone, f"{name}_importance")
            losses.append((importance * (param - old_param) ** 2).sum())

        return sum(losses)

    def compute_loss(self, predictions: Tensor, targets: Tensor, task_ids: Tensor, *args, **kwargs) -> Tensor:
        """Computes the loss. For tasks excluding the initial one, the loss also includes the regularization term.

        Args:
            predictions (Tensor): Model predictions.
            targets (Tensor): Targets of the current batch.
            task_ids (Tensor): Task ids of the current batch.

        Returns:
            Tensor: the overall loss.
        """
        loss = super().compute_loss(predictions, targets, task_ids, *args, **kwargs)
        if self.task_counter > 1:
            reg_loss = self.calculate_regularization_loss()
            loss += self.regularization_coefficient * (reg_loss / 2)

        return loss

    def calculate_parameter_importance(self) -> Dict[str, Tensor]:
        r"""Calculcates the per-parameter importance. Should return a dictionary with keys in the format
        "{name}_importance" where name corresponds the `torch.nn.Parameter` the importance is attached to.

        Raises:
            NotImplementedError: Should be implemented according to each algorithm.
        """
        raise NotImplementedError

    def on_after_training_task(self, *args, **kwargs):
        importances = self.calculate_parameter_importance()

        for name, param in self.backbone.named_parameters():
            name = name.replace(".", "_")
            setattr(self.backbone, f"{name}_importance", importances[name].clone())
            setattr(self.backbone, f"{name}_old", param.data.clone())

        return super().on_after_training_task(*args, **kwargs)
