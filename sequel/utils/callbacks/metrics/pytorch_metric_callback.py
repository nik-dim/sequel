from typing import TYPE_CHECKING, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchmetrics import MeanMetric, Metric, MetricCollection

from .metric_callback import MetricCallback

if TYPE_CHECKING:
    from sequel.algos.base_algo import BaseAlgorithm


class CrossEntropyLossMetric(MeanMetric):
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Wraps CrossEntropy into a torchmetrics MeanMetric.

        Args:
            preds (torch.Tensor): the logits of the current batch.
            target (torch.Tensor): the targets of the current batch.

        Returns:
            torch.Tensor: the computed cross-entropy loss.
        """
        value = F.cross_entropy(input=preds, target=target)
        return super().update(value)


class ForgettingMetric(Metric):
    full_state_update = False

    def __init__(self):
        super().__init__()
        # self.add_state("max_accuracy", default=torch.tensor(-10), dist_reduce_fx="max")
        # self.add_state("current_accuracy", default=torch.tensor(0), dist_reduce_fx="max")
        self.current_accuracy = 0
        self.max_accuracy = -10

    def update(self, accuracy: torch.Tensor):
        assert isinstance(accuracy, torch.Tensor)

        self.current_accuracy = accuracy
        if self.current_accuracy > self.max_accuracy:
            self.max_accuracy = accuracy

        return self.current_accuracy - self.max_accuracy

    def compute(self):
        return self.current_accuracy - self.max_accuracy


class BackwardTranferMetric(ForgettingMetric):
    """How much learning the current experience improves my performance on previous experiences?"""

    def compute(self):
        return -super().compute()


class PytorchMetricCallback(MetricCallback):
    """Base class for the MetricCallback in case of PyTorch.

    Inherits from `MetricCallback`.
    """

    def __init__(self, metrics: MetricCollection, logging_freq=10):
        super().__init__(metrics, logging_freq)

    def on_before_fit(self, algo: "BaseAlgorithm", *args, **kwargs):
        super().on_before_fit(algo, *args, **kwargs)
        self.forgetting = [ForgettingMetric() for i in range(self.num_tasks)]

    def connect(self, algo, *args, **kwargs):
        self.device = algo.device
        super().connect(algo, *args, **kwargs)

    def identify_seen_tasks(self, algo) -> List[int]:
        return torch.unique(algo.t - 1)

    def _reset_metrics(self, prefix):
        self.metrics = nn.ModuleList(
            self.original_metrics.clone(postfix=f"/{self.get_task_id(i)}", prefix=prefix).to(self.device)
            for i in range(self.num_tasks)
        ).to(self.device)
        self.avg_loss = MeanMetric(prefix=prefix).to(self.device)


class StandardMetricCallback(PytorchMetricCallback):
    """Defines the standard Metric Callback used for classificaiton."""

    def __init__(self, logging_freq=1):
        metrics = torchmetrics.MetricCollection(
            {
                "acc": torchmetrics.Accuracy(),
                "loss": CrossEntropyLossMetric(),
            },
        )
        super().__init__(metrics, logging_freq)
