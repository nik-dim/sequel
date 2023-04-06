from typing import TYPE_CHECKING, List

import jax.numpy as jnp
import numpy as np


from .jax_metrics import AccuracyMetric, CrossEntropyLossMetric, ForgettingMetric, MeanMetric, MetricCollection
from .metric_callback import MetricCallback

if TYPE_CHECKING:
    from sequel.algos.jax.jax_base_algo import JaxBaseAlgorithm


class JaxMetricCallback(MetricCallback):
    """Handles the computation and logging of metrics.

    Callback hooks after train/val/test steps/epochs etc. Inherits from Callback.
    """

    def __init__(self, metrics, logging_freq=10):
        super().__init__(metrics, logging_freq)

    def on_before_fit(self, algo: "JaxBaseAlgorithm", *args, **kwargs):
        super().on_before_fit(algo, *args, **kwargs)
        self.forgetting = [ForgettingMetric() for i in range(self.num_tasks)]

    def identify_seen_tasks(self, algo) -> List[int]:
        return jnp.unique(algo.t - 1).tolist()

    def _reset_metrics(self, prefix):
        self.metrics = [
            self.original_metrics.clone(postfix=f"/{self.get_task_id(i)}", prefix=prefix)
            for i in range(self.num_tasks)
        ]
        self.avg_loss = MeanMetric()

    def compute_mask(self, algo, task_id):
        mask = super().compute_mask(algo, task_id)
        return np.array(mask)

    def on_after_val_step(self, algo):
        self.avg_loss(algo.loss)
        tasks_seen = self.identify_seen_tasks(algo)
        assert len(tasks_seen) == 1
        task_id = tasks_seen[0]
        self.metrics[task_id](algo.y_hat, algo.y)

        if (algo.batch_idx + 1) % self.logging_freq == 0:
            msg: dict = self.metrics[algo.current_val_task - 1].compute()
            msg.update({"val/avg_loss": self.avg_loss.compute()})
            msg = self.register_metric_callback_message(msg, algo)


class StandardMetricCallback(JaxMetricCallback):
    def __init__(self, logging_freq=1):
        metrics = MetricCollection({"acc": AccuracyMetric(), "loss": CrossEntropyLossMetric()})
        super().__init__(metrics, logging_freq)
