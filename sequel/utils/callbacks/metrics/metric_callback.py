import copy
import logging
from collections import ChainMap
from typing import TYPE_CHECKING, List, Union

import torchmetrics
from beautifultable import BeautifulTable

from sequel.utils.callbacks.algo_callback import AlgoCallback
from sequel.utils.utils import safe_conversion

from .jax_metrics import Metric as JaxMetric

if TYPE_CHECKING:
    from sequel.algos.base_algo import BaseAlgorithm


class MetricCallback(AlgoCallback):
    """MetricCallback is the parent clas for the PyTorch and Jax metric callbacks. Handles the computation of metrics
    during training, validation etc."""

    forgetting: List[Union[torchmetrics.Metric, JaxMetric]]

    def __init__(self, metrics, logging_freq=10):
        super().__init__()
        self.logging_freq = logging_freq
        self.original_metrics = metrics.clone()

    def on_before_fit(self, algo: "BaseAlgorithm", *args, **kwargs):
        self.num_tasks = algo.num_tasks
        return super().on_before_fit(algo, *args, **kwargs)

    def _reset_metrics(self, prefix):
        raise NotImplementedError

    def log(self, algo: "BaseAlgorithm", key, value):
        algo.log({key: value})

    def get_task_id(self, i):
        return f"task-{i}"

    def register_metric_callback_message(self, msg: dict, algo: "BaseAlgorithm"):
        # some weird bug due to wandb. it adds _timestamp and _runtime to the msg dict
        msg = {k: v for k, v in msg.items() if not k.startswith("_")}
        msg = {k.split("/")[1]: safe_conversion(v) for k, v in msg.items()}
        setattr(algo, "metric_callback_msg", msg)
        return msg

    def identify_seen_tasks(self, algo: "BaseAlgorithm") -> List[int]:
        raise NotImplementedError

    def compute_mask(self, algo, task_id):
        return (algo.t - 1) == task_id

    # ------- STEPS -------
    def on_after_training_step(self, algo: "BaseAlgorithm"):
        self.avg_loss(algo.loss)
        tasks_seen = self.identify_seen_tasks(algo)
        for task_id in tasks_seen:
            mask = self.compute_mask(algo, task_id)
            self.metrics[task_id](algo.y_hat[mask], algo.y[mask])

        if (algo.batch_idx + 1) % self.logging_freq == 0:
            msg: dict = self.metrics[algo.task_counter - 1].compute()
            msg.update({"train/avg_loss": self.avg_loss.compute()})
            algo.log(copy.deepcopy(msg))
            msg = self.register_metric_callback_message(msg, algo)

    def on_after_val_step(self, algo: "BaseAlgorithm"):
        self.avg_loss(algo.loss)
        tasks_seen = self.identify_seen_tasks(algo)
        for task_id in tasks_seen:
            mask = self.compute_mask(algo, task_id)
            self.metrics[task_id](algo.y_hat[mask], algo.y[mask])

        if (algo.batch_idx + 1) % self.logging_freq == 0:
            msg: dict = self.metrics[algo.current_val_task - 1].compute()
            msg.update({"val/avg_loss": self.avg_loss.compute()})
            msg = self.register_metric_callback_message(msg, algo)

    # ------- EPOCHS - before -------
    def on_before_training_epoch(self, *args, **kwargs):
        self._reset_metrics(prefix="train/")

    def on_before_val_epoch(self, *args, **kwargs):
        self._reset_metrics(prefix="val/")

    def on_before_testing_epoch(self, *args, **kwargs):
        self._reset_metrics(prefix="test/")

    def on_before_validating_algorithm_on_all_tasks(self, algo: "BaseAlgorithm", *args, **kwargs):
        self.metric_results = []

    # ------- EPOCHS - after -------
    def on_after_val_epoch(self, algo: "BaseAlgorithm", *args, **kwargs):
        res = self.metrics[algo.current_val_task - 1].compute()
        self.task_counter_metrics = res
        self.metric_results.append(res)

        to_log = copy.deepcopy(res)
        to_log["epoch"] = algo.epoch_counter
        algo.log(to_log)

        key = list(filter(lambda x: "acc" in x, list(self.task_counter_metrics.keys())))[0]
        self.forgetting[algo.current_val_task - 1](self.task_counter_metrics[key])

    def on_after_validating_algorithm_on_all_tasks(self, algo: "BaseAlgorithm", *args, **kwargs):
        forgetting = {
            f"val/forgetting/task-{i+1}": safe_conversion(k.compute())
            for i, k in enumerate(self.forgetting)
            if i < algo.task_counter
        }
        algo.log(forgetting)

        # compute averages
        avg = {}
        for key in self.original_metrics:
            temp = [[v for k, v in m.items() if key in k][0] for m in self.metric_results]
            avg[key] = safe_conversion(sum(temp)) / len(temp)

        avg = {f"avg/{k}": v for k, v in avg.items()}
        avg["avg/forgetting"] = sum(forgetting.values()) / len(forgetting)
        algo.log(avg)

        # only print table at the end of fitting one task
        if algo.epoch_counter % algo.epochs == 0:
            self.print_task_metrics(self.metric_results, epoch=algo.epoch_counter)
        else:
            logging.info({k: round(v, 3) for k, v in avg.items()})
            logging.info({k: round(safe_conversion(v), 3) for k, v in self.metric_results[-1].items()})

        _metrics = dict(ChainMap(*self.metric_results))
        _metrics = {k: round(safe_conversion(v), 3) for k, v in _metrics.items()}
        self.register_results_to_algo(algo, "val_metrics", _metrics)

    def on_after_fit(self, algo: "BaseAlgorithm", *args, **kwargs):
        if algo.loggers is not None:
            for logger in algo.loggers:
                logger.log_all_results()

    def register_results_to_algo(self, algo, results_name, results_dict):
        setattr(algo, results_name, results_dict)

    def print_task_metrics(self, metrics: list, epoch):
        table = BeautifulTable(default_alignment=BeautifulTable.ALIGN_LEFT, default_padding=1, maxwidth=250)
        table.set_style(BeautifulTable.STYLE_BOX_ROUNDED)
        keys = list(metrics[0].keys())
        keys = [k.split("/")[1] for k in keys]
        table.rows.header = keys
        for i, m in enumerate(metrics):
            column_name = f"Task-{i+1}"
            table.columns.append(m.values(), header=column_name)
            table.columns.alignment[column_name] = BeautifulTable.ALIGN_RIGHT

        avg = {}
        for key in keys:
            temp = [[v for k, v in m.items() if key in k][0] for m in metrics]
            avg[key] = sum(temp) / len(temp)

        table.columns.append(avg.values(), header="AVG")
        table.columns.alignment["AVG"] = BeautifulTable.ALIGN_RIGHT
        f = [safe_conversion(k.compute()) for i, k in enumerate(self.forgetting) if i < len(metrics)]
        f.append(sum(f) / len(f))
        table.rows.append(f, header="Forgetting")
        logging.info(f"EVAL METRICS for epoch {epoch}:\n{table}")
