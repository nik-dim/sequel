import os

import numpy as np
import torch

from sequel.utils.utils import get_experiment_root_dir

from .base_logger import Logger


class MetricHistory:
    def __init__(self):
        self.data = {}

    def update(self, metrics: dict):
        for k, v in metrics.items():
            if len(k.split("/")) == 3:
                mode, metric, task_id = k.split("/")
                if mode == "val":
                    if metric not in self.data.keys():
                        self.data[metric]: dict = {}
                    if task_id not in self.data[metric].keys():
                        self.data[metric][task_id]: list = []
                    if isinstance(v, torch.Tensor):
                        v = v.cpu().item()
                    self.data[metric][task_id].append(v)
            else:
                continue
                raise ValueError(
                    f"The metrics should be given in the following format MODE/METRIC/TAKS-NAME. \
                    For example, MODE=val, METRIC=acc and TASK-NAME=task-0. The function received the following key {k}."
                )

    def get_full_history(self, metric: str) -> np.array:
        data = self.data[metric].values()
        pad = max(map(len, data))
        return np.array([[None] * (pad - len(x)) + x for x in data])

    def save_metrics(self, root_dir):
        for metric in self.data.keys():
            filepath = os.path.join(root_dir, metric + ".npy")
            with open(filepath, "wb") as f:
                np.save(f, self.get_full_history(metric=metric))


class LocalLogger(Logger):
    """Handles the local logging; the metrics are saved in local .npy files. Inherits from Logger.

    Note:
        The LocalLogger does not handle hyperparameter/config file saving.
    """

    def __init__(self):
        """Inits the LocalLogger."""
        super().__init__()
        self.metric_history = MetricHistory()

    def log(self, item, step=None, epoch=None):
        self.metric_history.update(item)

    def log_figure(self, figure, name, step=None, epoch=None):
        pass

    def log_all_results(self):
        """Saves the results in the .npy format.

        This method should be called after fitting.
        """
        print(f"Saving in {get_experiment_root_dir()}")
        self.metric_history.save_metrics(root_dir=get_experiment_root_dir())
