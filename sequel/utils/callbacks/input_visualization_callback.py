from typing import TYPE_CHECKING, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from .algo_callback import AlgoCallback

if TYPE_CHECKING:
    from sequel.algos.base_algo import BaseAlgorithm


class InputVisualizationCallback(AlgoCallback):
    """Visualizes random samples from each task and uses the loggers to save the plots."""

    def __init__(self, samples_per_task=5):
        """Inits the InputVisualizationCallback.

        Args:
            samples_per_task (int, optional): number of samples to be saved for each tasks. Defaults to 5.
        """
        super().__init__()
        self.samples_per_task = samples_per_task

    def select_random_samples(self, dataset: torch.utils.data.Dataset) -> List[torch.Tensor]:
        """Selects a prefefined number of samples per each CL dataset. Each task corresponds to a different dataset.

        Args:
            dataset (torch.data.utils.Dataset): The PyTorch Datatet.

        Returns:
            List[torch.Tensor]: The Tensors corresponding to the selected input samples.
        """
        indices = np.random.choice(len(dataset), self.samples_per_task, replace=False)
        samples = [dataset[i] for i in indices]
        return samples

    def on_before_fit(self, algo: "BaseAlgorithm", *args, **kwargs) -> None:
        """Retrieves and diplays in a single plot the input images from all tasks of the benchmark that the algorithm
        has been initialized with. The final plot is saved via the loggers.

        Args:
            algo (BaseAlgorithm): The BaseAlgorithm instance.
        """
        datasets = algo.benchmark.trains
        num_tasks = algo.num_tasks

        samples = []
        for dataset in datasets.values():
            task_samples = self.select_random_samples(dataset)
            samples.append(task_samples)

        s = 2
        figure, axes = plt.subplots(
            nrows=num_tasks,
            ncols=self.samples_per_task,
            figsize=(s * self.samples_per_task, s * num_tasks),
        )

        for i, task_samples in enumerate(samples):
            for j, (x, y, t) in enumerate(task_samples):
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                axes[i][j].imshow(x.permute(1, 2, 0))
                axes[i][j].title.set_text(f"t={t}: y={y}")

        plt.setp(axes, xticks=[], yticks=[])
        figure.subplots_adjust(wspace=0.5)

        # save the plot via the algorithm loggers
        algo.log_figure(name="input/viz", figure=figure)
