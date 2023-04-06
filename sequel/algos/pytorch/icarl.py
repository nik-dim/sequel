from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from .pytorch_base_algo import PytorchBaseAlgorithm


class FeatureExtractor(nn.Module):
    """Wrapper that returns a flattened version of the output of specific PyTorch model. It is used to retrieve the
    feature representations (e.g. for iCaRL algorithm).
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, x: Tensor, *agrs, **kwargs) -> Tensor:
        bs = x.size(0)
        x = self.model(x)
        return x.view(bs, -1)


class Icarl(PytorchBaseAlgorithm):
    """iCaRL: Incremental Classifier and Representation Learning algorithm. Inherits from BaseAlgorithm."""

    def __init__(self, memory_size: int, *args, **kwargs):
        """Inits the iCaRL algorithm.

        Args:
            memory_size (int): The overall memory size used by the algorithm.
        """
        super().__init__(*args, **kwargs)
        self.memory_size = memory_size

    @classmethod
    def from_config(cls, config, callbacks, loggers, *args, **kwargs):
        memory_size = config.algo.memory_size
        return cls(
            memory_size=memory_size,
            config=config,
            callbacks=callbacks,
            loggers=loggers,
            *args,
            **kwargs,
        )

    def prepare_train_loader(self, task_id: int, batch_size: Optional[int] = None) -> DataLoader:
        """Prepares the train_loader. After the initical task, the train dataloader is augmented with the memory
        samples.

        Args:
            task_id (int): The id of the task to be loaded.
            batch_size (Optional[int], optional): The batch size for the dataloader. If set to None,
                the default batch size (for the current experiment) is used. Defaults to None.

        Returns:
            DataLoader: The train dataloader for the current epoch.
        """

        if task_id == 1:
            return super().prepare_train_loader(task_id, batch_size)
        else:
            return self.benchmark.train_dataloader_with_memory(task_id, batch_size=batch_size, verbose=True)

    def on_after_training_task(self, *args, **kwargs):
        """Handles memory specifics for the iCaRL algorithm, such as memory resizing and selecting new memory indices
        using the Herding algorithm.

        Raises:
            ValueError: The current task must not have memory yet, since this methods sets it.
        """
        k = self.memory_size // self.task_counter
        for task in range(1, self.task_counter):
            old_indices = self.benchmark.get_memory_indices(task)
            new_indices = self.resize_memory(old_indices, k)
            self.benchmark.set_memory_indices(task, new_indices)

        # compute current task indices. The excess of exemplars goes to the last task for simplicity.
        num_indices = k + self.memory_size % self.task_counter
        indices = self.compute_new_indices(num_indices=num_indices)

        if self.task_counter in self.benchmark.memory_indices.keys():
            raise ValueError("Overwriting memory...Is something wrong?")

        self.benchmark.set_memory_indices(self.task_counter, indices)

    @torch.no_grad()
    def compute_new_indices(self, num_indices: int) -> List[int]:
        """Selects the indices for the current task, using the herding algorithm.

        Args:
            num_indices (int): The number of indices to be selected.

        Returns:
            List[int]: The selected indices.
        """
        dataloader = super().prepare_data_loader(self.task_counter)
        self.backbone.eval()
        model = FeatureExtractor(self.backbone.encoder)
        model.eval()
        features = torch.cat([model(batch[0].to(self.device)) for batch in dataloader])
        indices = self.select_indices_with_herding(features, num_indices)
        return indices

    def select_indices_with_herding(self, features: torch.Tensor, num_indices: int) -> List[int]:
        """Implements the herding algorithm. The most representative `num_indices` samples are selected based on their
        L2 distance to the feature mean.

        Args:
            features (torch.Tensor): The features of all samples.
            num_indices (int): The number of samples to be selected.

        Raises:
            ValueError: The features must be given in a 2D tensor.

        Returns:
            List[int]: The indices of the selected samples.
        """
        if features.dim() != 2:
            raise ValueError(
                "The features must be a Tensor of two dimensions, where the first dimension \
                corresponds to the number of samples."
            )

        selected_indices = []

        center = features.mean(dim=0)
        current_center = center * 0

        for i in range(num_indices):
            # Compute distances with real center
            candidate_centers = current_center * i / (i + 1) + features / (i + 1)
            distances = pow(candidate_centers - center, 2).sum(dim=1)
            distances[selected_indices] = torch.inf

            # Select best candidate
            new_index = distances.argmin().tolist()
            selected_indices.append(new_index)
            current_center = candidate_centers[new_index]

        return selected_indices

    @staticmethod
    def resize_memory(indices: List[int], k: int) -> list[int]:
        """Resizes memory by selecting the first `k` indices.

        Args:
            indices (List[int]): The current memory indices.
            k (int): the new size of the memory.

        Raises:
            ValueError: The new memory size `k` cannot be larger than the previous one.

        Returns:
            list[int]: the new memory indices.
        """
        if k > len(indices):
            raise ValueError("The new memory cannot be larger than the current.")

        return indices[:k]
