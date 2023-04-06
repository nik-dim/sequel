import logging
from typing import Optional

import torch
import torch.nn as nn


class BaseBackbone(nn.Module):
    """The PyTorch base class for neural networks.

    Inherits from torch.nn.Module and the BaseCallback class that endows callbacks for each stage of training, e.g.,
    before and after trainining/validation steps/epochs/tasks etc.
    """

    def __init__(self, multihead: bool = False, classes_per_task: Optional[int] = None, masking_value: float = -10e10):
        """Inits the BaseBackbone class. This class defines the PyTorch base class for neural networks. All models
        should inherit from this class. Inherits from torch.nn.Module and the BaseCallback class that endows callbacks
        for each stage of training, e.g., before and after trainining/validation steps/epochs/tasks etc.

        Args:
            multihead (bool, optional): Set to True if the backbone is multi-headed. Defaults to False.
            classes_per_task (Optional[int], optional): The number of classes per task. Defaults to None.
            masking_value (float, optional): The value that replaces the logits. Only used if multihead is set to True. Defaults to -10e10.

        Note:
            Currently, the BaseBackbone only considers tasks with equal number of classes.
        """
        super().__init__()
        self.multihead = multihead
        logging.info(f"multihead is set to {self.multihead}")
        if self.multihead:
            assert classes_per_task is not None
        self.classes_per_task = classes_per_task
        self.masking_value = masking_value

    def select_output_head(self, x: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        """Utility function in case `multihead=True` that replaces the original logits by a low value so that almost
        zero probability is given to the corresponding classes.

        Args:
            x (torch.Tensor): The original logits.
            task_ids (torch.Tensor): The task id for each sample in the batch.

        Returns:
            torch.Tensor: the manipulated logits.
        """
        assert self.multihead
        assert isinstance(x, torch.Tensor)
        for i, task_id in enumerate(task_ids):
            task_id = task_id - 1
            if isinstance(task_id, torch.Tensor):
                task_id = task_id.cpu().int().item()
            start = task_id * self.classes_per_task
            end = (task_id + 1) * self.classes_per_task
            x[i, :start].data.fill_(self.masking_value)
            x[i, end:].data.fill_(self.masking_value)
        return x

    def forward(self, x: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        """Implements the forward function of the backbone. Models must ovveride this method.

        Example:
            # perform the forward.
            x = ...
            # select the correct output head.
            if self.multihead:
                return self.select_output_head(x, task_ids)

        Args:
            x (torch.Tensor): The batch inputs.
            task_ids (torch.Tensor): The batch task ids.

        Returns:
            torch.Tensor: The batch predicitons.
        """
        raise NotImplementedError


class BackboneWrapper(BaseBackbone):
    def __init__(
        self,
        model: nn.Module,
        multihead: bool = False,
        classes_per_task: Optional[int] = None,
        masking_value: float = -10e10,
    ):
        super().__init__(multihead, classes_per_task, masking_value)
        self.model = model

    def forward(self, x: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        if self.multihead:
            x = self.select_output_head(x, task_ids)
        return x
