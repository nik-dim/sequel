from typing import Dict, Optional

from torch.utils.data import DataLoader

from .jax_base_algo import JaxBaseAlgorithm


class JointTraining(JaxBaseAlgorithm):
    """The JoinTraining algorithm. It is a variant of MultiTask Learning, where the model is trained with increasingly
    more samples. Specifically, during the t-th task, the model sees samples from all the previous and the current
    task.

    Inherits from BaseAlgorithm. Only the `prepare_train_loader` method is overwritten.

    The equivalent PyTorch implementation is [`JointTraining in Pytorch`][sequel.algos.pytorch.joint.JointTraining].
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        return f"JointTraining()"

    def prepare_train_loader(self, task_id: int, batch_size: Optional[int] = None) -> DataLoader:
        """Prepares the train_loader for Joint Training. The dataloader consists of all samples up to task `task_id`.

        Args:
            task_id (int): The last task to be loaded.
            batch_size (Optional[int], optional): The dataloader batch size. Defaults to None.

        Returns:
            DataLoader: The JointTraining train dataloder.
        """
        return self.benchmark.train_dataloader_joint(task_id, batch_size=batch_size)
