from typing import Optional

from torch.utils.data import DataLoader

from .pytorch_base_algo import PytorchBaseAlgorithm


class JointTraining(PytorchBaseAlgorithm):
    """The JoinTraining algorithm. It is a variant of MultiTask Learning, where the model is trained with increasingly
    more samples. Specifically, during the t-th task, the model sees samples from all the previous and the current
    task.

    Inherits from BaseAlgorithm. Only the `prepare_train_loader` method is overwritten.

    The equivalent JAX implementation is [`JointTraining in JAX`][sequel.algos.jax.joint.JointTraining].

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
