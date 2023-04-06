from typing import TYPE_CHECKING

from sequel.benchmarks.memory import MemoryMechanism
from sequel.utils.callbacks.algo_callback import AlgoCallback

if TYPE_CHECKING:
    from sequel.algos.base_algo import BaseAlgorithm


class MemoryMechanismCallback(AlgoCallback):
    """Wraps an AlgoCallback around the MemoryMechanism for ease of use."""

    def __init__(self, per_task_memory_samples: int, groupby: str = "class"):
        super().__init__()
        self.memory = MemoryMechanism(per_task_memory_samples, groupby)
        self.per_task_memory_samples = per_task_memory_samples
        self.groupby = groupby

    def on_after_training_task(self, algo: "BaseAlgorithm", *args, **kwargs):
        """Updates the memory Mechanism.

        Args:
            algo (BaseAlgorithm): The current BaseAlgorithm instance.
        """
        self.memory.update_memory(algo)
        algo.update_episodic_memory()
