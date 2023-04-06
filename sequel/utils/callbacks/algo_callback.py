from typing import TYPE_CHECKING

from .base_callback import BaseCallback

if TYPE_CHECKING:
    from sequel.algos.base_algo import BaseAlgorithm


class AlgoCallback(BaseCallback):
    """Base class for algorithm continual callback.

    The main purpose of this class is to enhance readability and type checking by adding the `BaseAlgorithm` as an
    argument.
    """

    def __init__(self):
        pass

    def connect(self, algo: "BaseAlgorithm", *args, **kwargs):
        pass

    def on_before_setup(self, algo: "BaseAlgorithm", *args, **kwargs):
        pass

    def on_after_setup(self, algo: "BaseAlgorithm", *args, **kwargs):
        pass

    def on_before_teardown(self, algo: "BaseAlgorithm", *args, **kwargs):
        pass

    def on_after_teardown(self, algo: "BaseAlgorithm", *args, **kwargs):
        pass

    def on_before_fit(self, algo: "BaseAlgorithm", *args, **kwargs):
        pass

    def on_after_fit(self, algo: "BaseAlgorithm", *args, **kwargs):
        pass

    def on_before_training_epoch(self, algo: "BaseAlgorithm", *args, **kwargs):
        pass

    def on_after_training_epoch(self, algo: "BaseAlgorithm", *args, **kwargs):
        pass

    def on_before_val_epoch(self, algo: "BaseAlgorithm", *args, **kwargs):
        pass

    def on_after_val_epoch(self, algo: "BaseAlgorithm", *args, **kwargs):
        pass

    def on_before_testing_epoch(self, algo: "BaseAlgorithm", *args, **kwargs):
        pass

    def on_after_testing_epoch(self, algo: "BaseAlgorithm", *args, **kwargs):
        pass

    def on_before_training_step(self, algo: "BaseAlgorithm", *args, **kwargs):
        pass

    def on_after_training_step(self, algo: "BaseAlgorithm", *args, **kwargs):
        pass

    def on_before_backward(self, algo: "BaseAlgorithm", *args, **kwargs):
        pass

    def on_after_backward(self, algo: "BaseAlgorithm", *args, **kwargs):
        pass

    def on_before_forward(self, algo: "BaseAlgorithm", *args, **kwargs):
        pass

    def on_after_forward(self, algo: "BaseAlgorithm", *args, **kwargs):
        pass

    def on_before_optimizer_step(self, algo: "BaseAlgorithm", *args, **kwargs):
        pass

    def on_after_optimizer_step(self, algo: "BaseAlgorithm", *args, **kwargs):
        pass

    def on_before_val_step(self, algo: "BaseAlgorithm", *args, **kwargs):
        pass

    def on_after_val_step(self, algo: "BaseAlgorithm", *args, **kwargs):
        pass

    def on_before_testing_step(self, algo: "BaseAlgorithm", *args, **kwargs):
        pass

    def on_after_testing_step(self, algo: "BaseAlgorithm", *args, **kwargs):
        pass

    def on_before_validating_algorithm_on_all_tasks(self, algo: "BaseAlgorithm", *args, **kwargs):
        pass

    def on_after_validating_algorithm_on_all_tasks(self, algo: "BaseAlgorithm", *args, **kwargs):
        pass

    def on_after_training_task(self, algo: "BaseAlgorithm", *args, **kwargs):
        pass

    def on_before_training_task(self, algo: "BaseAlgorithm", *args, **kwargs):
        pass
