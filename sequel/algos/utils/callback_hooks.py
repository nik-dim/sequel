import abc
from typing import List

from sequel.utils.callbacks.base_callback import BaseCallback


class BaseCallbackHook(abc.ABC):
    callbacks: List[BaseCallback] = []

    def on_before_setup_callbacks(self):
        """Callbacks before the setup."""
        for cb in self.callbacks:
            cb.on_before_setup(self)

    def on_after_setup_callbacks(self):
        """Callbacks after the setup."""
        for cb in self.callbacks:
            cb.on_after_setup(self)

    def on_before_teardown_callbacks(self):
        """Callbacks before the teardown."""
        for cb in self.callbacks:
            cb.on_before_teardown(self)

    def on_after_teardown_callbacks(self):
        """Callbacks after the teardown."""
        for cb in self.callbacks:
            cb.on_after_teardown(self)

    def on_before_fit_callbacks(self):
        """Callbacks before fitting the data."""
        for cb in self.callbacks:
            cb.on_before_fit(self)

    def on_after_fit_callbacks(self):
        """Callbacks after fitting the data."""
        for cb in self.callbacks:
            cb.on_after_fit(self)

    def on_before_training_task_callbacks(self):
        """Callbacks before training a single task."""
        for cb in self.callbacks:
            cb.on_before_training_task(self)

    def on_after_training_task_callbacks(self):
        """Callbacks after training a single task."""
        for cb in self.callbacks:
            cb.on_after_training_task(self)

    def on_before_training_epoch_callbacks(self):
        """Callbacks before training one epoch."""
        for cb in self.callbacks:
            cb.on_before_training_epoch(self)

    def on_after_training_epoch_callbacks(self):
        """Callbacks after training one epoch."""
        for cb in self.callbacks:
            cb.on_after_training_epoch(self)

    def on_before_val_epoch_callbacks(self):
        """Callbacks before validating one epoch."""
        for cb in self.callbacks:
            cb.on_before_val_epoch(self)

    def on_after_val_epoch_callbacks(self):
        """Callbacks after validating one epoch."""
        for cb in self.callbacks:
            cb.on_after_val_epoch(self)

    def on_before_val_step_callbacks(self):
        """Callbacks before the val step (single batch step)."""
        for cb in self.callbacks:
            cb.on_before_val_step(self)

    def on_after_val_step_callbacks(self):
        """Callbacks after the val step."""
        for cb in self.callbacks:
            cb.on_after_val_step(self)

    def on_before_training_step_callbacks(self):
        """Callbacks before the training step (single batch step)."""
        for cb in self.callbacks:
            cb.on_before_training_step(self)

    def on_after_training_step_callbacks(self):
        """Callbacks after the training step."""
        for cb in self.callbacks:
            cb.on_after_training_step(self)

    def on_before_backward_callbacks(self):
        """Callbacks before backpropagation."""
        for cb in self.callbacks:
            cb.on_before_backward(self)

    def on_after_backward_callbacks(self):
        """Callbacks after backpropagation."""
        for cb in self.callbacks:
            cb.on_after_backward(self)

    def on_before_optimizer_step_callbacks(self):
        """Callbacks before optimizer step."""
        for cb in self.callbacks:
            cb.on_before_optimizer_step(self)

    def on_after_optimizer_step_callbacks(self):
        """Callbacks after optimizer step."""
        for cb in self.callbacks:
            cb.on_after_optimizer_step(self)

    def on_before_validating_algorithm_on_all_tasks_callbacks(self):
        for cb in self.callbacks:
            cb.on_before_validating_algorithm_on_all_tasks(self)

    def on_after_validating_algorithm_on_all_tasks_callbacks(self):
        for cb in self.callbacks:
            cb.on_after_validating_algorithm_on_all_tasks(self)
