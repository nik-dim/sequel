class BaseCallback:
    """Base class for callbacks.

    Defines methods for all the various callback points in the trainer.
    """

    def __init__(self):
        pass

    def connect(self, *args, **kwargs):
        pass

    def on_before_setup(self, *args, **kwargs):
        pass

    def on_after_setup(self, *args, **kwargs):
        pass

    def on_before_teardown(self, *args, **kwargs):
        pass

    def on_after_teardown(self, *args, **kwargs):
        pass

    def on_before_fit(self, *args, **kwargs):
        pass

    def on_after_fit(self, *args, **kwargs):
        pass

    def on_before_training_epoch(self, *args, **kwargs):
        pass

    def on_after_training_epoch(self, *args, **kwargs):
        pass

    def on_before_val_epoch(self, *args, **kwargs):
        pass

    def on_after_val_epoch(self, *args, **kwargs):
        pass

    def on_before_testing_epoch(self, *args, **kwargs):
        pass

    def on_after_testing_epoch(self, *args, **kwargs):
        pass

    def on_before_training_step(self, *args, **kwargs):
        pass

    def on_after_training_step(self, *args, **kwargs):
        pass

    def on_before_backward(self, *args, **kwargs):
        pass

    def on_after_backward(self, *args, **kwargs):
        pass

    def on_before_forward(self, *args, **kwargs):
        pass

    def on_after_forward(self, *args, **kwargs):
        pass

    def on_before_optimizer_step(self, *args, **kwargs):
        pass

    def on_after_optimizer_step(self, *args, **kwargs):
        pass

    def on_before_val_step(self, *args, **kwargs):
        pass

    def on_after_val_step(self, *args, **kwargs):
        pass

    def on_before_testing_step(self, *args, **kwargs):
        pass

    def on_after_testing_step(self, *args, **kwargs):
        pass

    def on_before_training_task(self, *args, **kwargs):
        pass

    def on_after_training_task(self, *args, **kwargs):
        pass

    def on_before_validating_algorithm_on_all_tasks(self, *args, **kwargs):
        pass

    def on_after_validating_algorithm_on_all_tasks(self, *args, **kwargs):
        pass
