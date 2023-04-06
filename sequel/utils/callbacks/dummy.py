from .base_callback import BaseCallback


class DummyCallback(BaseCallback):
    """Dummy callback.

    Mainly used for debugging.
    """

    def __init__(self):
        super().__init__(name="dummy")
        self.printfn = print

    def on_before_setup(self, *args, **kwargs):
        self.printfn("on_before_setup")

    def on_after_setup(self, *args, **kwargs):
        self.printfn("on_after_setup")

    def on_before_teardown(self, *args, **kwargs):
        self.printfn("on_before_teardown")

    def on_after_teardown(self, *args, **kwargs):
        self.printfn("on_after_teardown")

    def on_before_fit(self, *args, **kwargs):
        self.printfn("on_before_fit")

    def on_after_fit(self, *args, **kwargs):
        self.printfn("on_after_fit")

    def on_before_training_task(self, *args, **kwargs):
        self.printfn("on_before_training_task")

    def on_after_training_task(self, *args, **kwargs):
        self.printfn("on_after_training_task")

    def on_before_training_epoch(self, *args, **kwargs):
        self.printfn("on_before_training_epoch")

    def on_after_training_epoch(self, *args, **kwargs):
        self.printfn("on_after_training_epoch")

    def on_before_training_step(self, *args, **kwargs):
        self.printfn("on_before_training_step")

    def on_after_training_step(self, *args, **kwargs):
        self.printfn("on_after_training_step")

    def on_before_backward(self, *args, **kwargs):
        self.printfn("on_before_backward")

    def on_after_backward(self, *args, **kwargs):
        self.printfn("on_after_backward")
