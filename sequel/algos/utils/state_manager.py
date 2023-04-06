import abc
import logging


class BaseStateManager(abc.ABC):
    step_counter: int = 0
    epoch_counter: int = 0
    task_counter: int = 0

    is_training: bool

    def set_training_mode(self):
        self.is_training = True

    def set_evaluation_mode(self):
        self.is_training = False

    def _increment_step(self):
        self.step_counter += 1

    def _increment_epoch(self):
        self.epoch_counter += 1

    def _increment_task(self):
        self.task_counter += 1
        logging.info(f"Current task is {self.task_counter}")

    def increment(self, interval: str):
        if interval == "step":
            self._increment_step()
        elif interval == "epoch":
            self._increment_epoch()
        elif interval == "task":
            self._increment_task()
        else:
            raise ValueError("The interval must be one of step, epoch or task.")
