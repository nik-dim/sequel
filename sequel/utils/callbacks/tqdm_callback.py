import sys
from typing import TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from sequel.algos.base_algo import BaseAlgorithm

from sequel.utils.callbacks.algo_callback import AlgoCallback

import time


class TqdmCallback(AlgoCallback):
    """This Callback displays a tqdm progress bar for training and validation allowing the user to gauge the progress
    of training/evaluation. This callback uses the `metric_callback_msg` of the current `BaseAlgorithm` instance that
    is set by the corresponing `MetricCallback` instance to infuse additional information as a posfix to the progress
    bar.

    Given that for Continual Learning validation is performed for all past and current datasets, the console output
    would get filled with increasingly more uninformative bars. For this reason, the tqdm progress bar for validaiton
    is erased upon completion.
    """

    def on_before_training_step(self, algo: "BaseAlgorithm", *args, **kwargs):
        """Initializes the progress bar for the current training set. The tqdm bar is not defined by
        `on_before_train_epoch` callback hook so that is compatible with the structure of the `BaseAlgorithm`; the
        `current_dataloader` attribute is set in the `training_epoch` method.

        Args:
            algo (BaseAlgorithm): the current algorighm instance.
        """
        if algo.batch_idx == 0:
            self.tqdm_dl = tqdm(
                algo.current_dataloader,
                desc=f"Task {algo.task_counter} -- Epoch {algo.epoch_counter}",
                file=sys.stdout,  # so that tqdm does not print out of order,
                total=len(algo.current_dataloader),
            )

    def on_after_training_epoch(self, algo: "BaseAlgorithm", *args, **kwargs):
        self.tqdm_dl.close()

    def on_before_validating_step(self, algo: "BaseAlgorithm", *args, **kwargs):
        """Initializes the progress bar for the current validation set. The tqdm bar is not defined by
        `on_before_eval_epoch` callback hook so that is compatible with the structure of the `BaseAlgorithm`; the
        `current_dataloader` attribute is set by the `eval_epoch` method.

        Args:
            algo (BaseAlgorithm): the current algorighm instance.
        """
        if algo.batch_idx == 0:
            self.tqdm_dl = tqdm(
                algo.current_dataloader,
                desc=f"Validating Task {algo.task_counter} -- Epoch {algo.epoch_counter}",
                file=sys.stdout,  # so that tqdm does not print out of order,
                leave=False,
                total=len(algo.current_dataloader),
            )

    def update_bar(self, algo: "BaseAlgorithm"):
        """Increments the progress bar. If `metric_callback_msg` is set, it is set as a postfix.

        Args:
            algo (BaseAlgorithm): the current algorighm instance.
        """
        self.tqdm_dl.update(1)
        if algo.metric_callback_msg is not None:
            self.tqdm_dl.set_postfix(algo.metric_callback_msg)

    def on_after_training_step(self, algo: "BaseAlgorithm", *args, **kwargs):
        self.update_bar(algo)

    # def on_after_val_step(self, algo: "BaseAlgorithm", *args, **kwargs):
    #     self.update_bar(algo)

    def on_after_testing_step(self, algo: "BaseAlgorithm", *args, **kwargs):
        self.update_bar(algo)

    def on_before_validating_algorithm_on_all_tasks(self, algo: "BaseAlgorithm", *args, **kwargs):
        # print("\nhey\n", algo.task_counter)
        self.val_tqdm = tqdm(total=algo.task_counter, position=0, leave=True, desc="Validating...")

    def on_after_val_epoch(self, algo: "BaseAlgorithm", *args, **kwargs):
        pass
        # print(algo.current_val_task)
        self.val_tqdm.update(1)
        # self.val_tqdm.set_description_str(str(algo.current_val_task))
        # self.val_tqdm.set_postfix_str(algo.current_val_task)

    def on_after_validating_algorithm_on_all_tasks(self, algo: "BaseAlgorithm", *args, **kwargs):
        self.val_tqdm.close()
