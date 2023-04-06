import logging
from typing import Any, Dict, Iterable, Optional, Union
import omegaconf

import optax
import torch

from sequel.algos.utils import BaseCallbackHook, BaseStateManager
from sequel.backbones.jax import BaseBackbone as JaxBaseBackbone
from sequel.backbones.pytorch import BaseBackbone as PytorchBaseBackbone
from sequel.benchmarks import Benchmark
from sequel.utils.callbacks.base_callback import BaseCallback
from sequel.utils.loggers import Logger, install_logging


class BaseAlgorithm(BaseStateManager, BaseCallbackHook, BaseCallback):
    """Base class for Trainer component. Handles all the engineering code. Connects the algorighm with callback and
    logging functionallities. The class also inherits from BaseCallback and the user can implement desired
    functionalities either as standalone callbacks or by overwriting the parent callback hooks of the algorithm.

    Attributes:
        metric_callback_msg (Optional[str]): A message set by the MetricCallback that informs about the progress of
            training/validation etc. Can be used by other callbacks, e.g., TqdmCallback, to display such information
            in the console.
        num_tasks (int): number of tasks. Set by [`parse_benchmark`][sequel.algos.base_algo.BaseAlgorithm.parse_benchmark].
        classes_per_task (int): the number of classes per task. For the moment, all tasks should have the same number
            of classes. Set by [`parse_benchmark`][sequel.algos.base_algo.BaseAlgorithm.parse_benchmark].
        episodic_memory_loader (torch.utils.data.DataLoader): The dataloader for the memory. Applies to methods that
            utilize memoreis, such as GEM.
        episodic_memory_iter (Iterable[torch.utils.data.DataLoader]): An iterator for `episodic_memory_loader`
        loss (Union[torch.Tensor, numpy.array]): The loss of the current batch.
        current_dataloader (torch.utils.data.DataLoader): The current training/validation/testing dataloader.
        x (Union[torch.Tensor, numpy.array]): The input tensors of the current batch. Set by
            [`unpack_batch`][sequel.algos.base_algo.BaseAlgorithm.unpack_batch].
        y (Union[torch.Tensor, numpy.array]): The targets of the current batch. Set by
            [`unpack_batch`][sequel.algos.base_algo.BaseAlgorithm.unpack_batch].
        t (Union[torch.Tensor, numpy.array]): The task ids of the current batch. Set by
            [`unpack_batch`][sequel.algos.base_algo.BaseAlgorithm.unpack_batch].
        bs (int): The size of the current batch. Set by [`unpack_batch`][sequel.algos.base_algo.BaseAlgorithm.unpack_batch].
        epochs (int): The epochs each task is trained for.
    """

    metric_callback_msg = None
    episodic_memory_loader = None
    episodic_memory_iter = None

    def __init__(
        self,
        backbone: Union[PytorchBaseBackbone, JaxBaseBackbone],
        benchmark: Benchmark,
        optimizer: Union[torch.optim.Optimizer, optax.GradientTransformation],
        callbacks: Iterable[BaseCallback] = [],
        loggers: Optional[Iterable[Logger]] = None,
        lr_decay: Optional[float] = None,
        grad_clip: Optional[float] = None,
        reinit_optimizer: bool = True,
    ) -> None:
        """Inits the BaseAlgorithm class. Handles all the engineering code. Base classes for algorithms in Pytorch and
        Jax inherit from this class.

        Args:
            backbone (Union[PytorchBaseBackbone, JaxBaseBackbone]): The backbone model, e.g., a CNN.
            benchmark (Benchmark): The benchmark, e.g., SplitMNIST.
            optimizer (Union[torch.optim.Optimizer, optax.GradientTransformation]): The optimizer used to update the
                backbone weights.
            callbacks (Iterable[BaseCallback], optional): A list of callbacks. At least one instance of MetricCallback
                should be given. Defaults to [].
            loggers (Optional[Logger], optional): A list of logger, e.g. for Weights&Biases logging functionality.
                Defaults to None.
            lr_decay (Optional[float], optional): A learning rate decay used for every new task. Defaults to None.
            reinit_optimizer (bool): Indicates whether the optimizer state is reinitialized before fitting a new task.
                Defaults to True.
        """

        install_logging()
        self.benchmark = benchmark
        self.parse_benchmark()
        self.backbone = backbone

        self.callbacks = self.check_and_parse_callbacks(callbacks)
        self.loggers = loggers
        self.optimizer = optimizer

        self.lr_decay = lr_decay
        self.reinit_optimizer = reinit_optimizer
        self.grad_clip = grad_clip

        if self.grad_clip is not None:
            logging.info(f"Gradient clipping has been set to {self.grad_clip}.")

        logging.info(f"The backbone model has {self.count_parameters()/1e6:.3f}m parameters")

    def check_and_parse_callbacks(self, callbacks: Iterable[BaseCallback]) -> Iterable[BaseCallback]:
        """Checks that the callbacks is a list containing exaclty one MetricCallback.

        Args:
            callbacks (Iterable[BaseCallback]): list of callbacks

        Returns:
            Iterable[BaseCallback]: the parsed list of callbacks.
        """
        from sequel.utils.callbacks.metrics.metric_callback import MetricCallback

        assert isinstance(callbacks, list), "The callbacks should be given as a list."
        assert (
            sum([isinstance(c, MetricCallback) for c in callbacks]) == 1
        ), "Exactly one instance of MetricCallback should be given."

        # make sure that the MetricCallback is last.
        parsed_callbacks = [c for c in callbacks if not isinstance(c, MetricCallback)]
        parsed_callbacks += [c for c in callbacks if isinstance(c, MetricCallback)]
        return parsed_callbacks

    def parse_benchmark(self):
        """Extracts attributes from the benchmark and registers them to the algo for quick access."""
        self.num_tasks = self.benchmark.num_tasks
        self.classes_per_task = self.benchmark.classes_per_task
        self.input_dimensions = self.benchmark.dimensions

    def update_episodic_memory(self) -> None:
        """Updates the episodic memory. This funciton is called after fitting one task."""
        logging.info("Updating episodic memory for task {}".format(self.task_counter))
        self.episodic_memory_loader = self.benchmark.memory_dataloader(self.task_counter)
        self.episodic_memory_iter = iter(self.episodic_memory_loader)

    def sample_batch_from_memory(self):
        try:
            batch = next(self.episodic_memory_iter)
        except StopIteration:
            # makes the dataloader an infinite stream
            self.episodic_memory_iter = iter(self.episodic_memory_loader)
            batch = next(self.episodic_memory_iter)

        return batch

    def log(self, item):
        # logger: Logger
        if self.loggers is not None:
            for logger in self.loggers:
                logger.log(item, step=self.step_counter, epoch=self.epoch_counter)

    def log_figure(self, figure, name):
        if self.loggers is not None:
            for logger in self.loggers:
                logger.log_figure(name=name, figure=figure)

    def count_parameters(self):
        raise NotImplementedError

    def setup(self):
        for cb in self.callbacks:
            cb.connect(self)

    def teardown(self):
        pass

    def _configure_criterion(self, task_id=None):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """Calls the forward function of the model."""
        raise NotImplementedError

    def update_tqdm(self, msg):
        self.metric_callback_msg = msg
        # self.tqdm_dl.set_postfix(msg)

    def unpack_batch(self, batch: Any):
        """Unpacks the batch and registers to the algorithm the current batch input, targets and task ids as `self.x`,
        `self.y` and `self.t`, respectively. It also registers the current batch size as `self.bs`"""
        raise NotImplementedError

    def optimizer_zero_grad(self):
        raise NotImplementedError

    def backpropagate_loss(self):
        raise NotImplementedError

    def optimizer_step(self):
        raise NotImplementedError

    def perform_gradient_clipping(self):
        raise NotImplementedError

    def training_step(self, *args, **kwargs):
        """The training step, i.e. training for each batch.

        Goes through the usual hoops of zeroing out the optimizer, forwarding the input, computing the loss,
        backpropagating and updating the weights. For each different steps, callabacks are offered for maximum
        versatility and ease of use.
        """
        self.optimizer_zero_grad()
        y_hat = self.forward()
        self.loss = self.compute_loss(y_hat, self.y, self.t)

        self.on_before_backward()
        self.on_before_backward_callbacks()
        self.backpropagate_loss()
        self.on_after_backward()
        self.on_after_backward_callbacks()

        self.perform_gradient_clipping()

        self.on_before_optimizer_step()
        self.on_before_optimizer_step_callbacks()
        self.optimizer_step()
        self.on_after_optimizer_step()
        self.on_after_optimizer_step_callbacks()

    def valid_step(self, *args, **kwargs):
        """Performs the validation step.Callbacks are offered for each step of the process."""
        raise NotImplementedError

    def test_step(self, *args, **kwargs):
        """Performs the testing step. Callbacks are offered for each step of the process."""
        pass

    def training_epoch(self, *args, **kwargs):
        """Trains the model for a single epoch. Callbacks are offered for each method."""
        self.increment("epoch")
        self.set_training_mode()
        self.current_dataloader = self.train_loader
        for self.batch_idx, batch in enumerate(self.current_dataloader):
            self.unpack_batch(batch)
            self.on_before_training_step()
            self.on_before_training_step_callbacks()
            self.increment("step")
            self.training_step()
            self.on_after_training_step()
            self.on_after_training_step_callbacks()

    def eval_epoch(self, *args, **kwargs):
        """Performs the evaluation of the model on the validation set. If no validation dataloader is provided, the
        method returns without any computation."""
        if self.valid_loader is None:
            return

        self.set_evaluation_mode()
        self.current_dataloader = self.valid_loader

        for self.batch_idx, batch in enumerate(self.current_dataloader):
            self.unpack_batch(batch)
            self.on_before_val_step()
            self.on_before_val_step_callbacks()
            self.valid_step()
            self.on_after_val_step()
            self.on_after_val_step_callbacks()

    def test_epoch(self, *args, **kwargs):
        pass

    def prepare_for_next_task(self, task):
        raise NotImplementedError

    def prepare_train_loader(self, task):
        return self.benchmark.train_dataloader(task)

    def train_algorithm_on_task(self, task: int):
        """Fits a *single* task."""
        self.train_loader = self.prepare_train_loader(task)
        self.prepare_for_next_task(task)

        assert isinstance(self._epochs, (list, int, omegaconf.listconfig.ListConfig))
        if not isinstance(self._epochs, int):
            self.epochs = self._epochs[self.task_counter - 1]
        else:
            self.epochs = self._epochs

        for self.current_task_epoch in range(1, self.epochs + 1):
            self._train_loop()
            self._val_loop()

    def _train_loop(self):
        self.on_before_training_epoch()
        self.on_before_training_epoch_callbacks()
        self.training_epoch()
        self.on_after_training_epoch()
        self.on_after_training_epoch_callbacks()

    def _val_loop(self):
        # after each epoch, the model is validated on current and previous tasks.
        self.on_before_validating_algorithm_on_all_tasks()
        self.on_before_validating_algorithm_on_all_tasks_callbacks()
        self.validate_algorithm_on_all_tasks()
        self.on_after_validating_algorithm_on_all_tasks()
        self.on_after_validating_algorithm_on_all_tasks_callbacks()

    def validate_algorithm_on_all_tasks(self) -> Dict[str, float]:
        for task in range(1, self.task_counter + 1):
            self.current_val_task = task
            self.valid_loader = self.benchmark.val_dataloader(task)

            self.on_before_val_epoch()
            self.on_before_val_epoch_callbacks()
            self.eval_epoch()
            self.on_after_val_epoch()
            self.on_after_val_epoch_callbacks()

    def _fit(self):
        """Fits all tasks to the model, one after the other."""
        for task in range(1, self.num_tasks + 1):
            self.on_before_training_task()
            self.on_before_training_task_callbacks()
            self.increment("task")
            self.train_algorithm_on_task(task)
            self.on_after_training_task()
            self.on_after_training_task_callbacks()

    def _run_setup(self):
        self.on_before_setup()
        self.on_before_setup_callbacks()
        self.setup()
        self.on_after_setup()
        self.on_after_setup_callbacks()

    def _run_fit(self):
        self.on_before_fit()
        self.on_before_fit_callbacks()
        self._fit()
        self.on_after_fit()
        self.on_after_fit_callbacks()

    def _run_teardown(self):
        self.on_before_teardown()
        self.on_before_teardown_callbacks()
        self.teardown()
        self.on_after_teardown()
        self.on_after_teardown_callbacks()

    def fit(self, epochs):
        self._epochs = epochs

        self._run_setup()
        self._run_fit()
        self._run_teardown()

    def compute_loss(self, predictions, targets, task_ids, *args, **kwargs):
        raise NotImplementedError
