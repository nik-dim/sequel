from typing import Dict, Literal

import numpy as np
import torch
from tqdm import tqdm

from sequel.benchmarks.memory import MemoryMechanism

from .pytorch_base_algo import PytorchBaseAlgorithm
from .utils.weight_gradient_manipulation import get_weights, set_grads, set_weights


class MCSGD(PytorchBaseAlgorithm):
    """MC-SGD: Mode Connectivity-Stochastic Gradient Descent. Inherits from BaseAlgorithm.

    The equivalent JAX implementation is [`MCSGD in JAX`][sequel.algos.jax.mcsgd.MCSGD].

    References:
        [1] Mirzadeh, S.-I., Farajtabar, M., Görür, D., Pascanu, R. & Ghasemzadeh, H. Linear Mode Connectivity in
            Multitask and Continual Learning. in 9th International Conference on Learning Representations, ICLR 2021.
    """

    def __init__(
        self,
        per_task_memory_samples: int = 100,
        memory_group_by: Literal["task", "class"] = "task",
        lmc_policy="offline",
        lmc_interpolation="linear",
        lmc_lr=0.05,
        lmc_momentum=0.8,
        lmc_batch_size=64,
        lmc_init_position=0.1,
        lmc_line_samples=10,
        lmc_epochs=1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.memory = MemoryMechanism(per_task_memory_samples=per_task_memory_samples, groupby=memory_group_by)
        self.w_bar_prev = None
        self.w_hat_curr = None

        # parse init arguments
        self.per_task_memory_samples = per_task_memory_samples
        self.lmc_policy = lmc_policy
        self.lmc_interpolation = lmc_interpolation
        self.lmc_lr = lmc_lr
        self.lmc_momentum = lmc_momentum
        self.lmc_batch_size = lmc_batch_size
        self.lmc_init_position = lmc_init_position
        self.lmc_line_samples = lmc_line_samples
        self.lmc_epochs = lmc_epochs

    def __repr__(self) -> str:
        return (
            "MCSGD("
            + f"per_task_memory_samples={self.per_task_memory_samples}, "
            + f"policy={self.lmc_policy}, "
            + f"interpolation={self.lmc_interpolation}, "
            + f"lr={self.lmc_lr}, "
            + f"momentum={self.lmc_momentum}, "
            + f"batch_size={self.lmc_batch_size}, "
            + f"init_position={self.lmc_init_position}, "
            + f"line_samples={self.lmc_line_samples}, "
            + f"epochs={self.lmc_epochs}"
            + ")"
        )

    def calculate_line_loss(self, w_start, w_end, loader):
        line_samples = np.arange(0.0, 1.01, 1.0 / float(self.lmc_line_samples))
        grads = 0
        for t in tqdm(line_samples, desc="Line samples"):
            w_mid = w_start + (w_end - w_start) * t
            m = set_weights(self.backbone, w_mid)
            self.calculate_point_loss(m, loader).backward()
            grads += torch.cat([p.grad.view(-1) for _, p in m.named_parameters()])
        return grads

    def calculate_point_loss(self, model, loader):
        criterion = self._configure_criterion()
        model.eval()
        total_loss, total_count = 0.0, 0.0
        for batch in loader:
            self.unpack_batch(batch)
            self.y_hat = model(self.x, self.t)

            total_loss += criterion(self.y_hat, self.y)
            total_count += self.bs

        return total_loss / total_count

    def find_connected_minima(self, task):
        bs = self.lmc_batch_size
        loader_curr = self.benchmark.train_dataloader_subset(
            task, batch_size=bs, subset_size=self.per_task_memory_samples
        )
        loader_prev = self.benchmark.memory_dataloader(task, batch_size=bs, return_infinite_stream=False)

        mc_model = set_weights(
            self.backbone, self.w_bar_prev + (self.w_hat_curr - self.w_bar_prev) * self.lmc_init_position
        )
        optimizer = torch.optim.SGD(mc_model.parameters(), lr=self.lmc_lr, momentum=self.lmc_momentum)

        mc_model.train()
        optimizer.zero_grad()
        grads_prev = self.calculate_line_loss(self.w_bar_prev, get_weights(mc_model), loader_prev)
        grads_curr = self.calculate_line_loss(self.w_hat_curr, get_weights(mc_model), loader_curr)
        mc_model = set_grads(mc_model, (grads_prev + grads_curr))
        optimizer.step()
        return mc_model

    def on_after_training_epoch(self, *args, **kwargs):
        # save the weights of the current Continual Learning solution
        self.w_hat_curr = get_weights(self.backbone)

    def validate_algorithm_on_all_tasks(self) -> Dict[str, float]:
        if self.task_counter == 1:
            super().validate_algorithm_on_all_tasks()

    def on_after_validating_algorithm_on_all_tasks_callbacks(self):
        if self.task_counter == 1:
            return super().on_after_validating_algorithm_on_all_tasks_callbacks()

    def on_after_training_task(self, *args, **kwargs):
        """After training for a task similarly to the naïve algorithm, MCSGD performs another round of epochs
        corresponding to the linear connectivity updates of the algorithm.

        Note that the validation is performed with the weights obtained at the end of these updates.
        """

        # update the memory to include samples from the current task
        self.memory.update_memory(self)
        if self.task_counter > 1:
            self.backbone = self.find_connected_minima(self.task_counter)
            # perform the validation with the weights obtained after the mode-connectivity updates
            super().on_before_validating_algorithm_on_all_tasks_callbacks()
            super().validate_algorithm_on_all_tasks()
            super().on_after_validating_algorithm_on_all_tasks_callbacks()

        # save the backbone obtained from the mode-connectivity updates
        # as the Multi-Task approximate solution
        self.w_bar_prev = get_weights(self.backbone)

        # revert the weights of the backbone to the Continual Learning solution
        self.backbone = set_weights(self.backbone, self.w_hat_curr)
