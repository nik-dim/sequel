from typing import Optional
import torch

from sequel.algos.pytorch.pytorch_base_algo import PytorchBaseAlgorithm
from sequel.benchmarks.buffer import Buffer
import torch.nn.functional as F


class DER(PytorchBaseAlgorithm):
    """Dark Experience Replay algorithm implemented in PyTorch.

    The equivalent JAX implementation is [`DER in JAX`][sequel.algos.jax.der.DER].

    References:
        [1] Buzzega, P., Boschini, M., Porrello, A., Abati, D. & Calderara, S. Dark experience for general continual
            learning: a strong, simple baseline. in Advances in neural information processing systems 2020.
    """

    def __init__(self, memory_size: int, alpha: float, beta: Optional[float] = None, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.buffer = Buffer(memory_size=memory_size, return_logits=True)
        self.memory_size = memory_size
        self.alpha = alpha

        # Beta is used for DER++
        self.beta = beta

    def on_before_backward(self, *args, **kwargs):
        if len(self.buffer) > 0:
            # if self.task_counter > 1:
            x, y, t, logits = self.buffer.sample_from_buffer(batch_size=self.benchmark.batch_size)
            self.mem_x, self.mem_y, self.mem_t, self.mem_logits = x, y, t, logits
            self.mem_y_hat = self.backbone(self.mem_x, self.mem_t)
            loss = F.mse_loss(self.mem_y_hat, self.mem_logits)
            self.loss += self.alpha * loss

            if self.beta is not None:
                x, y, t, _ = self.buffer.sample_from_buffer(batch_size=self.benchmark.batch_size)
                self.mem_x2, self.mem_y2, self.mem_t2 = x.to(self.device), y.to(self.device), t.to(self.device)
                self.mem_y_hat2 = self.backbone(self.mem_x2, self.mem_t2)
                self.loss += self.beta * self.compute_loss(self.mem_y_hat2, self.mem_y2, self.mem_t2)

    def on_before_optimizer_step(self, *args, **kwargs):
        if self.task_counter > 1:

            return super().on_before_optimizer_step(*args, **kwargs)

    def on_after_training_step(self, *args, **kwargs):
        self.buffer.add_data(self.x, self.y, self.t, self.y_hat.data)

    def __repr__(self) -> str:
        if self.beta is None:
            return f"DER(memory_size={self.memory_size}, alpha={self.alpha})"
        else:
            return f"DER++(memory_size={self.memory_size}, alpha={self.alpha}, beta={self.beta})"
