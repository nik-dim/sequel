import warnings
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch import FunctionalModule, make_functional

from sequel.benchmarks.buffer import Buffer

from .pytorch_base_algo import PytorchBaseAlgorithm


class LaMAML(PytorchBaseAlgorithm):
    """Look-Ahead Model Agnostic Meta Learning implementation in PyTorch.

    LaMAML is not yet implemented in JAX.

    References:
        [1] Gupta, G., Yadav, K. & Paull, L. Look-ahead meta learning for continual learning. in Advances in neural
            information processing systems 202.
    """

    backbone_func: FunctionalModule
    params: List[torch.nn.Parameter]

    def __init__(
        self,
        mem_size: int,
        glances: int = 5,
        n_inner_updates: int = 5,
        second_order: bool = False,
        grad_clip_norm: float = 2.0,
        learn_lr: bool = True,
        lr_alpha: float = 0.3,
        sync_update: bool = False,
        initial_alpha_value: float = 0.15,
        lr_weights: float = 0.1,
        *args,
        **kwargs
    ):
        """Inits the LaMAML algorithm class.

        Args:
            mem_size (int): The size of the memory.
            glances (int, optional): The number of gradient steps performed on the current batch. Defaults to 5.
            n_inner_updates (int, optional): The number of updates performed for the inner step of the Meta Learning
                process. The batch is split into `n_inner_updates` sub-batches. Defaults to 5.
            second_order (bool, optional): Boolean denoting whether the computational graph is kept for second-order
                derivative calculations. Defaults to False.
            grad_clip_norm (float, optional): The max norm of the gradients. Defaults to 2.0.
            learn_lr (bool, optional): Boolean denoting whether the per-parameter learning rate is learned or not.
                Defaults to True.
            lr_alpha (float, optional): The learning rate for the parameters corresponding to the learnt learning rate
                for the weights. Defaults to 0.3.
            sync_update (bool, optional): _description_. Defaults to False.
            initial_alpha_value (float, optional): The initial value for the per-parameter learning rate. Defaults to 0.15.
            lr_weights (float, optional): The learning rate for the weights. Applies onl if `sync_update` is set to
                True. Defaults to 0.1.
        """
        super().__init__(*args, **kwargs)

        self.glances = glances
        self.n_inner_updates = n_inner_updates
        self.second_order = second_order
        self.grad_clip_norm = grad_clip_norm
        self.learn_lr = learn_lr
        self.lr_alpha = lr_alpha
        self.sync_update = sync_update
        self.initial_alpha_value = initial_alpha_value
        self.mem_size = mem_size
        self.lr_weights = lr_weights
        self.buffer = Buffer(memory_size=mem_size)

        self.backbone_func, self.params = make_functional(self.backbone)

        alpha_params = [nn.Parameter(initial_alpha_value * torch.ones_like(p)) for p in self.params]
        self.alpha_lr = nn.ParameterList(alpha_params).to(self.device)

        self.opt_lr = torch.optim.SGD(self.alpha_lr.parameters(), lr=lr_alpha)
        if self.sync_update:
            self.opt_wt = torch.optim.SGD(self.params, lr=self.lr_weights)

        warnings.warn(
            "The LaMAML implementation disposes of the optimizer provided in the class arguments. The newly-defined"
            " optimizer concerns the parameters responsible for the learning rate of the underlying backbone parameters."
        )

        warnings.warn(
            "The argument `n_inner_updates` is not used at the moment. It is automatically set to the number of "
            "samples in a batch. Hence, the inner update is performed with only one sample."
        )

        warnings.warn("LaMAML does not currently support benchmarks with task ids, such as SPlitCIFAR100.")

    def forward(self) -> torch.Tensor:
        self.y_hat = self.backbone_func(self.params, self.x)
        return self.y_hat

    def meta_loss(self, fast_weights, x, y, t) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.backbone_func(fast_weights, x)
        loss_q = self.compute_loss(logits.squeeze(1), y)
        return loss_q, logits

    def inner_update(self, fast_weights, x, y, t) -> List[torch.nn.Parameter]:
        if fast_weights is None:
            fast_weights = self.params

        logits = self.backbone_func(fast_weights, x)
        loss = self.compute_loss(logits.squeeze(), y)

        # NOTE if we want higher order grads to be allowed, change create_graph=False to True
        graph_required = self.second_order
        grads = torch.autograd.grad(loss, fast_weights, create_graph=graph_required, retain_graph=graph_required)
        grads = [torch.clamp(g, min=-self.grad_clip_norm, max=self.grad_clip_norm) for g in grads]

        fast_weights = list(map(lambda p, g, a: p - g * F.relu(a), fast_weights, grads, self.alpha_lr))
        return fast_weights

    def observe(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> float:
        # self.backbone.train()
        self.orig_x, self.orig_y, self.orig_t = x, y, t
        for self.glance_idx in range(self.glances):
            perm = torch.randperm(x.size(0))
            x = x[perm]
            y = y[perm]

            self.opt_lr.zero_grad()

            fast_weights = None
            meta_losses = []

            self.x, self.y, self.t = self.buffer.augment_batch_with_memory(x, y, t)
            # `n_inner_updates` is set to the batch size implicitly.
            for batch_x, batch_y, batch_t in zip(x, y, t):
                fast_weights = self.inner_update(fast_weights, batch_x, batch_y, batch_t)
                if self.current_task_epoch == 1:
                    self.buffer.add_data(batch_x, batch_y, batch_t)

                meta_loss, self.y_hat = self.meta_loss(fast_weights, self.x, self.y, self.t)
                meta_losses.append(meta_loss)

            # Taking the meta gradient step (will update the learning rates)
            self.opt_lr.zero_grad()
            meta_loss: torch.Tensor = sum(meta_losses) / len(meta_losses)
            meta_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(self.alpha_lr.parameters(), self.grad_clip_norm)

            if self.learn_lr:
                self.opt_lr.step()

            if self.sync_update:
                self.opt_wt.step()
                self.opt_wt.zero_grad()
                self.alpha_lr.zero_grad()
            else:
                for i, p in enumerate(self.params):
                    p.data = p.data - p.grad * F.relu(self.alpha_lr[i])
                for p in self.params:
                    p.grad.zero_()
                self.alpha_lr.zero_grad()

        self.loss = meta_loss
        return meta_loss.item()

    def training_step(self, *args, **kwargs):
        self.observe(self.x, self.y, self.t)

    def _configure_optimizers(self, task):
        pass
