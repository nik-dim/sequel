import logging
import math
from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from sequel.backbones.pytorch.base_backbone import BaseBackbone

from .pytorch_base_algo import PytorchBaseAlgorithm


class InferenceBlock(nn.Module):
    def __init__(self, input_units: int, d_theta: int, output_units: int):
        """Inits the inference block used by the Kernel Continual Learning algorithm.

        Args:
            input_units (int): dimensionality of the input.
            d_theta (int): dimensionality of the intermediate hidden layers.
            output_units (int): dimensionality of the output.
        """
        super(InferenceBlock, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(input_units, d_theta, bias=True),
            nn.ELU(inplace=True),
            nn.Linear(d_theta, d_theta, bias=True),
            nn.ELU(inplace=True),
            nn.Linear(d_theta, output_units, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class Amortized(nn.Module):
    def __init__(self, input_units: int, d_theta: int, output_units: int):
        """Inits the inference block used by the Kernel Continual Learning algorithm.

        Args:
            input_units (int): dimensionality of the input.
            d_theta (int): dimensionality of the intermediate hidden layers.
            output_units (int): dimensionality of the output.
        """
        super(Amortized, self).__init__()
        self.output_units = output_units
        self.weight_mean = InferenceBlock(input_units, d_theta, output_units)
        self.weight_log_variance = InferenceBlock(input_units, d_theta, output_units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_mean = self.weight_mean(x)
        weight_log_variance = self.weight_log_variance(x)
        return weight_mean, weight_log_variance


class KernelBackboneWrapper(BaseBackbone):
    def __init__(
        self,
        model: BaseBackbone,
        hiddens: int,
        lmd: float,
        num_tasks: int,
        d_rn_f: int,
        kernel_type: Literal["rbf", "rff", "linear", "poly"] = "rff",
    ):
        """Model Wrapper for Kernel Continual Learning. Extracts the encoder of the original backbone and performs the k
        ernel computations outlined in [1].


        Notes:
            The `hiddens` argument can be removed and instead inferred.

        Args:
            model (BaseBackbone): the original backbone. The model must have an encoder component.
            hiddens (int): the dimensionality of the hidden dimensions for the kernel-specific modules.
            lmd (float): The initial value for the lmd Parameter.
            num_tasks (int): the number of tasks to be solved.
            d_rn_f (int): dimensionality of the Random Fourier Features (RFFs). Applicable only if `kernel_type` is 'rff'.
            kernel_type (str, optional): _description_. Defaults to "rbf".
        """
        multihead, classes_per_task, masking_value = model.multihead, model.classes_per_task, model.masking_value
        super().__init__(multihead=multihead, classes_per_task=classes_per_task, masking_value=masking_value)
        self.encoder = model.encoder
        self.d_rn_f = d_rn_f

        self.post = Amortized(hiddens, hiddens, hiddens)
        self.prior = Amortized(hiddens, hiddens, hiddens)

        device = next(model.parameters()).device
        self.lmd = nn.Parameter(torch.tensor([lmd for _ in range(num_tasks)])).to(device)
        self.gma = nn.Parameter(torch.tensor([1.0 for _ in range(num_tasks)])).to(device)
        self.bta = nn.Parameter(torch.tensor([0.0 for _ in range(num_tasks)])).to(device)
        self.kernel_type = kernel_type
        self.bias = 2 * math.pi * torch.rand(d_rn_f, 1).to(device)

    def inner_forward(self, x: torch.Tensor, post: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.encoder(x)
        out_features = self.normalize(out)
        out_mean = torch.mean(out_features, dim=0, keepdim=True)
        if post:
            mu, logvar = self.post(out_mean)
        else:
            mu, logvar = self.prior(out_mean)
        return out_features, mu, logvar

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.kernel_type == "rbf":
            support_kernel = torch.exp(-0.25 * torch.norm(x.unsqueeze(1) - y, dim=2, p=1))
        elif self.kernel_type == "linear":
            support_kernel = x @ y.T
        elif self.kernel_type == "poly":
            support_kernel = (torch.matmul(x, y.T) + 1).pow(3)
        elif self.kernel_type == "rff":
            support_kernel = x.T @ y
        else:
            raise Exception(f"Unknown kenrel. Only support RBF, RFF, POLY, LIN.")
        return support_kernel

    @staticmethod
    def sample(mu: torch.Tensor, logvar: torch.Tensor, L: int, device) -> torch.Tensor:
        shape = (L,) + mu.size()
        eps = torch.randn(shape).to(device)
        return mu.unsqueeze(0) + eps * logvar.exp().sqrt().unsqueeze(0)

    def rand_features(self, bases: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return math.sqrt(2 / self.bias.shape[0]) * torch.cos(torch.matmul(bases, features) + self.bias)

    def compute_kernels(
        self, features_train: torch.Tensor, features_coreset: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = features_coreset.device
        if self.kernel_type == "rff":
            # project to random features
            rs = self.sample(self.r_mu, self.r_log_var, self.d_rn_f, device).squeeze()
            features_coreset = self.rand_features(rs, torch.transpose(features_coreset, 1, 0))
            features_train = self.rand_features(rs, torch.transpose(features_train, 1, 0))

        support_kernel = self.kernel(features_coreset, features_coreset)
        cross_kernel = self.kernel(features_coreset, features_train)
        return support_kernel, cross_kernel

    def forward(
        self, x: torch.Tensor, task_ids: torch.Tensor, coreset_input: torch.Tensor, coreset_target: torch.Tensor
    ) -> torch.Tensor:
        current_task = torch.unique(task_ids)
        assert len(current_task) == 1
        features_train, self.p_mu, self.p_log_var = self.inner_forward(x, post=False)
        features_coreset, self.r_mu, self.r_log_var = self.inner_forward(coreset_input, post=True)

        support_kernel, cross_kernel = self.compute_kernels(features_train, features_coreset)

        alpha = torch.matmul(
            torch.inverse(
                support_kernel
                + (torch.abs(self.lmd[current_task - 1]) + 0.01) * torch.eye(support_kernel.shape[0]).to(x.device)
            ),
            coreset_target,
        )

        out = self.gma[current_task - 1] * torch.matmul(cross_kernel.T, alpha) + self.bta[current_task - 1]

        if self.multihead:
            out = self.select_output_head(out, task_ids)

        return out

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        max_val = x.max()
        min_val = x.min()
        return (x - min_val) / (max_val - min_val)


class KCL(PytorchBaseAlgorithm):
    """Kernel Continual Learning algorithm. The code is adapted from https://github.com/mmderakhshani/KCL/blob/main/stable_sgd/main.py

    KCL is not yet implemented in JAX.

    References:
        [1] Derakhshani, M. M., Zhen, X., Shao, L. & Snoek, C. Kernel Continual Learning. in Proceedings of the 38th
            International Conference on Machine Learning, ICML 2021.
    """

    def __init__(
        self,
        lmd: float,
        core_size: int,
        d_rn_f: int,
        kernel_type: Literal["rbf", "rff", "linear", "poly"],
        tau: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.__check_valid__()

        self.core_size = core_size
        self.kernel_type = kernel_type
        self.tau = tau
        self.coresets = {}

        device = next(self.backbone.parameters()).device
        embedding = self.backbone.encoder(torch.ones(self.input_dimensions).unsqueeze(0).to(device))

        self.backbone = KernelBackboneWrapper(
            self.backbone, hiddens=embedding.numel(), lmd=lmd, num_tasks=self.num_tasks, d_rn_f=d_rn_f
        ).to(device)

    def __check_valid__(self):
        if getattr(self.backbone, "encoder", None) is None:
            raise AttributeError(
                "The backbone must have an encoder subnetwork to be compatible with the implementation of Kernel "
                "Continual Learning. The encoder consists of the entire original backbone except from the last Linear "
                "layer, i.e., the classifier."
            )

    def count_parameters(self) -> int:
        if not isinstance(self.backbone, KernelBackboneWrapper):
            return super().count_parameters()
        return sum([p.numel() for p in self.backbone.parameters() if p.requires_grad])

    def prepare_train_loader(self, task: int) -> DataLoader:
        """Splits the training dataset of the given `task` to training and coreset."""
        dataset = self.benchmark.get_train_dataset(task)
        dataset, coreset = random_split(dataset, lengths=[len(dataset) - self.core_size, self.core_size])
        self.coresets[task] = coreset
        self.register_coreset(coreset)
        return DataLoader(dataset, self.benchmark.batch_size, shuffle=True, **self.benchmark.dl_kwargs)

    def register_coreset(self, coreset):
        num_classes = self.benchmark.num_classes
        x = [sample[0] for sample in coreset]
        y = [sample[1] for sample in coreset]
        self.coreset_input = torch.stack(x).to(self.device)
        self.coreset_target = F.one_hot(torch.tensor(y), num_classes=num_classes).to(self.device).float()

    def forward(self):
        """Performs the forward for the Kernel Continual Learning backbone."""
        self.y_hat = self.backbone.forward(self.x, self.t, self.coreset_input, self.coreset_target)
        return self.y_hat

    def kl_div(self, m: torch.Tensor, log_v: torch.Tensor, m0: torch.Tensor, log_v0: torch.Tensor) -> torch.Tensor:
        """Computes the Kullback-Leibler divergence assuming two normal distributions parameterized by the arguments."""
        v = log_v.exp()
        v0 = log_v0.exp()

        dout, din = m.shape
        const_term = -0.5 * dout * din

        log_std_diff = 0.5 * torch.sum(torch.log(v0) - torch.log(v))
        mu_diff_term = 0.5 * torch.sum((v + (m0 - m) ** 2) / v0)
        kl = const_term + log_std_diff + mu_diff_term
        return kl

    def training_step(self, *args, **kwargs):
        self.optimizer_zero_grad()

        self.y_hat = self.forward()
        self.loss = F.cross_entropy(self.y_hat, self.y)

        if self.kernel_type == "rff":
            r_mu, r_log_var = self.backbone.r_mu, self.backbone.r_log_var
            p_mu, p_log_var = self.backbone.p_mu, self.backbone.p_log_var
            self.loss += self.tau * self.kl_div(r_mu, r_log_var, p_mu, p_log_var)
        self.loss.backward()
        self.optimizer.step()

    def on_before_val_epoch(self, *args, **kwargs):
        logging.info(f"Setting the coreset for validating task {self.current_val_task}.")
        self.register_coreset(self.coresets[self.current_val_task])
        return super().on_before_val_epoch(*args, **kwargs)
