import warnings
from sequel.algos.jax import ALGOS as JAX_ALGOS
from sequel.algos.pytorch import ALGOS as PYTORCH_ALGOS
from sequel.benchmarks.mnist import SplitMNIST
from sequel.backbones.pytorch import MLP as PytorchMLP
from sequel.backbones.jax import MLP as JaxMLP
import optax
import torch
from sequel.utils.callbacks.metrics import PytorchStandardMetricCallback
from sequel.utils.callbacks.metrics import JaxStandardMetricCallback

benchmark = SplitMNIST(num_tasks=2, batch_size=1024)

jax_backbone = JaxMLP()
jax_optimizer = optax.sgd(0.1)
jax_callback = JaxStandardMetricCallback()

pytorch_backbone = PytorchMLP(10, 2)
pytorch_optimizer = torch.optim.SGD(pytorch_backbone.parameters(), lr=1)
pytorch_callback = PytorchStandardMetricCallback()


jax_kwargs = dict(
    backbone=jax_backbone,
    benchmark=benchmark,
    optimizer=jax_optimizer,
    callbacks=[jax_callback],
)

pytorch_kwargs = dict(
    backbone=pytorch_backbone,
    benchmark=benchmark,
    optimizer=pytorch_optimizer,
    callbacks=[pytorch_callback],
)


def test_lamaml():
    algo_name = "lamaml"
    kwargs = dict(
        glances=5,
        n_inner_updates=5,
        second_order=False,
        grad_clip_norm=2.0,
        learn_lr=True,
        lr_alpha=0.3,
        sync_update=False,
        mem_size=200,
    )

    warnings.warn("Not yet implemented for Jax.")
    algo = PYTORCH_ALGOS[algo_name](**pytorch_kwargs, **kwargs)
    # algo = JAX_ALGOS[algo_name](**jax_kwargs, **kwargs)


def test_naive():
    algo_name = "naive"
    kwargs = dict()

    algo = PYTORCH_ALGOS[algo_name](**pytorch_kwargs, **kwargs)
    algo = JAX_ALGOS[algo_name](**jax_kwargs, **kwargs)


def test_agem():
    algo_name = "agem"
    kwargs = dict(per_task_memory_samples=10, memory_batch_size=2, memory_group_by="class")

    algo = PYTORCH_ALGOS[algo_name](**pytorch_kwargs, **kwargs)
    algo = JAX_ALGOS[algo_name](**jax_kwargs, **kwargs)


def test_der():
    algo_name = "der"
    kwargs = dict(alpha=1, memory_size=1000)

    algo = PYTORCH_ALGOS[algo_name](**pytorch_kwargs, **kwargs)
    algo = JAX_ALGOS[algo_name](**jax_kwargs, **kwargs)


def test_derpp():
    algo_name = "der"
    kwargs = dict(alpha=1, memory_size=1000, beta=1)

    algo = PYTORCH_ALGOS[algo_name](**pytorch_kwargs, **kwargs)
    algo = JAX_ALGOS[algo_name](**jax_kwargs, **kwargs)


def test_er():
    algo_name = "er"
    kwargs = dict(memory_batch_size=10, per_task_memory_samples=100, memory_group_by="task")

    warnings.warn("Not yet implemented for Jax.")
    algo = PYTORCH_ALGOS[algo_name](**pytorch_kwargs, **kwargs)
    # algo = JAX_ALGOS[algo_name](**jax_kwargs, **kwargs)


def test_ewc():
    algo_name = "ewc"
    kwargs = dict(ewc_lambda=10)

    algo = PYTORCH_ALGOS[algo_name](**pytorch_kwargs, **kwargs)
    algo = JAX_ALGOS[algo_name](**jax_kwargs, **kwargs)


def test_icarl():
    algo_name = "icarl"
    kwargs = dict(memory_size=10)

    warnings.warn("Not yet implemented for Jax.")
    algo = PYTORCH_ALGOS[algo_name](**pytorch_kwargs, **kwargs)
    # algo = JAX_ALGOS[algo_name](**jax_kwargs, **kwargs)


def test_joint():
    algo_name = "joint"
    kwargs = dict()

    algo = PYTORCH_ALGOS[algo_name](**pytorch_kwargs, **kwargs)
    algo = JAX_ALGOS[algo_name](**jax_kwargs, **kwargs)


def test_kcl():
    algo_name = "kcl"
    kwargs = dict(core_size=20, d_rn_f=2048, kernel_type="rff", lmd=0.1, tau=0.01)

    warnings.warn("Not yet implemented for Jax.")
    algo = PYTORCH_ALGOS[algo_name](**pytorch_kwargs, **kwargs)
    # algo = JAX_ALGOS[algo_name](**jax_kwargs, **kwargs)


def test_lfl():
    algo_name = "lfl"
    kwargs = dict(lfl_lambda=1.0)

    warnings.warn("Not yet implemented for Jax.")
    algo = PYTORCH_ALGOS[algo_name](**pytorch_kwargs, **kwargs)
    # algo = JAX_ALGOS[algo_name](**jax_kwargs, **kwargs)


def test_mas():
    algo_name = "mas"
    kwargs = dict(mas_lambda=1)

    algo = PYTORCH_ALGOS[algo_name](**pytorch_kwargs, **kwargs)
    algo = JAX_ALGOS[algo_name](**jax_kwargs, **kwargs)


def test_mcsgd():
    algo_name = "mcsgd"
    kwargs = dict(
        per_task_memory_samples=100,
        lmc_policy="offline",
        lmc_interpolation="linear",
        lmc_lr=0.05,
        lmc_momentum=0.8,
        lmc_batch_size=64,
        lmc_init_position=0.1,
        lmc_line_samples=10,
        lmc_epochs=1,
    )

    algo = PYTORCH_ALGOS[algo_name](**pytorch_kwargs, **kwargs)
    algo = JAX_ALGOS[algo_name](**jax_kwargs, **kwargs)


def test_si():
    algo_name = "si"
    kwargs = dict(si_lambda=1, xi=1)

    algo = PYTORCH_ALGOS[algo_name](**pytorch_kwargs, **kwargs)
    algo = JAX_ALGOS[algo_name](**jax_kwargs, **kwargs)
