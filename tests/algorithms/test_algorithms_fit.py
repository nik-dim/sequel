import optax
import pytest
import torch

from sequel.algos.jax import ALGOS as JAX_ALGOS
from sequel.algos.pytorch import ALGOS as PYTORCH_ALGOS
from sequel.backbones.jax import MLP as JaxMLP
from sequel.backbones.pytorch import MLP as PytorchMLP
from sequel.benchmarks.mnist import PermutedMNIST
from sequel.benchmarks.utils import ContinualSubsetDataset
from sequel.utils.callbacks.metrics import JaxStandardMetricCallback, PytorchStandardMetricCallback
from sequel.utils.callbacks.tqdm_callback import TqdmCallback


# class DebugBenchmark(SplitMNIST):
#     def _load_mnist(self):
#         super()._load_mnist()
#         num_samples = 2000
#         self.mnist_train = ContinualSubsetDataset(self.mnist_train, indices=list(range(num_samples)))
#         self.mnist_test = ContinualSubsetDataset(self.mnist_test, indices=list(range(num_samples)))


benchmark = PermutedMNIST(num_tasks=2, batch_size=500, subset=1000)


@pytest.mark.skip(reason="Test construction utility function")
@pytest.mark.run_on(["gpu"])
def run_test(algo_name: str, algo_kwargs: dict = {}):
    backbone_kwargs = dict(width=100, n_hidden_layers=2)
    tqdmCallback = TqdmCallback()

    pytorch_backbone = PytorchMLP(**backbone_kwargs)
    pytorch_optimizer = torch.optim.SGD(pytorch_backbone.parameters(), lr=0.01)
    pytorch_callback = PytorchStandardMetricCallback()

    pytorch_kwargs = dict(
        backbone=pytorch_backbone,
        benchmark=benchmark,
        optimizer=pytorch_optimizer,
        callbacks=[pytorch_callback, tqdmCallback],
    )

    pt_algo = PYTORCH_ALGOS[algo_name](**pytorch_kwargs, **algo_kwargs)
    pt_algo.fit(epochs=1)

    jax_backbone = JaxMLP(**backbone_kwargs)
    jax_optimizer = optax.inject_hyperparams(optax.sgd)(learning_rate=0.01)
    jax_callback = JaxStandardMetricCallback()
    jax_kwargs = dict(
        backbone=jax_backbone,
        benchmark=benchmark,
        optimizer=jax_optimizer,
        callbacks=[jax_callback, tqdmCallback],
    )
    jax_algo = JAX_ALGOS[algo_name](**jax_kwargs, **algo_kwargs)
    jax_algo.fit(epochs=1)

    assert set(pt_algo.val_metrics.keys()) == set(jax_algo.val_metrics.keys())


def test_naive():
    algo_name = "naive"
    kwargs = dict()
    run_test(algo_name, kwargs)


def test_agem():
    algo_name = "agem"
    kwargs = dict(per_task_memory_samples=10, memory_batch_size=2, memory_group_by="class")
    run_test(algo_name, kwargs)


def test_ewc():
    algo_name = "ewc"
    kwargs = dict(ewc_lambda=10)
    run_test(algo_name, kwargs)


# def test_icarl():
#     algo_name = "icarl"
#     kwargs = dict(memory_size=10)
#     run_test(algo_name, kwargs)


def test_joint():
    algo_name = "joint"
    kwargs = dict()
    run_test(algo_name, kwargs)


def test_mas():
    algo_name = "mas"
    kwargs = dict(mas_lambda=1)
    run_test(algo_name, kwargs)


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
    run_test(algo_name, kwargs)


def test_si():
    algo_name = "si"
    kwargs = dict(si_lambda=1, xi=1)
    run_test(algo_name, kwargs)


if __name__ == "__main__":
    test_naive()
