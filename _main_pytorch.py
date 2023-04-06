import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from sequel.utils.loggers.logging import install_logging
from sequel.utils.callbacks.metrics.pytorch_metric_callback import StandardMetricCallback
from sequel.benchmarks import select_benchmark

from sequel.backbones.pytorch import select_backbone, select_optimizer
from sequel.utils.callbacks.tqdm_callback import TqdmCallback
from sequel.utils.loggers.wandb_logger import WandbLogger

from sequel.algos.pytorch import ALGOS
from sequel.utils.utils import set_seed


def without(d, key):
    new_d = d.copy()
    new_d.pop(key)
    return new_d


@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def my_app(config: DictConfig) -> None:
    install_logging()
    logging.info("The experiment config is:\n" + OmegaConf.to_yaml(config))
    logger = WandbLogger(config)

    set_seed(config.seed)

    mc = StandardMetricCallback()
    tq = TqdmCallback()

    # initialize benchmark (e.g. SplitMNIST)
    benchmark = select_benchmark(config.benchmark)
    logging.info(benchmark)

    # initialize backbone model (e.g. a CNN, MLP)
    backbone = select_backbone(config)
    logging.info(backbone)

    optimizer = select_optimizer(config, backbone)

    algo = ALGOS[config.algo.name.lower()](
        **without(dict(config.algo), "name"),
        backbone=backbone,
        benchmark=benchmark,
        optimizer=optimizer,
        callbacks=[mc, tq],
        loggers=[logger],
    )
    logging.info(algo)

    # start the learning process!
    algo.fit(epochs=config.training.epochs_per_task)


if __name__ == "__main__":
    my_app()
