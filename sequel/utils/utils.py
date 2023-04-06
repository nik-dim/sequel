import logging
import os
import random
from pathlib import Path
from typing import Callable, NamedTuple, Union

import hydra
import numpy as np
import omegaconf
import optax
import pandas as pd
import torch

import wandb


def get_experiment_root_dir() -> str:
    if wandb.run is None:
        return os.getcwd()
    else:
        return wandb.run.dir


def get_original_root_dir() -> str:
    if hydra.utils.HydraConfig().initialized():
        return hydra.utils.get_original_cwd()
    else:
        return os.getcwd()


def get_original_source_code_root_dir() -> str:
    if hydra.utils.HydraConfig().initialized():
        root = hydra.utils.get_original_cwd()
    else:
        root = os.getcwd()

    return Path(root, "sequel").absolute().as_posix()


def safe_conversion(item: Union[torch.Tensor, float]) -> float:
    """Performs a conversion, if necessary, from a PyTorch Tensor to a float. This function is mainly used prior to
    logging.

    Args:
        item (Union[torch.Tensor, float]): The value.

    Returns:
        float: The converted value.
    """
    if isinstance(item, torch.Tensor):
        return item.cpu().item()
    else:
        return item


class JaxOptState(NamedTuple):
    """Convenience class used to store the state for JAX optimizer. Useful for reinitializing the optimizer and in
    conjuction with schedulers.

    Note:
        This clas is not used at the moment.
    """

    tx: optax.GradientTransformation
    hyperparams: dict = None

    def make_optimizer(self) -> Callable[..., optax.GradientTransformation]:
        assert "learning_rate" in self.hyperparams
        return optax.inject_hyperparams(self.tx)(**self.hyperparams)

    def update(self, **kwargs):
        self.hyperparams.update(kwargs)
        return self


def convert_omegaconf_to_flat_dict(config: omegaconf.OmegaConf):
    return pd.json_normalize(omegaconf.OmegaConf.to_container(config), sep=".").to_dict(orient="records")[0]


def set_seed(seed=-1):
    logging.info(f"Setting seed to {seed} for reproducibility.")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    if seed != -1:
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    # logging.info(f"Setting seed to {torch.initial_seed()}")
