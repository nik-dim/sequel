import hydra
from omegaconf import DictConfig, open_dict
import _main_jax
import _main_pytorch


@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def my_app(config: DictConfig) -> None:
    with open_dict(config):
        config.wandb.group = config.mode
    if config.mode == "jax":
        _main_jax.my_app(config)
    elif config.mode == "pytorch":
        _main_pytorch.my_app(config)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    my_app()
