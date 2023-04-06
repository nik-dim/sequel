from typing import Optional
from .base_logger import Logger
import wandb
from omegaconf import OmegaConf
from sequel.utils.utils import get_original_root_dir


class WandbLogger(Logger):
    def __init__(self, config: Optional[OmegaConf] = None, disabled: bool = False):
        """Inits the Weights & Biases Logger. The class handles the initialization of the experiment, saving of source
        code, and metric tracking.

        Args:
            config (Optional[OmegaConf], optional): The configuration of the current experiment. Defaults to None.
            disabled (bool, optional): A utility boolean to quickly disable Weight&Biases logging. Useful when
                debugging. Defaults to False.
        """

        super().__init__()

        if config is None and not disabled:
            wandb.init()
        elif config is None and disabled:
            wandb.init(mode="disabled")
        else:
            if disabled:
                mode = "disabled"
            else:
                mode = getattr(config.wandb, "mode", "disabled")

            wandb.init(
                entity=config.wandb.entity,
                project=config.wandb.project,
                config=OmegaConf.to_container(config),
                tags=getattr(config.wandb, "tags", []),
                mode=mode,
                group=getattr(config.wandb, "group", None),
                name=getattr(config.wandb, "name", None),
                settings=wandb.Settings(_disable_stats=True),
            )

        self.log_code()

        # make sure that validation metrics are logged with epoch as the x-axis
        wandb.define_metric("epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("val/accuracy/*", step_metric="epoch")
        wandb.define_metric("val/loss/*", step_metric="epoch")

    def log(self, item, step=None, epoch=None):
        wandb.log(item)

    def log_code(self):
        wandb.run.log_code(get_original_root_dir())

    def log_parameters(self, config: dict):
        # The hyperparameters are logged when calling `wandb.init` via the config argument.
        pass

    def log_figure(self, figure, name, step=None, epoch=None):
        wandb.log({name: wandb.Image(figure)})

    def terminate(self):
        wandb.finish()
