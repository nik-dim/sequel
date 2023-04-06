import os
from typing import Optional

from comet_ml import Experiment
import omegaconf

from sequel.utils.utils import convert_omegaconf_to_flat_dict, get_original_source_code_root_dir

from .base_logger import Logger


class CometLogger(Logger):
    """[Comet](https://www.comet.com/docs/v2/) Logger.

    Handles the logging for the Comet service. Inherits from Logger.
    """

    def __init__(self, config: omegaconf.OmegaConf, api_key: Optional[str] = None):
        """Inits the CometLogger class.

        Args:
            config (omegaconf.OmegaConf): The experiment config file. It is automatically logged to Comet.
            api_key (Optional[str], optional): The COMET api key. If None, the API is inferred via the environment
                variables. Defaults to None.

        Raises:
            KeyError: If the `api_key` is None and the `COMET_API_KEY` environment variable is not set.
        """
        super().__init__()

        self.config = config
        if api_key is None:
            if os.environ.get("COMET_API_KEY") is None:
                raise KeyError(
                    "The COMET_API_KEY has not been set up as an environment variable. In order to add the "
                    "COMET_API_KEY to the environment variables, run in your terminal: "
                    "export COMET_API_KEY='YOUR_API_TOKEN'."
                )
            else:
                api_key = os.environ.get("COMET_API_KEY")

        self.experiment = Experiment(
            api_key=api_key,
            project_name="rot-mnist-20",
            display_summary_level=0,
        )

        self.log_parameters(config)
        self.log_code()

    def log(self, item, step=None, epoch=None):
        self.experiment.log_metrics(item, step=step, epoch=epoch)

    def log_figure(self, figure, name, step=None, epoch=None):
        self.experiment.log_image(figure, name=name, step=step)

    def log_code(self):
        self.experiment.log_code(folder=get_original_source_code_root_dir())

    def log_parameters(self, config: dict):
        self.experiment.log_parameters(convert_omegaconf_to_flat_dict(config))

    def terminate(self):
        pass
