from torch.utils.tensorboard import SummaryWriter

from sequel.utils.loggers.base_logger import Logger
from sequel.utils.utils import convert_omegaconf_to_flat_dict
import warnings


class TensorBoardLogger(Logger):
    def __init__(self, config):
        super().__init__()
        self.writer = SummaryWriter()

        flat_config = convert_omegaconf_to_flat_dict(config)

        # remove values that are not TensorBoard compatible, such as lists.
        flat_config = {k: v for k, v in flat_config.items() if isinstance(v, (bool, str, float, int))}
        self.writer.add_hparams(flat_config, metric_dict={})

    def log(self, item, step=None, epoch=None):
        self.writer.add_scalars(main_tag="", tag_scalar_dict=item, global_step=step)

    def log_figure(self, figure, name, step=None, prefix=None):
        self.writer.add_figure(tag="", figure=figure, global_step=step)

    def log_code(self):
        warnings.warn("Logging the entire source code is not supported for TensorBoardLogger.")

    def terminate(self):
        self.writer.close()
