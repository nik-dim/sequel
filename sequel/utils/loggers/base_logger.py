import abc


class Logger(abc.ABC):
    """The base class of the Logger Module.

    ALl loging services, such as TensorBoard of Weights and Biases, implement their own logger modules which are
    children of this class. This class shows the API of all loggers. The logger module is invoked by the [`Algorithm
    class`][sequel.algos.base_algo.BaseAlgorithm] via the homonym methods.
    """

    def __init__(self):
        """Inits the base class for the Loggers module."""
        pass

    def log(self, item, step=None, epoch=None):
        raise NotImplementedError

    def log_parameters(self, config: dict):
        raise NotImplementedError

    def log_code(self):
        raise NotImplementedError

    def log_figure(self, figure, name, step=None, epoch=None):
        raise NotImplementedError

    def terminate(self):
        raise NotImplementedError

    def log_all_results(self):
        pass
