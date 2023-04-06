import copy

from sequel.algos.pytorch.pytorch_base_algo import PytorchBaseAlgorithm
import torch.nn.functional as F


class LFL(PytorchBaseAlgorithm):
    """Less-Forgetting Learning implementation in PyTorch.

    The equivalent JAX implementation is [`LFL in JAX`][sequel.algos.jax.lfl.LFL].

    References:
        [1] Jung, H., Ju, J., Jung, M. & Kim, J. Less-forgetful learning for domain expansion in deep neural
            networks. Proceedings of the AAAI Conference on Artificial Intelligence 32, (2018).
    """

    def __init__(self, lfl_lambda: float, *args, **kwargs):
        """Inits the LFL class.

        Args:
            lfl_lambda (float): the regularization coefficient.
        """
        super().__init__(*args, **kwargs)
        self.regularization_coefficient = lfl_lambda

    def __repr__(self) -> str:
        return f"LFL(regularization_coefficient={self.regularization_coefficient})"

    def on_after_training_task(self, *args, **kwargs):
        # freeze previous model
        # assert isinstance
        self.prev_backbone = copy.deepcopy(self.backbone)
        self.prev_backbone.eval()

        for p in self.prev_backbone.parameters():
            p.requires_grad = False

    def on_before_backward(self, *args, **kwargs):
        if self.task_counter > 1:
            self.prev_backbone.eval()
            self.backbone.eval()

            features = self.backbone.encoder(self.x)
            prev_features = self.prev_backbone.encoder(self.x)
            self.loss += self.regularization_coefficient * F.mse_loss(features, prev_features)
