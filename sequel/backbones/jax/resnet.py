from functools import partial
from jax_resnet.resnet import ResNet, STAGE_SIZES, ResNetStem, ResNetBlock
import jax.numpy as jnp
import jax
import flax.linen as nn
from .base_backbone import BaseBackbone


n = 20
hidden_sizes = (n, 2 * n, 4 * n, 8 * n)


class ResNet18Thin(BaseBackbone):
    num_classes: int = 100
    # parent attributes
    classes_per_task: int = None
    multihead: bool = False
    hidden_sizes = hidden_sizes

    @nn.compact
    def __call__(self, x: jnp.ndarray, task_ids: jnp.ndarray = None, training: bool = True) -> jnp.ndarray:
        x = ResNet(
            stage_sizes=STAGE_SIZES[18],
            stem_cls=ResNetStem,
            block_cls=ResNetBlock,
            hidden_sizes=hidden_sizes,
            n_classes=self.num_classes,
        )(x)
        if self.multihead and task_ids is not None:
            x = self.select_output_head(x, task_ids)
        return x
