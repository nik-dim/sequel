from typing import List, Optional

import jax.numpy as jnp
from flax import linen as nn

from .base_backbone import BaseBackbone


class CNN(BaseBackbone):
    channels: List[int]
    linear_layers: Optional[List[int]]
    num_classes = 10
    multiplier = 1
    kernel_size = 3
    activation = "relu"
    use_maxpool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, task_ids: jnp.ndarray = None, train: bool = True) -> jnp.ndarray:
        kernel_size = (self.kernel_size, self.kernel_size)
        for c in self.channels:
            x = nn.Conv(features=c * self.multiplier, kernel_size=kernel_size)(x)
            x = nn.relu(x)
            if self.use_maxpool:
                x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # flatten convolutional features
        x = x.reshape((x.shape[0], -1))
        for feats in self.linear_layers:
            x = nn.Dense(features=feats)(x)
            x = nn.relu(x)

        x = nn.Dense(features=self.num_classes)(x)

        if self.multihead and task_ids is not None:
            x = self.select_output_head(x, task_ids)
        return x
