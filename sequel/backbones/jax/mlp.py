from typing import Optional

import jax.numpy as jnp
from flax import linen as nn

from .base_backbone import BaseBackbone


class MLPEncoder(nn.Module):
    width: int = 256
    n_hidden_layers: int = 2
    dropout: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # flatten
        x = x.reshape((x.shape[0], -1))
        for feats in range(self.n_hidden_layers):
            x = nn.Dense(features=self.width)(x)
            x = nn.relu(x)
            if self.dropout:
                x = nn.Dropout(self.dropout, deterministic=not training)(x)

        return x


class MLP(BaseBackbone):
    width: int = 256
    n_hidden_layers: int = 2
    dropout: Optional[float] = None
    num_classes: int = 10

    # parent attributes
    classes_per_task: int = None
    multihead: bool = False

    def setup(self) -> None:
        self.encoder = MLPEncoder(width=self.width, n_hidden_layers=self.n_hidden_layers, dropout=self.dropout)
        self.classifier = nn.Dense(features=self.num_classes)

    def __call__(self, x: jnp.ndarray, task_ids: jnp.ndarray = None, training: bool = True) -> jnp.ndarray:
        x = self.encoder(x, training=training)
        x = self.classifier(x)

        if self.multihead and task_ids is not None:
            x = self.select_output_head(x, task_ids)
        return x
