from typing import Optional

import omegaconf
import torch
import torch.nn as nn

from .base_backbone import BaseBackbone


class MLP(BaseBackbone):
    def __init__(
        self,
        width: int,
        n_hidden_layers: int,
        dropout: Optional[float] = None,
        num_classes: int = 10,
        *args,
        **kwargs,
    ) -> None:
        """A Multi-Layer Peceptron of `n_hidden_layers`, each of which has `width` neurons. This class is used as the
        encoder for the SharedBottom architecture.

        Args:
            n_hidden_layers (int): Number of hidden layers
            width (int): Width of (all) hidden layers
            dropout (Optional[float], optional): If set, the model includes a dropout layer after
                every Linear with probability equal to the set value. Defaults to None.
        """
        super().__init__(*args, **kwargs)
        self.widths = width
        self.num_classes = num_classes
        self.n_hidden_layers = n_hidden_layers
        layers = [nn.Flatten()]
        for w in range(n_hidden_layers):
            layers.append(nn.LazyLinear(width))
            layers.append(nn.ReLU(inplace=True))
            if dropout:
                layers.append(nn.Dropout(p=dropout))

        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.LazyLinear(self.num_classes)

    def forward(self, x: torch.Tensor, task_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.encoder(x)
        x = self.classifier(x)
        if self.multihead:
            x = self.select_output_head(x, task_ids)
        return x

    @classmethod
    def from_config(cls, config: omegaconf.ListConfig) -> BaseBackbone:
        n_hidden_layers = config.backbone.n_hidden_layers
        width = config.backbone.width
        dropout = getattr(config.backbone, "dropout", None)
        num_classes = getattr(config.backbone, "num_classes", 10)

        return cls(
            widths=n_hidden_layers,
            width=width,
            dropout=dropout,
            num_classes=num_classes,
        )
