import warnings
from typing import List, Optional, Union

import omegaconf
import torch
import torch.nn as nn

from .base_backbone import BaseBackbone

ACTIVATION_MAP = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
}


def get_model_output_features(config: omegaconf.ListConfig, model: torch.nn.Module) -> int:
    """Calculates the output dimension of a model. This method is used to infer he number of out features of an
    encoder, which serve as `in_features` of the decoders (task-specific layers).

    Args:
        config (omegaconf.ListConfig): The hydra config for the current experiment.
        model (torch.nn.Module): The PyTorch model (the encoder of the Shared-Bottom architecture.)

    Returns:
        int: The number of out_features of the model.
    """
    c = getattr(config.benchmark, "channels", 1)
    h, w = getattr(config.benchmark, "dimensions")

    # generate a random sample.
    x = torch.rand(1, c, h, w)
    with torch.no_grad():
        x = model(x)
    assert x.dim() == 2
    return x.size(1)


class CNN(BaseBackbone):
    def __init__(
        self,
        channels: List[int],
        linear_layers: Optional[Union[List, int]] = None,
        num_classes: int = 10,
        multiplier: int = 1,
        kernel_size: int = 3,
        activation="relu",
        stride: int = 2,
        use_maxpool: bool = False,
    ) -> None:
        """Inits the CNN backbone.

        Args:
            channels (int): The number if in channels.
            linear_layers (Optional[Union[List, int]], optional): The linear layers' widths. These linear layers suceed
                the convolutional layers. If set to None, no linear layers are used except the output layer, whose
                width is defined by `num_classes`. Defaults to None.
            num_classes (int, optional): The number of output logits. Defaults to 10.
            multiplier (int, optional): Multiplies the number of channels for all layers,
                making the model wider. Defaults to 1.
            kernel_size (int, optional): The kernel size of the convolutions. Currenrly all convolutions
                have the share kernel size. Defaults to 3.
            activation (str, optional): The type of activation used. Defaults to "relu".
            stride (int, optional): The convolutional stride. Defaults to 2.
            use_maxpool (bool, optional): If set, the model uses maxpool layers. Defaults to False.

        Raises:
            ValueError: If stride is not equal to 1 or 2.
            ValueError: If maxpool and stride of 2 are used at the same time.
            ValueError: If `num_classes` is not a positive integer.
        """
        super().__init__()
        if not (stride == 1 or stride == 2):
            raise ValueError("Only strides of 1 or 2 are supported.")
        if int(stride == 2) + int(use_maxpool) != 1:
            raise ValueError("You cannot use a stride of 2 and maxpool concurrently!")
        if not isinstance(num_classes, int) and num_classes < 1:
            raise ValueError("The number of output classes must be positive integer.")

        warnings.warn("The CNN class does not include Batch Normalization.")
        self.num_classes = num_classes

        self.multiplier = multiplier
        self.kernel_size = kernel_size
        self.activation = activation

        if linear_layers is not None:
            self.linear_layers = linear_layers if isinstance(linear_layers, list) else [linear_layers]
        else:
            self.linear_layers = []

        # multiply number of channels
        self.channels = [c * multiplier for c in channels]

        # construct convolutional encoder layers
        layers = []
        for i, c in enumerate(self.channels):
            layers.append(nn.LazyConv2d(c, kernel_size=kernel_size, stride=stride, padding=1))
            layers.append(ACTIVATION_MAP[activation])
            if use_maxpool:
                layers.append(nn.MaxPool2d(kernel_size=2))

        layers.append(nn.Flatten())
        # construct fully-connected layers
        for feats in self.linear_layers:
            layers.append(nn.LazyLinear(feats))
            layers.append(ACTIVATION_MAP[activation])
        self.encoder = nn.Sequential(*layers)

        self.classifier = nn.LazyLinear(num_classes)

    def forward(self, x: torch.Tensor, task_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.encoder(x)
        x = self.classifier(x)
        if self.multihead:
            x = self.select_output_head(x, task_ids=task_ids)
        return x

    @classmethod
    def from_config(cls, config: omegaconf.ListConfig):
        linear_layers = config.backbone.linear_layers
        channels = config.backbone.channels
        multiplier = getattr(config.backbone, "multiplier", 1)
        kernel_size = getattr(config.backbone, "kernel_size", 3)
        activation = getattr(config.backbone, "activation", "relu")
        stride = getattr(config.backbone, "stride", 2)
        use_maxpool = getattr(config.backbone, "use_maxpool", False)

        return cls(
            linear_layers=linear_layers,
            channels=channels,
            multiplier=multiplier,
            kernel_size=kernel_size,
            activation=activation,
            stride=stride,
            use_maxpool=use_maxpool,
        )
