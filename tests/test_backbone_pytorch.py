import sys
from pathlib import Path

# setting path
sys.path.append(Path().parent.absolute().as_posix())
print(sys.path)

import torch
from sequel.backbones.pytorch import MLP, CNN


def test_mlp():
    x = torch.rand(10, 1, 28, 28)

    model = MLP(width=2, n_hidden_layers=3)
    y = model(x)
    assert y.shape == (10, 10)

    model = MLP(width=2, n_hidden_layers=3, num_classes=123)
    y = model(x)
    assert y.shape == (10, 123)


def test_cnn():
    x = torch.rand(10, 1, 28, 28)

    model = CNN(channels=[10, 10])
    y = model(x)
    assert y.shape == (10, 10)

    model = CNN(channels=[10, 10], num_classes=123)
    y = model(x)
    assert y.shape == (10, 123)

    model = CNN(channels=[10, 10], num_classes=123, linear_layers=20)
    y = model(x)
    assert y.shape == (10, 123)
