"""Models defined by Kernel Continual Learning."""

from functools import partial

import torch.nn as nn
import torch.nn.functional as F

from sequel.backbones.pytorch.resnet import ResNet, conv3x3


class KCLBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout=0.02):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
            )
        self.IC1 = nn.Sequential(nn.BatchNorm2d(planes), nn.Dropout(p=dropout))

        self.IC2 = nn.Sequential(nn.BatchNorm2d(planes), nn.Dropout(p=dropout))

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.IC1(out)

        out += self.shortcut(x)
        out = F.relu(out)
        out = self.IC2(out)
        return out


class ResNet18ThinKCL(ResNet):
    def __init__(self, num_classes=100, dropout=0.02, *args, **kwargs):
        block = partial(KCLBasicBlock, dropout=dropout)
        super().__init__(block, [2, 2, 2, 2], num_classes, *args, **kwargs)
