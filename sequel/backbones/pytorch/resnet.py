import torch
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d
from .base_backbone import BaseBackbone
from typing import Optional, Iterable

BN_MOMENTUM = 0.05


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False, momentum=BN_MOMENTUM)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False, momentum=BN_MOMENTUM)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, affine=False, track_running_stats=False, momentum=BN_MOMENTUM),
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNetEncoder(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf=20):
        super().__init__()
        self.in_planes = nf

        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=False, momentum=BN_MOMENTUM)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, inp: torch.Tensor):
        bsz = inp.size(0)
        out = relu(self.bn1(self.conv1(inp.view(bsz, 3, 32, 32))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        return out


class ResNet(BaseBackbone):
    """ResNet18 backbone with the number of features as an arguments. For `nf=20`, the ResNet has 1/3 of the features
    of the original. Code adapted from:
        1.  https://github.com/facebookresearch/GradientEpisodicMemory
        2.  https://worksheets.codalab.org/rest/bundles/0xaf60b5ed6a4a44c89d296aae9bc6a0f1/contents/blob/models.py
    """

    def __init__(self, block, num_blocks, num_classes, nf=20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = ResNetEncoder(block, num_blocks, num_classes, nf)
        self.classifier = nn.Linear(nf * 8 * block.expansion, num_classes)

    def forward(self, inp: torch.Tensor, head_ids: Optional[Iterable] = None):
        out = self.encoder(inp)
        out = self.classifier(out)

        if self.multihead:
            out = self.select_output_head(out, head_ids)
        return out


class ResNet18Thin(ResNet):
    def __init__(self, num_classes=100, *args, **kwargs):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes, *args, **kwargs)
