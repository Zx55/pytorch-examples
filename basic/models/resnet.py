# -*- coding:utf-8  -*-

import torch.nn as nn
import torch.nn.functional as F

from .net import Net


def flatten(x):
    """
    Collapse image data from (C, H, W) to (C, H * W)
    """
    return x.view(x.size(0), -1)


class Block(nn.Module):
    """
    Basic residual block.
    """
    def __init__(self, in_channels, out_channels, stride=1, down_sample=None):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.down_sample = down_sample

    def forward(self, x):
        out = self.residual(x)
        identity = x if self.down_sample is None else self.down_sample(x)
        out += identity
        return F.relu(out, inplace=True)


class Bottleneck(nn.Module):
    """
    Bottleneck residual block.
    """
    def __init__(self, in_channels, out_channels, stride=1, down_sample=None):
        super().__init__()
        compress = out_channels // 4
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, compress, 1, stride=1),
            nn.BatchNorm2d(compress),
            nn.ReLU(inplace=True),
            nn.Conv2d(compress, compress, 3, stride, padding=1),
            nn.BatchNorm2d(compress),
            nn.ReLU(inplace=True),
            nn.Conv2d(compress, out_channels, 1, stride=1),
            nn.BatchNorm2d(out_channels)
        )
        self.down_sample = down_sample

    def forward(self, x):
        out = self.residual(x)
        identity = x if self.down_sample is None else self.down_sample(x)
        try:
            out += identity
        except RuntimeError:
            print(self.down_sample)
            print(out.size())
            print(x.size())
            print(identity.size())
        return F.relu(out, inplace=True)


class ResNet(Net):
    """
    Residual Network.
    """
    def __init__(self, layers, block, num_classes=1000, visual=False):
        super().__init__(visual)
        self.preprocess = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layers(block, 64, 64, layers[0])
        self.layer2 = self._make_layers(block, 64, 128, layers[1], stride=2)
        self.layer3 = self._make_layers(block, 128, 256, layers[2], stride=2)
        self.layer4 = self._make_layers(block, 256, 512, layers[3], stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.preprocess(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.pool(out)
        out = flatten(out)
        return self.fc(out)

    @staticmethod
    def _make_layers(block, in_channels, out_channels, blocks, stride=1):
        down_sample = None
        if stride != 1:
            down_sample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

        layers = [block(in_channels, out_channels, stride, down_sample)]
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)


def resnet18(num_classes=1000, visual=False):
    model = ResNet([2, 2, 2, 2], Block, num_classes, visual)
    model.name += '18'
    return model


def resnet34(num_classes=1000, visual=False):
    model = ResNet([3, 4, 6, 3], Bottleneck, num_classes, visual)
    model.name += '34'
    return model
