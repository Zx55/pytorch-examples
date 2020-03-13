# -*- coding:utf-8  -*-

import torch.nn as nn
import torch.nn.functional as F
from .us_ops import USConv2d, USBatchNorm2d, USModule


class USBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, down_sample=None):
        super(USBlock, self).__init__()

        self.residual = nn.Sequential(
            USConv2d(in_ch, out_ch, 3, stride, 1, bias=False),
            USBatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            USConv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            USBatchNorm2d(out_ch),
        )

        self.down_sample = down_sample

    def forward(self, x):
        out = self.residual(x)
        identity = x if self.down_sample is None else self.down_sample(x)
        out += identity
        return F.relu(out, inplace=True)


class USBottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, down_sample=None):
        super(USBottleneck, self).__init__()

        compress = out_ch // 4
        self.residual = nn.Sequential(
            USConv2d(in_ch, compress, 1, 1, bias=False),
            USBatchNorm2d(compress),
            nn.ReLU(inplace=True),
            USConv2d(compress, compress, 3, stride, 1, bias=False),
            USBatchNorm2d(compress),
            nn.ReLU(inplace=True),
            USConv2d(compress, out_ch, 1, 1, bias=False),
            USBatchNorm2d(out_ch)
        )

        self.down_sample = down_sample

    def forward(self, x):
        out = self.residual(x)
        identity = x if self.down_sample is None else self.down_sample(x)
        out += identity
        return F.relu(out, inplace=True)


class USResNet(nn.Module):
    def __init__(self, layers, block, num_classes=1000, input_size=224):
        super(USResNet, self).__init__()

        self.preprocess = nn.Sequential(
            USConv2d(3, 64, 3, 1, 1, bias=False, us_switch=[False, True]),
            USBatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_layers(block, 64, 64, layers[0])
        self.layer2 = self._make_layers(block, 64, 128, layers[1], stride=2)
        self.layer3 = self._make_layers(block, 128, 256, layers[2], stride=2)
        self.layer4 = self._make_layers(block, 256, 512, layers[3], stride=2)

        self.tail = nn.Sequential(
            USConv2d(512, 512, 1, 1, 0, bias=False, us_switch=[True, False]),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.preprocess(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.tail(out)
        out = out.view(out.size(0), -1)
        return self.classifier(out)

    def uniform_set_width(self, width):
        for m in self.modules():
            if isinstance(m, USModule):
                m.set_width(width)

    @staticmethod
    def _make_layers(block, in_ch, out_ch, blocks, stride=1):
        down_sample = None
        if stride != 1:
            down_sample = nn.Sequential(
                USConv2d(in_ch, out_ch, 1, stride),
                USBatchNorm2d(out_ch)
            )

        layers = [block(in_ch, out_ch, stride, down_sample)]
        for i in range(1, blocks):
            layers.append(block(out_ch, out_ch))

        return nn.Sequential(*layers)


class USResNet18(USResNet):
    def __init__(self, block=USBlock, num_classes=1000, input_size=224):
        super(USResNet18, self).__init__([2, 2, 2, 2], block, num_classes, input_size)


class USResNet34(USResNet):
    def __init__(self, block=USBottleneck, num_classes=1000, input_size=224):
        super(USResNet34, self).__init__([3, 4, 6, 3], block, num_classes, input_size)


def us_resnet18(num_classes=1000, input_size=224):
    return USResNet18(USBlock, num_classes, input_size)


def us_resnet34(num_classes=1000, input_size=224):
    return USResNet34(USBottleneck, num_classes, input_size)
