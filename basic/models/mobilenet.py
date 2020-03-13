# -*- coding:utf-8  -*-

import torch.nn as nn
from .net import Net


def make_divisible(x, divisor=8, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8.

    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor

    new_v = max(min_value, int(x + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * x:
        new_v += divisor

    return int(new_v)


class InvertedResidual(nn.Module):
    def __init__(self, ch_in, ch_out, stride, expand):
        super(InvertedResidual, self).__init__()

        assert stride in [1, 2]
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.stride = stride
        hidden_dim = int(ch_in * expand)

        layers = []
        if expand != 1:
            # point-wise
            layers.extend([
                nn.Conv2d(ch_in, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        layers.extend([
            # depth-wise
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim,
                      bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # point-wise-linear
            nn.Conv2d(hidden_dim, ch_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ch_out),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.stride == 1 and self.ch_in == self.ch_out:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(Net):
    def __init__(self, num_classes=1000, input_size=224,
                 multiplier=1.0, visual=False):
        super(MobileNetV2, self).__init__(visual)
        self.multiplier = multiplier

        self.last_channel = make_divisible(1280 * multiplier) \
            if multiplier > 1.0 else 1280

        self.preprocess = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        self.blocks, blocks_out = self._make_blocks()
        self.last_conv = nn.Sequential(
            nn.Conv2d(blocks_out, self.last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.last_channel),
            nn.ReLU6(inplace=True)
        )
        self.classifier = nn.Linear(5120, num_classes)

        self._init_weights()

    def forward(self, x):
        out = self.preprocess(x)
        out = self.blocks(out)
        out = self.last_conv(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_blocks(self):
        blocks_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        blocks = []
        ch_in, ch_out = 32, 0
        for t, c, n, s in blocks_setting:
            ch_out = make_divisible(c * self.multiplier) if t > 1 else c
            for i in range(n):
                blocks.append(InvertedResidual(ch_in, ch_out,
                                               s if i == 0 else 1, t))
                ch_in = ch_out

        return nn.Sequential(*blocks), ch_out


def mobilenet_v2(num_classes=1000, input_size=224, visual=False):
    return MobileNetV2(num_classes, input_size, visual=visual)
