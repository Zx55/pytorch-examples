# -*- coding:utf-8  -*-

import torch.nn as nn
from .us_ops import USModule, USConv2d, USBatchNorm2d
from .utils import make_divisible


class USInvertedResidual(nn.Module):
    def __init__(self, ch_in, ch_out, stride, expand_ratio):
        super(USInvertedResidual, self).__init__()

        assert stride in [1, 2]
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.stride = stride
        hidden_dim = int(ch_in * expand_ratio)

        layers = []
        if expand_ratio != 1:
            # point-wise
            layers.extend([
                USConv2d(ch_in, hidden_dim, 1, 1, 0, bias=False, expand_ratio=[1, expand_ratio]),
                USBatchNorm2d(hidden_dim, expand_ratio=expand_ratio),
                nn.ReLU6(inplace=True),
            ])
        layers.extend([
            # depth-wise
            USConv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim,
                     bias=False, expand_ratio=[expand_ratio, expand_ratio]),
            USBatchNorm2d(hidden_dim, expand_ratio=expand_ratio),
            nn.ReLU6(inplace=True),
            # point-wise-linear
            USConv2d(hidden_dim, ch_out, 1, 1, 0, bias=False, expand_ratio=[expand_ratio, 1]),
            USBatchNorm2d(ch_out),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.stride == 1 and self.ch_in == self.ch_out:
            return x + self.conv(x)
        else:
            return self.conv(x)


class USMobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, input_size=224, max_width=1.0):
        super(USMobileNetV2, self).__init__()

        self.last_channel = make_divisible(1280 * max_width) \
            if max_width > 1.0 else 1280

        ch_out = make_divisible(32 * max_width)
        self.preprocess = nn.Sequential(
            USConv2d(3, ch_out, 3, 2, 1, bias=False, us_switch=[False, True]),
            USBatchNorm2d(ch_out),
            nn.ReLU6(inplace=True)
        )
        self.blocks, blocks_out = self._make_blocks()
        self.tail = nn.Sequential(
            USConv2d(blocks_out, self.last_channel, 1, 1, 0, bias=False,
                     us_switch=[True, False]),
            USBatchNorm2d(self.last_channel),
            nn.ReLU6(inplace=True)
        )
        self.pool = nn.AvgPool2d(input_size // 32)
        self.classifier = nn.Linear(self.last_channel, num_classes)

        self._init_weights()

    def forward(self, x):
        out = self.preprocess(x)
        out = self.blocks(out)
        out = self.tail(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        return self.classifier(out)

    def uniform_set_width(self, width_idx):
        for m in self.modules():
            if isinstance(m, USModule):
                m.set_width(width_idx)

    def reset_bn_post_stats(self, sample):
        assert self.training is False
        assert len(sample) > 0

        bn_momentum = None
        for m in self.modules():
            if isinstance(m, US.USSyncBatchNorm2d):
                m.running_mean.fill_(0)
                m.running_var.fill_(1)
                if bn_momentum is None:
                    bn_momentum = m.momentum
                m.momentum = 1.0
                m.training = True

        self.forward(sample[0].cuda())

        for m in self.modules():
            if isinstance(m, US.USSyncBatchNorm2d):
                m.training = False
                m.momentum = bn_momentum

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

    @staticmethod
    def _make_blocks():
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
            ch_out = c
            for i in range(n):
                blocks.append(USInvertedResidual(ch_in, ch_out,
                                                 s if i == 0 else 1, t))
                ch_in = ch_out

        return nn.Sequential(*blocks), ch_out


def us_mobilenet_v2(num_classes=1000, input_size=224, max_width=1.0):
    return USMobileNetV2(num_classes, input_size, max_width)
