# -*- coding:utf-8  -*-

import torch.nn as nn
import torch.nn.functional as F
from .utils import make_divisible


class USModule:
    """
    Base class for universally slimmable layers.

    :param us_switch: two boolean indicating if input channel and output
                    channel is slimmable, e.g., [True, True]
    """
    def __init__(self, in_ch, out_ch, group, us_switch, expand_ratio):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.group = group
        self.us_switch = us_switch
        self.expand_ratio = expand_ratio

        # number of channel at current width
        self.cur_in_ch = in_ch
        self.cur_out_ch = out_ch
        self.cur_group = group

    def set_width(self, width):
        self._set_input_width(width)
        self._set_output_width(width)

    def _set_input_width(self, width, specific_ch=None):
        if not self.us_switch[0]:
            return

        if specific_ch is not None:
            self.cur_in_ch = specific_ch
        else:
            self.cur_in_ch = int(make_divisible(self.in_ch * width / self.expand_ratio[0]) * self.expand_ratio[0])

        if self.group != 1:
            self.cur_group = self.cur_in_ch

    def _set_output_width(self, width, specific_ch=None):
        if not self.us_switch[1]:
            return

        if specific_ch is not None:
            self.cur_out_ch = specific_ch
        else:
            self.cur_out_ch = int(make_divisible(self.out_ch * width / self.expand_ratio[1]) * self.expand_ratio[1])


class USConv2d(USModule, nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, us_switch=None, expand_ratio=None):
        us_switch = [True, True] if us_switch is None else us_switch
        expand_ratio = [1, 1] if expand_ratio is None else expand_ratio
        super(USConv2d, self).__init__(in_channels, out_channels, groups, us_switch, expand_ratio)
        super(USModule, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight[:self.cur_out_ch, :self.cur_in_ch, :, :]
        bias = None if self.bias is None else self.bias[:self.cur_out_ch]
        return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.cur_group)


class USLinear(USModule, nn.Linear):
    def __init__(self, input_features, output_features, bias=True, us_switch=None):
        us_switch = [True, True] if us_switch is None else us_switch
        super(USLinear, self).__init__(input_features, output_features, None, us_switch, [1, 1])
        super(USModule, self).__init__(input_features, output_features, bias=bias)

    def forward(self, x):
        weight = self.weight[:self.cur_out_ch, :self.cur_in_ch]
        bias = None if self.bias is None else self.bias[:self.cur_out_ch]
        return F.linear(x, weight, bias)


class USBatchNorm2d(USModule, nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, us_switch=None, expand_ratio=1):
        us_switch = [False, True] if us_switch is None else us_switch
        super(USBatchNorm2d, self).__init__(None, num_features, None, us_switch, [1, expand_ratio])
        super(USModule, self).__init__(num_features, eps, momentum, affine, track_running_stats=True)

    def forward(self, x):
        weight = self.weight[:self.cur_out_ch] if self.affine else self.weight
        bias = self.bias[:self.cur_out_ch] if self.affine else self.bias

        running_mean = self.running_mean[:self.cur_out_ch]
        running_var = self.running_var[:self.cur_out_ch]

        return F.batch_norm(x, running_mean, running_var, weight, bias, self.training, self.momentum, self.eps)
