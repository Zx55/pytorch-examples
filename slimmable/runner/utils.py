# -*- coding:utf-8  -*-

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothCELoss(nn.Module):
    def __init__(self, smooth_ratio, num_classes):
        super(LabelSmoothCELoss, self).__init__()

        self.smooth_ratio = smooth_ratio
        self.val = smooth_ratio / num_classes
        self.log_soft = nn.LogSoftmax(dim=1)

    def forward(self, x, label):
        one_hot = torch.zeros_like(x)
        one_hot.fill_(self.val)
        y = label.to(torch.long).view(-1, 1)
        one_hot.scatter_(1, y, 1 - self.smooth_ratio + self.val)

        loss = -torch.sum(self.log_soft(x) * one_hot.detach()) / x.size(0)
        return loss


def KL(temperature):
    def kl_loss(student_outputs, teacher_outputs):
        loss = nn.KLDivLoss(size_average=False, reduce=False)(
            F.log_softmax(student_outputs / temperature, dim=1),
            F.softmax(teacher_outputs.detach() / temperature, dim=1)) \
                * (temperature * temperature)
        return torch.mean(torch.sum(loss, dim=-1))
    return kl_loss


def accuracy(output, target, top_k=(1,)):
    """Computes the precision@k for the specified values of k"""
    max_k = max(top_k)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


def unique_width_sampler(num_width, num_sample):
    output = []
    sample = list(range(1, num_width - 1))

    while True:
        if len(output) == 0:
            random.shuffle(sample)
            output = sorted(sample[:(num_sample - 2)])
        yield output.pop()


def calc_model_flops(model, input_size, mul_add=False):
    hook_list = []
    module_flops = []

    def conv_hook(self, input, output):
        output_channels, output_height, output_width = output[0].size()
        bias_ops = 1 if self.bias is not None else 0
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.cur_in_ch / self.cur_group)
        flops = (kernel_ops * (2 if mul_add else 1) + bias_ops) * output_channels * output_height * output_width
        module_flops.append(flops)

    def linear_hook(self, input, output):
        weight_ops = self.weight.nelement() * (2 if mul_add else 1)
        bias_ops = self.bias.nelement()
        flops = weight_ops + bias_ops
        module_flops.append(flops)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hook_list.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hook_list.append(m.register_forward_hook(linear_hook))

    dummy_input = torch.rand(2, 3, input_size, input_size).cuda()
    model(dummy_input)

    for hook in hook_list:
        hook.remove()
    return round(sum(module_flops) / 1e6, 2)
