# -*- coding:utf-8  -*-


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