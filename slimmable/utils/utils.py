# -*- coding:utf-8  -*-

from easydict import EasyDict
import logging
from tensorboardX import SummaryWriter
import torch
import yaml
from .scheduler import CosineLRScheduler, StepLRScheduler
from .data import get_cifar10_loader


def get_data_loader(config, last_iter=-1):
    if config.dataset.type == 'CIFAR10':
        return get_cifar10_loader(config.dataset.kwargs, last_iter)
    else:
        raise KeyError('invalid dataset type')


def get_optimizer(model, config):
    model_params = []
    for params in model.parameters():
        ps = list(params.size())
        if len(ps) == 4 and ps[1] != 1:
            weight_decay = config.optimizer.weight_decay
        elif len(ps) == 2:
            weight_decay = config.optimizer.weight_decay
        else:
            weight_decay = 0
        item = {'params': params, 'weight_decay': weight_decay,
                'lr': config.optimizer.base_lr, 'momentum': config.optimizer.momentum,
                'nesterov': config.optimizer.nesterov}
        model_params.append(item)
    return torch.optim.SGD(model_params)


def get_scheduler(optimizer, config, last_iter=-1):
    if config.scheduler.type == 'COSINE':
        scheduler = CosineLRScheduler(
            optimizer, config.scheduler.max_iter, config.scheduler.min_lr, config.scheduler.base_lr,
            config.scheduler.warmup_lr, config.scheduler.warmup_steps, last_iter=last_iter)
    elif config.scheduler.type == 'STEP':
        scheduler = StepLRScheduler(
            optimizer, config.scheduler.lr_steps, config.scheduler.lr_mults, config.scheduler.base_lr,
            config.scheduler.warmup_lr, config.scheduler.warmup_steps, last_iter=last_iter)
    else:
        raise KeyError('invalid lr scheduler type')
    return scheduler


def get_config(config_path):
    """load experiment config.
    Args:
        config_path: (args.confg)

    Returns:
        config: EasyDict(config)

    Details:
        config.save_path: dirname(config_path)
    """
    with open(config_path) as f:
        config = yaml.load(f)
    config = EasyDict(config)
    return config


def get_logger(path, name='global_logger'):
    tb_logger = SummaryWriter(path + '/events')

    logger = logging.getLogger(name)
    formatter = logging.Formatter('[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
    fh = logging.FileHandler(path + '/log.txt')
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    return tb_logger, logger
