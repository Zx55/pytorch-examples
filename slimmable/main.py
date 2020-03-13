# -*- coding:utf-8  -*-

import os
import random
import torch
from slimmable.utils import get_config, get_logger, get_optimizer, get_data_loader
from slimmable.models import us_mobilenet_v2
from slimmable.runner import USNetRunner


def main():
    config = get_config(os.path.join(os.path.dirname(__file__), 'config.yaml'))

    # init
    random_seed = config.random_seed
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    tb_logger, logger = get_logger(config.log_path)

    # load model
    model = us_mobilenet_v2(**config.model).cuda()
    optimizer, scheduler = get_optimizer(model, config)

    train_loader, val_loader = get_data_loader(config)
    USNetRunner(config).train(train_loader, val_loader, model, optimizer, scheduler, 0, tb_logger)

    logger.info('Done')


if __name__ == '__main__':
    main()
