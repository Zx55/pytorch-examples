# -*- coding:utf-8  -*-

import os
import random
import torch
from slimmable.utils import get_config, get_logger, get_optimizer, get_scheduler, get_data_loader
from slimmable.models import us_mobilenet_v2, us_resnet34
from slimmable.runner import USNetRunner


def main():
    config = get_config(os.path.join(os.path.dirname(__file__), 'config.yaml'))

    # init
    random_seed = config.random_seed
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    tb_logger, logger = get_logger(config.log_path)

    # load model
    model = us_resnet34(num_classes=10, input_size=32).cuda()
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # load data
    train_loader, val_loader, test_loader = get_data_loader(config)
    runner = USNetRunner(config, model)

    # checkpoint = torch.load(r'./checkpoints/USMobileNetV2_0313_2011.pth')
    # runner.load(checkpoint)

    # train and calibrate
    runner.train(train_loader, val_loader, optimizer, scheduler, tb_logger)
    runner.infer(test_loader, train_loader)

    logger.info('Done')


if __name__ == '__main__':
    main()
