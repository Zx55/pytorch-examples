# -*- coding:utf-8  -*-

import warnings

import torch.nn.functional as F
import torch.optim as optim

from basic.data import get_cifar10_loader
from basic.models import mobilenet_v2
# from basic.models import resnet18 as resnet
from basic.option import opt

train_loader, val_loader, test_loader = get_cifar10_loader(opt, -1)


def main():
    warnings.filterwarnings('ignore')

    # model = resnet(num_classes=10, visual=True)
    model = mobilenet_v2(num_classes=10, input_size=32)
    loss_fn = F.cross_entropy
    optimizer = optim.SGD(model.parameters(), opt.lr, opt.momentum,
                          nesterov=True, weight_decay=opt.reg)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, verbose=True, patience=1,
        factor=opt.lr_decay, threshold=opt.lr_decay_threshold)

    model.fit(train_loader, loss_fn, optimizer, opt.epochs, opt.device,
              val=val_loader, scheduler=scheduler, metrics=['f1-score'],
              save=False)
    model.evaluate(test_loader, opt.device)


if __name__ == "__main__":
    main()
