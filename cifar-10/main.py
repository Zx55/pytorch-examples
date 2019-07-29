# -*- coding:utf-8  -*-

import warnings

import torch.nn.functional as F
import torch.optim as optim

from data import *
from models import resnet34 as resnet
from option import opt


warnings.filterwarnings('ignore')

train_data = CIFAR10(opt, train=True)
test_data = CIFAR10(opt, train=False)

train_loader = get_train_loader(opt, train_data)
val_loader = get_val_loader(opt, train_data)
test_loader = get_test_loader(opt, test_data)

model = resnet(num_classes=10, visual=True)
loss_fn = F.cross_entropy
optimizer = optim.SGD(model.parameters(), opt.lr, opt.momentum,
                      nesterov=True, weight_decay=opt.reg)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, verbose=True, patience=1,
    factor=opt.lr_decay, threshold=opt.lr_decay_threshold)

model.fit(train_loader, loss_fn, optimizer, opt.epochs, opt.device,
          val=val_loader, scheduler=scheduler, metrics=['f1-score'])
model.evaluate(test_loader, opt.device)
