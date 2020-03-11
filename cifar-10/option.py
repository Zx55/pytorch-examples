# -*- coding:utf-8  -*-

from torch import device
from torch.cuda import is_available


class Option:
    """
    All options for this program, including hyperparameters and other runtime settings.
    """
    def __init__(self):
        # hyperparameters
        # initial learning rate
        self.lr = 0.005
        self.momentum = 0.9
        # learning rate decay
        self.lr_decay = 0.1
        self.lr_decay_threshold = 1e-5
        # weight decay
        self.reg = 1e-2

        # runtime settings
        # the root of dataset
        self.root = './data/raw/cifar-10-batches-py/'
        self.num_train = 45000
        self.num_tot = 50000
        # statistics of cifar-10
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2023, 0.1994, 0.2010)

        # which model to use
        self.model = 'ResNet18'

        self.epochs = 20
        self.batch_size = 64

        self.print_freq = 200
        # number of processes used to load data
        self.num_workers = 4

        self.use_gpu = True
        self._upgrade_device()

        # visualization option
        self.visual = True

    def __setattr__(self, key, value):
        self.__dict__[key] = value
        if key == 'use_gpu':
            self._upgrade_device()

    def _upgrade_device(self):
        self.device = device('cuda:0' if self.use_gpu and is_available() else 'cpu')
        if self.device.type == 'cuda':
            self.pin_memory = True
        else:
            self.pin_memory = False


opt = Option()
