# -*- coding:utf-8  -*-

import os
import pickle

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms as T

from .prefetcher import PreFetcher


class CIFAR10(Dataset):
    """
    A dataset wrapper for CIFAR-10.
    Load one image and its label in DataLoader Object at a time.
    """
    def __init__(self, opt, train=True, transform=None):
        self.train = train

        # initialize data path
        self.meta_root = os.path.join(opt.root, 'batches.meta')
        if train:
            self.root = os.path.join(opt.root, 'data_batch')
        else:
            self.root = os.path.join(opt.root, 'test_batch')

        # initialize transform
        if transform is None:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=opt.mean, std=opt.std)
            ])
        else:
            self.transform = transform

        # load data
        self._load_cifar10()

    def __getitem__(self, index):
        img = self.imgs[index]
        return self.transform(img), self.labels[index]

    def __len__(self):
        return len(self.labels)

    def _load_cifar10(self):
        """
        Load CIFAR-10 training or test data
        :return: a numpy array of images and their labels
        """
        if self.train:
            self.imgs, self.labels = [], []

            for i in range(1, 6):
                path = self.root + '_{:d}'.format(i)
                x, y = self._load_batch(path)
                self.imgs.append(x)
                self.labels.append(y)

            self.imgs = np.concatenate(self.imgs)
            self.labels = np.concatenate(self.labels)

        else:
            self.imgs, self.labels = self._load_batch(self.root)

    def _load_meta(self):
        with open(self.meta_root, 'rb') as f:
            meta = pickle.load(f, encoding='latin1')
        self.classes = meta['label_name']
        self.idx_to_classes = {i: label for i, label in enumerate(self.classes)}
        self.classes_to_idx = {label: i for i, label in enumerate(self.classes)}

    @staticmethod
    def _load_batch(root):
        """
        Load a data batch of CIFAR-10.
        :param root: data path
        :return: a numpy array of image batch and their labels
        """
        with open(root, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            x, y = data['data'], data['labels']
            # N * H * W * C
            x = np.asarray(x, dtype=np.uint8)\
                .reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
            y = np.asarray(y, dtype=np.int64)

        return x, y


def get_train_loader(opt, dataset=None):
    if dataset is None:
        dataset = CIFAR10(opt, train=True)
    
    loader = DataLoader(
        dataset, opt.batch_size, drop_last=True, pin_memory=opt.pin_memory,
        sampler=SubsetRandomSampler(range(0, opt.num_train))
    )
    
    if opt.device.type == 'cuda':
        return PreFetcher(loader)
    return loader


def get_val_loader(opt, dataset=None):
    if dataset is None:
        dataset = CIFAR10(opt, train=True)
    
    loader = DataLoader(
        dataset, opt.batch_size, drop_last=True, pin_memory=opt.pin_memory,
        sampler=SubsetRandomSampler(range(opt.num_train, opt.num_tot))
    )
    
    if opt.device.type == 'cuda':
        return PreFetcher(loader)
    return loader


def get_test_loader(opt, dataset=None):
    if dataset is None:
        dataset = CIFAR10(opt, train=False)
        
    loader = DataLoader(
        dataset, opt.batch_size, drop_last=True, 
        pin_memory=opt.pin_memory, shuffle=True
    )
    
    if opt.device.type == 'cuda':
        return PreFetcher(loader)
    return loader
