# -*- coding:utf-8  -*-

from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.datasets
import torchvision.transforms as transforms
import os


def get_cifar10_loader(config, last_iter):
    # Set up dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        config.cifar10_path, transform=transform_train, train=True, download=False)

    val_dataset = torchvision.datasets.CIFAR10(
        config.cifar10_path, transform=transform_val, train=False, download=False)

    # Set up data loader
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.workers,
        pin_memory=True, sampler=SubsetRandomSampler(range(0, config.num_train)))

    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.workers,
        pin_memory=True, sampler=SubsetRandomSampler(range(config.num_train, config.num_total)))

    return train_loader, val_loader
