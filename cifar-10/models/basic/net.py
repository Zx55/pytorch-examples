# -*- coding:utf-8  -*-

from abc import ABCMeta
import time
from tqdm import tqdm

import torch
from torch.nn import Module
from tensorboardX import SummaryWriter

from models.metric import Metric


class Net(Module, metaclass=ABCMeta):
    """
    A wrapper for torch.nn.Module.
    """
    def __init__(self, visual=False):
        super().__init__()
        self.name = str(self.__class__.__name__)
        self.visual = visual
        self.visualizer = SummaryWriter(log_dir='visual')

    def __del__(self):
        self.visualizer.close()

    def load(self, checkpoint):
        """
        Load model.
        """
        self.load_state_dict(checkpoint['model'])

    def save(self, name=None, optimizer=None, scheduler=None,
             epoch=None, verbose=False):
        """
        Save model in the checkpoints, using 'model + time' as name by default.
        """
        if name is None:
            name = 'checkpoints/' + self.name + '_'
            name = time.strftime(name + '%m%d_%H%M.pth')

        state = {'model': self.state_dict()}
        if optimizer is not None:
            state['optimizer'] = optimizer.state_dict()
        if scheduler is not None:
            state['scheduler'] = scheduler.state_dict()
        if epoch is not None:
            state['epoch'] = epoch

        torch.save(state, name)
        if verbose:
            print('checkpoint \'' + name + '\' save complete')
        return name

    def fit(self, train, loss_fn, optimizer, epochs, device, val,
            scheduler=None, metrics=None, save=True):
        """
        Train a model on the training dataset.
        :param train: A DataLoader giving the training data.
        :param loss_fn: A Loss computing the loss between prediction and ground truth.
        :param optimizer: A Optimizer applying gradient descent on parameters.
        :param epochs: A scalar giving number of epochs to train.
        :param device: A Pytorch device specifying whether run on cpu or gpu.
        :param val: A DataLoader giving the validation data.
        :param scheduler: A Scheduler performing annealing learning rate.
        :param metrics: A list of string specifying metrics to be evaluated.
        :return:
        """
        print('train model on train dataset')
        name = 'checkpoints/' + self.name + '_'
        name = time.strftime(name + '%m%d_%H%M.pth')

        self.to(device)

        # default metrics
        if metrics is not None and 'acc' not in metrics:
            metrics = list(metrics) + ['acc']
        elif metrics is None:
            metrics = ['acc']
        train_metric = Metric(self, metrics, 'train', train, device)
        val_metric = Metric(self, metrics, 'val', val, device)

        for epoch in range(epochs):
            time.sleep(0.3)
            pbar = tqdm(total=len(train), desc='epoch {:3d}/{:3d}'.
                        format(epoch + 1, epochs))

            loss_avg = 0

            for idx, (x, y) in enumerate(train):
                self.train()
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()

                # gradient descent
                scores = self(x)
                loss = loss_fn(scores, y)
                loss.backward()
                optimizer.step()

                # update progress bar
                loss_avg = loss_avg * (idx / (idx + 1)) \
                    + loss.item() / (idx + 1)
                pbar.update(1)
                pbar.set_postfix({'loss': loss_avg})

                # visualize loss
                if self.visual:
                    global_step = epoch * len(train) + idx
                    self.visualizer.add_scalar('loss', loss.item(),
                                               global_step)

            pbar.close()
            time.sleep(0.3)

            # apply metrics
            train_metric.evaluate()
            time.sleep(0.3)
            val_metric.evaluate()
            time.sleep(0.3)

            # visualize metrics on train and validation data
            if self.visual:
                for key in train_metric.result.keys():
                    train_value = train_metric.result[key]
                    val_value = val_metric.result[key]
                    self.visualizer.add_scalars(key, {
                        'train': train_value,
                        'validation': val_value
                    }, epoch + 1)

            # learning rate annealing
            if scheduler is not None:
                scheduler.step(val_metric.result['acc'])

            # save checkpoint
            if save:
                self.save(name=name, optimizer=optimizer, scheduler=scheduler,
                          epoch=epoch, verbose=True)

            print()

        return metrics

    def evaluate(self, test, device, metrics=None):
        print('evaluate model on test dataset')
        self.to(device)
        self.eval()

        # initialize metric
        if metrics is not None and 'acc' not in metrics:
            metrics = list(metrics) + ['acc']
        elif metrics is None:
            metrics = ['acc']
        test_metrics = Metric(self, metrics, 'test', test, device)
        test_metrics.evaluate()
        time.sleep(0.3)
        print()

    @torch.no_grad()
    def infer(self, x, device, idx_to_classes=None):
        self.to(device)
        self.eval()

        x = x.to(device)
        scores = self(x)
        idx = scores.max(dim=1)

        if idx_to_classes is None:
            return idx
        return idx_to_classes[idx]

    def show_model(self, x, verbose=False):
        self.visualizer.add_graph(self, x, verbose=verbose)
