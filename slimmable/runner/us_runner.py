# -*- coding:utf-8  -*-

import logging
import os.path as path
import random
import time
import torch
import torch.nn as nn
from .utils import AverageMeter, accuracy, LabelSmoothCELoss, KL
from ..models.us_ops import USBatchNorm2d


class USNetRunner:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.logger = logging.getLogger('global_logger')

    def train(self, train_loader, val_loader, optimizer, scheduler, tb_logger):
        # meters
        batch_time = AverageMeter(self.config.print_freq)
        data_time = AverageMeter(self.config.print_freq)
        # track stats of min width, max width and random width
        losses = [AverageMeter(self.config.print_freq) for _ in range(3)]
        top1 = [AverageMeter(self.config.print_freq) for _ in range(3)]
        top5 = [AverageMeter(self.config.print_freq) for _ in range(3)]

        # loss_fn
        label_smooth = self.config.get('label_smooth', 0.0)
        if label_smooth > 0:
            criterion = LabelSmoothCELoss(label_smooth, 1000)
            self.logger.info('using label_smooth: {}'.format(label_smooth))
        else:
            criterion = nn.CrossEntropyLoss()
        distill_loss = KL(self.config.training.distillation.temperature)

        max_width, min_width = self.config.training.max_width, self.config.training.min_width
        cur_step = 0
        end = time.time()
        for e in range(self.config.training.epoch):
            data_time.update(time.time() - end)

            # train
            self.model.train()
            for batch_idx, (x, y) in enumerate(train_loader):
                scheduler.step(cur_step)
                cur_lr = scheduler.get_lr()[0]
                cur_step += 1

                x, y = x.cuda(), y.cuda()
                optimizer.zero_grad()

                sample_width = [max_width, min_width] + \
                               [random.uniform(min_width, max_width) for _ in range(self.config.training.num_sample - 2)]

                max_pred = None
                for width in sample_width:
                    # sandwich rule
                    top1_m, top5_m, loss_m = self._set_width(width, top1, top5, losses)

                    out = self.model(x)
                    if self.config.training.distillation.enabled:
                        if width == max_width:
                            max_pred = out.detach()
                            loss = criterion(out, y)
                        else:
                            loss = self.config.training.distillation.loss_weight * distill_loss(out, max_pred)
                            if self.config.training.distillation.hard_label:
                                loss += criterion(out, y)
                    else:
                        loss = criterion(out, y)

                    acc1, acc5 = accuracy(out, y, top_k=(1, 5))
                    loss_m.update(loss.item())
                    top1_m.update(acc1.item())
                    top5_m.update(acc5.item())

                    loss.backward()

                optimizer.step()
                batch_time.update(time.time() - end)

                if cur_step % self.config.print_freq == 0:
                    tb_logger.add_scalar('lr', cur_lr, cur_step)
                    self.logger.info('-' * 80)
                    self.logger.info('Epoch: [{0}/{1}]\tIter: [{2}/{3}]\t'
                                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                     'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                                     'LR {lr:.4f}'.format(
                                      e, self.config.training.epoch, batch_idx, len(train_loader),
                                      batch_time=batch_time, data_time=data_time, lr=cur_lr))

                    titles = ['min_width', 'max_width', 'random_width']
                    for idx in range(3):
                        tb_logger.add_scalar('loss_train@{}'.format(titles[idx]), losses[idx].avg, cur_step)
                        tb_logger.add_scalar('acc1_train@{}'.format(titles[idx]), top1[idx].avg, cur_step)
                        tb_logger.add_scalar('acc5_train@{}'.format(titles[idx]), top5[idx].avg, cur_step)
                        self.logger.info('{title}\t'
                                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                                         .format(title=titles[idx], loss=losses[idx],
                                                 top1=top1[idx], top5=top5[idx]))

                end = time.time()

            for loss_m, top1_m, top5_m in zip(losses, top1, top5):
                loss_m.reset()
                top1_m.reset()
                top5_m.reset()

            # validation
            val_loss, val_acc1, val_acc5 = self.validate(val_loader, self.config.training.val_width,
                                                         calibration=True, train_loader=train_loader)

            for i in range(len(val_loss)):
                tb_logger.add_scalar('loss_val@{}'.format(self.config.training.val_width[i]), val_loss[i], cur_step)
                tb_logger.add_scalar('acc1_val@{}'.format(self.config.training.val_width[i]), val_acc1[i], cur_step)
                tb_logger.add_scalar('acc5_val@{}'.format(self.config.training.val_width[i]), val_acc5[i], cur_step)

        self.save()

    def validate(self, val_loader, val_width, calibration=False, train_loader=None):
        batch_time = AverageMeter(0)
        losses = [AverageMeter(0) for _ in range(len(val_width))]
        top1 = [AverageMeter(0) for _ in range(len(val_width))]
        top5 = [AverageMeter(0) for _ in range(len(val_width))]
        final_loss, final_top1, final_top5 = [], [], []

        # switch to evaluate mode
        self.model.eval()

        criterion = nn.CrossEntropyLoss()
        end = time.time()

        with torch.no_grad():
            for idx, width in enumerate(val_width):
                self.logger.info('-' * 80)
                self.logger.info('Evaluating [{}/{}]@{}'.format(idx + 1, len(val_width), width))
                top1_m, top5_m, loss_m = self._set_width(width, top1, top5, losses, idx=idx)

                if calibration:
                    assert train_loader is not None
                    self.calibrate(train_loader)

                for j, (x, y) in enumerate(val_loader):
                    x, y = x.cuda(), y.cuda()
                    num = x.size(0)

                    out = self.model(x)
                    loss = criterion(out, y)
                    acc1, acc5 = accuracy(out.data, y, top_k=(1, 5))

                    loss_m.update(loss.item(), num)
                    top1_m.update(acc1.item(), num)
                    top5_m.update(acc5.item(), num)

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    if j % self.config.print_freq == 0:
                        self.logger.info('Test: [{0}/{1}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})'
                                         .format(j, len(val_loader), batch_time=batch_time))

                final_loss.append(loss_m.avg)
                final_top1.append(top1_m.avg)
                final_top5.append(top5_m.avg)

                self.logger.info('Prec@1 {:.3f}\tPrec@5 {:.3f}\tLoss {:.3f}\ttotal_num={}'
                                 .format(final_top1[-1], final_top5[-1], final_loss[-1], loss_m.count))

            return final_loss, final_top1, final_top5

    def infer(self, test_loader, train_loader):
        self.validate(test_loader, self.config.test_width, calibration=True, train_loader=train_loader)

    def calibrate(self, train_loader):
        self.model.eval()

        momentum_bk = None
        for m in self.model.modules():
            if isinstance(m, USBatchNorm2d):
                m.reset_running_stats()
                m.training = True
                if momentum_bk is None:
                    momentum_bk = m.momentum
                m.momentum = 1.0

        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(train_loader):
                if batch_idx == self.config.training.calibration_batches:
                    break

                x = x.cuda()
                self.model(x)

        for m in self.model.modules():
            if isinstance(m, USBatchNorm2d):
                m.momentum = momentum_bk
                m.training = False

    def save(self, optimizer=None, scheduler=None, epoch=None):
        name = '\\'.join(path.dirname(__file__).split('\\')[:-1])
        name += '\\checkpoints\\' + str(self.model.__class__.__name__) + '_'
        name = time.strftime(name + '%m%d_%H%M.pth')

        state = {'model': self.model.state_dict()}
        if optimizer is not None:
            state['optimizer'] = optimizer.state_dict()
        if scheduler is not None:
            state['scheduler'] = scheduler.state_dict()
        if epoch is not None:
            state['epoch'] = epoch

        torch.save(state, name)
        self.logger.info('model saved at {}'.format(name))

    def load(self, checkpoint):
        self.model.load_state_dict(checkpoint['model'])

    def _set_width(self, width, top1, top5, loss, idx=None):
        self.model.uniform_set_width(width)

        if self.model.training:
            if width == self.config.training.min_width:
                return top1[0], top5[0], loss[0]
            elif width == self.config.training.max_width:
                return top1[1], top5[1], loss[1]
            else:
                return top1[2], top5[2], loss[2]
        else:
            assert idx is not None
            return top1[idx], top5[idx], loss[idx]
