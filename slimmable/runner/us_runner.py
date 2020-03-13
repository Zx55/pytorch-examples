# -*- coding:utf-8  -*-

import logging
import time
import torch
import torch.nn as nn
from .utils import AverageMeter, accuracy, LabelSmoothCELoss, KL, \
    unique_width_sampler, calc_model_flops


class USNetRunner:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('global_logger')

        self.val_width_idx, num_width = self._get_val_width_idx()
        self.width_sampler = unique_width_sampler(num_width, self.config.training.num_sample)

    def train(self, train_loader, val_loader, model, optimizer, scheduler,
              start_iter, tb_logger):
        batch_time = AverageMeter(self.config.print_freq)
        data_time = AverageMeter(self.config.print_freq)
        # track stats of min switch, max switch and random switch
        losses = [AverageMeter(self.config.print_freq) for _ in range(3)]
        top1 = [AverageMeter(self.config.print_freq) for _ in range(3)]
        top5 = [AverageMeter(self.config.print_freq) for _ in range(3)]

        # switch to train mode
        model.train()

        # calculate model flops
        self.logger.info('init flops: {:.3f}'.format(calc_model_flops(model, self.config.model.input_size)))

        label_smooth = self.config.get('label_smooth', 0.0)
        if label_smooth > 0:
            criterion = LabelSmoothCELoss(label_smooth, 1000)
            self.logger.info('using label_smooth: {}'.format(label_smooth))
        else:
            criterion = nn.CrossEntropyLoss()
        distill_loss = KL(self.config.training.distillation.temperature)

        end = time.time()
        for _ in range(self.config.epoch):
            for i, (x, y) in enumerate(train_loader):
                x = x.cuda()
                y = y.cuda()

                cur_step = start_iter + i
                scheduler.step(cur_step)
                cur_lr = scheduler.get_lr()[0]

                # measure data loading time
                data_time.update(time.time() - end)

                # forward
                optimizer.zero_grad()
                max_width_pred = None
                for idx in range(self.config.training.num_sample):
                    top1_m, top5_m, loss_m = self._set_width(model, idx, top1, top5, losses)

                    output = model(x)
                    if self.config.training.distillation.enabled:
                        # max width model use ground truth
                        if idx == 0:
                            max_width_pred = output.detach()
                            cls_loss = criterion(output, y)
                        # other width use max model prediction
                        else:
                            cls_loss = self.config.training.distillation.loss_weight * distill_loss(output, max_width_pred)
                            if self.config.training.distillation.hard_label:
                                cls_loss += criterion(output, y)
                    else:
                        cls_loss = criterion(output, y)

                    acc1, acc5 = accuracy(output, y, top_k=(1, 5))
                    loss_m.update(cls_loss.item())
                    top1_m.update(acc1.item())
                    top5_m.update(acc5.item())

                    cls_loss.backward()

                optimizer.step()
                batch_time.update(time.time() - end)

                if cur_step % self.config.print_freq == 0:
                    tb_logger.add_scalar('lr', cur_lr, cur_step)
                    self.logger.info('-' * 80)
                    self.logger.info('Iter: [{0}/{1}]\t'
                                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                     'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                                     'LR {lr:.4f}'.format(
                                         cur_step, len(train_loader), batch_time=batch_time,
                                         data_time=data_time, lr=cur_lr))

                    titles = ['max_width', 'min_width', 'random_width']
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

                if cur_step >= self.config.start_val and cur_step % self.config.val_freq == 0:
                    val_loss, val_acc1, val_acc5 = self.validate(val_loader, model, x)

                    for j in range(len(val_loss)):
                        tb_logger.add_scalar('loss_val@{}'.format(self.config.training.val_width[j]), val_loss[j], cur_step)
                        tb_logger.add_scalar('acc1_val@{}'.format(self.config.training.val_width[j]), val_acc1[j], cur_step)
                        tb_logger.add_scalar('acc5_val@{}'.format(self.config.training.val_width[j]), val_acc5[j], cur_step)

                end = time.time()

    def validate(self, val_loader, model, sample):
        batch_time = AverageMeter(0)
        losses = [AverageMeter(0) for _ in range(len(self.val_width_idx))]
        top1 = [AverageMeter(0) for _ in range(len(self.val_width_idx))]
        top5 = [AverageMeter(0) for _ in range(len(self.val_width_idx))]
        final_loss, final_top1, final_top5 = [], [], []

        # switch to evaluate mode
        model.eval()

        criterion = nn.CrossEntropyLoss()
        end = time.time()

        with torch.no_grad():
            for i in range(len(self.val_width_idx)):
                self.logger.info('-' * 80)
                self.logger.info('Evaluating [{}/{}]@{}'.format(i + 1, len(self.val_width_idx),
                                                                self.config.training.val_width[i]))
                top1_m, top5_m, loss_m = self._set_width(model, i, top1, top5, losses)
                model.reset_bn_post_stats(list([sample]))

                for j, (x, y) in enumerate(val_loader):
                    x, y = x.cuda(), y.cuda()
                    num = x.size(0)

                    output = model(x)
                    loss = criterion(output, y)
                    acc1, acc5 = accuracy(output.data, y, top_k=(1, 5))

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

            model.train()
            return final_loss, final_top1, final_top5

    def _set_width(self, model, i, top1, top5, losses):
        if model.training:
            # max width
            if i == 0:
                model.uniform_set_width(-1)
                return top1[0], top5[0], losses[0]
            # min width
            elif i == 1:
                model.uniform_set_width(0)
                return top1[1], top5[1], losses[1]
            # (n - 2) random width
            else:
                model.uniform_set_width(next(self.width_sampler))
                return top1[2], top5[2], losses[2]
        else:
            # choose validation width index
            model.uniform_set_width(self.val_width_idx[i])
            return top1[i], top5[i], losses[i]

    def _get_val_width_idx(self):
        val_width = self.config.training.val_width
        min_width = self.config.training.min_width
        max_width = self.config.training.max_width
        offset = self.config.training.offset
        num_width = int((max_width - min_width) / offset + 1e-4) + 1
        all_width = [round(min_width + i * offset, 3) for i in range(num_width)]

        idx = list(map(lambda width: all_width.index(width), val_width))
        return idx, num_width
