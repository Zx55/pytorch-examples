# -*- coding:utf-8  -*-

from tqdm import tqdm

from numpy import concatenate
import sklearn.metrics as M
from torch import no_grad


class Metric:
    def __init__(self, model, metrics, mode, loader, device):
        self.model = model
        self.loader = loader
        self.mode = mode
        self.metrics = metrics
        self.device = device
        self.result = {}

    @no_grad()
    def evaluate(self):
        self.model.eval()
        ys, preds = [], []
        ys_np, preds_np = None, None

        pbar = tqdm(total=len(self.loader), desc='{:6s} metric'.
                    format(self.mode))
        for x, y in self.loader:
            ys.append(y.cpu())
            x, y = x.to(self.device), y.to(self.device)

            scores = self.model(x)
            _, pred = scores.max(dim=1)
            preds.append(pred.cpu())

            ys_np = concatenate(ys)
            preds_np = concatenate(preds)

            postfix = {}
            if 'acc' in self.metrics:
                postfix['acc'] = M.accuracy_score(ys_np, preds_np)
            if 'precision' in self.metrics:
                postfix['precision'] = M.precision_score(
                    ys_np, preds_np, [i for i in range(10)], average='macro')
            if 'recall' in self.metrics:
                postfix['recall'] = M.recall_score(
                    ys_np, preds_np, [i for i in range(10)], average='macro')
            if 'f1-score' in self.metrics:
                postfix['f1-score'] = M.f1_score(
                    ys_np, preds_np, [i for i in range(10)], average='macro')

            pbar.update(1)
            pbar.set_postfix(postfix)

        pbar.close()

        if 'acc' in self.metrics:
            self.result['acc'] = M.accuracy_score(ys_np, preds_np)
        if 'precision' in self.metrics:
            self.result['precision'] = M.precision_score(
                ys_np, preds_np, [i for i in range(10)], average='macro')
        if 'recall' in self.metrics:
            self.result['recall'] = M.recall_score(
                ys_np, preds_np, [i for i in range(10)], average='macro')
        if 'f1-score' in self.metrics:
            self.result['f1-score'] = M.f1_score(
                ys_np, preds_np, [i for i in range(10)], average='macro')

        return self.result
