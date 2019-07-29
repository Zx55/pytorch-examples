# -*- coding:utf-8  -*-

import torch


class PreFetcher:
    def __init__(self, loader):
        self.loader = loader
        
    def __iter__(self):
        return _PreFetcherIter(self.loader)

    def __len__(self):
        return len(self.loader)

            
class _PreFetcherIter:
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.device = torch.device('cuda:0')
        self._preload()
        
    def __iter__(self):
        return self
    
    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)

        x, y = self.next_x, self.next_y
        if x is None:
            raise StopIteration

        self._preload()
        return x, y

    def _preload(self):
        try:
            self.next_x, self.next_y = next(self.loader)
        except StopIteration:
            self.next_x, self.next_y = None, None
            return
        
        with torch.cuda.stream(self.stream):
            self.next_x = self.next_x.to(self.device, non_blocking=True)
            self.next_y = self.next_y.to(self.device, non_blocking=True)
