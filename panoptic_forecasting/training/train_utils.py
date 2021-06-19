# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Panoptic Forecasting licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import torch
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import os


def build_scheduler(opt, params):
    lr_scheduler_type = params['training'].get('lr_decay_type')
    if lr_scheduler_type == 'step':
        lr_decay_factor = params['training'].get('lr_decay_factor')
        lr_decay_steps = params['training'].get('lr_decay_steps')
        return torch.optim.lr_scheduler.StepLR(opt, lr_decay_steps, lr_decay_factor)
    elif lr_scheduler_type == 'poly':
        num_epochs = params['training']['num_epochs']
        fn = lambda epoch: 1 - epoch/num_epochs
        return torch.optim.lr_scheduler.MultiplicativeLR(opt, fn)
    else:
        return None


class build_writers:
    def __init__(self, working_dir, splits):
        self.writer_dir = os.path.join(working_dir, 'logs/')
        self.splits = splits

    def __enter__(self):
        self.writers = []
        for split in self.splits:
            writer_dir = os.path.join(self.writer_dir, split)
            writer = SummaryWriter(writer_dir)
            self.writers.append(writer)
        return self.writers

    def __exit__(self, type, value, traceback):
        for writer in self.writers:
            writer.close()


def batch2gpu(batch):
    def _item2gpu(item):
        if isinstance(item, dict) or isinstance(item, defaultdict):
            return {key: _item2gpu(val) for key, val in item.items()}
        elif isinstance(item, list):
            return [_item2gpu(val) for val in item]
        else:
            try:
                return item.cuda(non_blocking=True)
            except:
                return item
    result = {}
    for key, val in batch.items():
        if key == 'meta':
            result[key] = val
        else:
            result[key] = _item2gpu(val)
    return result