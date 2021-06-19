# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Panoptic Forecasting licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import time, os
from collections import defaultdict
import random

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from . import train_utils
from panoptic_forecasting.utils import dist
from panoptic_forecasting.models.dist_wrapper import DistWrapper


class InfiniteDataloader():
    def __init__(self, dataset, batch_size, collate_fn, num_workers, num_steps,
                 weights=None, batch_sampler=None):
        self.dataset = dataset
        self.num_steps = num_steps
        if batch_sampler is not None:
            self.dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn,
                                         num_workers=num_workers)
        else:
            if weights is None:
                self.dataloader = DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, drop_last=True, collate_fn=collate_fn,
                                            num_workers=num_workers)
            else:
                print("USING PROVIDED SAMPLE WEIGHTS")
                print("WEIGHTS SHAPE: ", weights.shape)
                sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
                self.dataloader = DataLoader(dataset, batch_size=batch_size,
                                            sampler=sampler, drop_last=True, collate_fn=collate_fn,
                                            num_workers=num_workers)
        self.iter = self.dataloader.__iter__()

    def __iter__(self):
        self.count = 0
        return self

    def __len__(self):
        return self.num_steps

    def __next__(self):
        self.count += 1
        if self.count == self.num_steps:
            raise StopIteration
        else:
            try:
                data = next(self.iter)
            except StopIteration:
                self.iter = self.dataloader.__iter__()
                data = next(self.iter)
            return data

def train(model, datasets, params, writers):
    dist.init_distributed_mode(params)
    train_data = datasets['train']
    if 'val' in datasets:
        val_data = datasets['val']
        train_writer, val_writer = writers
    else:
        val_data = None
        train_writer = writers[0]
    no_gpu = params.get('no_gpu', False)
    batch_size = params['training'].get('batch_size', 1000)
    val_batch_size = params['training'].get('val_batch_size', batch_size)
    if val_batch_size is None:
        val_batch_size = batch_size
    accumulate_steps = params['training'].get('accumulate_steps', 1)
    num_epochs = params['training'].get('num_epochs', 100)
    val_interval = params['training'].get('val_interval', 1)
    val_start = params['training'].get('val_start', 0)
    clip_grad = params['training'].get('clip_grad', None)
    clip_grad_norm = params['training'].get('clip_grad_norm', None)
    verbose = params['training'].get('verbose', False)
    collate_fn = params.get('collate_fn', None)
    sample_weights = params.get('sample_weights', None)
    continue_training = params.get('continue_training', False)
    num_workers = params['training'].get('num_data_workers', 0)
    num_val_workers = params['training'].get('num_val_data_workers', num_workers)
    print("BATCH SIZE: ", batch_size)
    steps_per_epoch = params['training'].get('steps_per_epoch')

    model_without_ddp = model
    if params['distributed']:
        device = torch.device('cuda')
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(DistWrapper(model), device_ids=[params['gpu']])
        model_without_ddp = model.module
        sampler_train = DistributedSampler(train_data)
        if val_data is not None:
            sampler_val = DistributedSampler(val_data, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_data)
        sampler_val = torch.utils.data.SequentialSampler(val_data)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, batch_size, drop_last=True
    )
    if steps_per_epoch is not None:
        train_data_loader = InfiniteDataloader(train_data, batch_size,
                                               collate_fn, num_workers, steps_per_epoch*accumulate_steps,
                                               batch_sampler=batch_sampler_train,
                                               weights=sample_weights)
    else:
        train_data_loader = DataLoader(train_data,
                                       batch_sampler=batch_sampler_train,
                                       collate_fn=collate_fn, num_workers=num_workers)
    print("NUM BATCHES: ", len(train_data_loader))
    if val_data is not None:
        val_data_loader = DataLoader(val_data, batch_size=val_batch_size, sampler=sampler_val,
                                    collate_fn=collate_fn, num_workers=num_val_workers)

    # Build Optimizers
    lr = params['training']['lr']
    wd = params['training'].get('wd', 0.)
    mom = params['training'].get('mom', 0.)

    model_params = [param for param in model_without_ddp.parameters() if param.requires_grad]
    if params['training'].get('use_adamw', False):
        print("USING ADAMW")
        opt = torch.optim.AdamW(model_params, lr=lr, weight_decay=wd)
    if params['training'].get('use_adam', False):
        opt = torch.optim.Adam(model_params, lr=lr, weight_decay=wd)
    else:
        opt = torch.optim.SGD(model_params, lr=lr, weight_decay=wd, momentum=mom)

    working_dir = params['working_dir']
    best_path = os.path.join(working_dir, 'best_model')
    checkpoint_dir = os.path.join(working_dir, 'model_checkpoint')
    training_path = os.path.join(working_dir, 'training_checkpoint')
    training_scheduler = train_utils.build_scheduler(opt, params)
    if continue_training:
        print("RESUMING TRAINING")
        model_without_ddp.load(checkpoint_dir)
        train_params = torch.load(training_path)
        start_epoch = train_params['epoch']
        opt.load_state_dict(train_params['optimizer'])
        best_val_result = train_params['best_val_result']
        best_val_epoch = train_params['best_val_epoch']
        model.steps = train_params['step']
        print("STARTING EPOCH: ", start_epoch)
    else:
        start_epoch = 1
        best_val_epoch = -1
        best_val_result = 10000000
        model.steps = 0

    end = start = 0
    seed = dist.get_rank()*10000 + start_epoch
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if training_scheduler is not None:
        for _ in range(0, start_epoch):
            training_scheduler.step()

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    for epoch in range(start_epoch, num_epochs + 1):

        if params['distributed']:
            sampler_train.set_epoch(epoch)
        torch.cuda.empty_cache()
        print("EPOCH", epoch, (end - start))

        model.train_percent = epoch / num_epochs
        start = time.time()
        loss_counters = defaultdict(float)
        batch_count = 0
        if verbose:
            iterator = enumerate(train_data_loader)
        else:
            iterator = tqdm(enumerate(train_data_loader), total=len(train_data_loader))
        for batch_ind, batch in iterator:
            model.train()
            if not no_gpu:
                batch = train_utils.batch2gpu(batch)
            inputs = batch['inputs']
            labels = batch['labels']
            #with torch.autograd.detect_anomaly():
            if params['distributed']:
                loss_dict = model(inputs, labels)
            else:
                loss_dict = model.loss(inputs, labels)
            loss = loss_dict['loss']

            if loss.dim() == 0:
                batch_count += 1
            else:
                batch_count += loss.size(0)
            loss = loss.mean() / accumulate_steps
            loss.backward()
            loss_dict = dist.reduce_dict(loss_dict)


            for loss_name, loss_val in loss_dict.items():
                loss_counters[loss_name] += loss_val.sum().item()
            if verbose:
                print("\tBATCH %d OF %d: %f" % (batch_ind + 1, len(train_data_loader), loss.item()))
            if accumulate_steps == -1 or (batch_ind + 1) % accumulate_steps == 0:
                if verbose and accumulate_steps > 0:
                    print("\tUPDATING WEIGHTS")
                if clip_grad is not None:
                    nn.utils.clip_grad_value_(model.parameters(), clip_grad)
                elif clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                opt.step()
                model.steps += 1
                opt.zero_grad()
                if accumulate_steps > 0 and accumulate_steps > len(train_data_loader) - batch_ind - 1:
                    break

        if training_scheduler is not None:
            training_scheduler.step()

        if train_writer is not None:
            for loss_name, loss_val in loss_counters.items():
                avg_loss = loss_val / batch_count
                train_writer.add_scalar(loss_name, avg_loss, global_step=epoch)
        if ((epoch + 1) % val_interval != 0):
            end = time.time()
            continue
        epoch_train_loss = loss_counters['loss'] / batch_count
        if val_data is None:
           epoch_loss = epoch_train_loss
        else:
            model.eval()
            opt.zero_grad()
            if verbose:
                print("COMPUTING VAL LOSSES")
            loss_counters = defaultdict(float)
            batch_count = 0
            with torch.no_grad():
                if verbose:
                    iterator = enumerate(val_data_loader)
                else:
                    iterator = tqdm(enumerate(val_data_loader), total=len(val_data_loader))
                for batch_ind, batch in iterator:
                    if not no_gpu:
                        batch = train_utils.batch2gpu(batch)
                    inputs = batch['inputs']
                    labels = batch['labels']
                    if params['distributed']:
                        loss_dict = model(inputs, labels)
                        loss_dict = dist.reduce_dict(loss_dict)
                    else:
                        loss_dict = model.loss(inputs, labels)
                    loss = loss_dict['loss']
                    if loss.dim() == 0:
                        batch_count += 1
                    else:
                        batch_count += loss.size(0)
                    for loss_name, loss_val in loss_dict.items():
                        loss_counters[loss_name] += loss_val.sum().item()
                    if verbose:
                        print("\tVAL BATCH %d of %d: %f" % (batch_ind + 1, len(val_data_loader), loss.mean().item()))
            if val_writer is not None:
                for loss_name, loss_val in loss_counters.items():
                    avg_loss = loss_val / batch_count
                    val_writer.add_scalar(loss_name, avg_loss, global_step=epoch)
            epoch_loss = loss_counters['loss'] / batch_count
            epoch_val_loss = epoch_loss

        if epoch_loss < best_val_result:
            best_val_epoch = epoch
            best_val_result = epoch_loss
            print("BEST VAL RESULT. SAVING MODEL...")
            if dist.is_main_process():
                model_without_ddp.save(best_path)
        if dist.is_main_process():
            model_without_ddp.save(checkpoint_dir)
            torch.save({
                'epoch': epoch + 1,
                'optimizer': opt.state_dict(),
                'best_val_result': best_val_result,
                'best_val_epoch': best_val_epoch,
                'step': model.steps,
            }, training_path)
        print("EPOCH %d EVAL: " % epoch)
        print("\tCURRENT TRAIN LOSS: %f" % epoch_train_loss)
        if val_data is None:
            print("\tBEST TRAIN LOSS:    %f" % best_val_result)
            print("\tBEST TRAIN EPOCH:   %d" % best_val_epoch)
        else:
            print("\tCURRENT VAL LOSS: %f" % epoch_val_loss)
            print("\tBEST VAL LOSS:    %f" % best_val_result)
            print("\tBEST VAL EPOCH:   %d" % best_val_epoch)
        end = time.time()
        seed = dist.get_rank()*10000 + epoch + 1
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)