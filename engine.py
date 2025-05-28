# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable, Optional
import torch.nn.functional as F

import torch
import numpy as np

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils

import itertools


def get_random_batch(cached_data, batch_size):
    """
    Randomly sample batch_size items from the cached subdataset.
    
    Args:
      cached_data: A list containing all (data, target) tuples.
      batch_size: The desired batch size.
      
    Returns:
      A tuple (images, targets), where images is a tensor and targets is a tensor.
    """
    indices = np.random.choice(len(cached_data), size=batch_size, replace=False)
    batch = [cached_data[i] for i in indices]
    # Assume each item in cached_data is a tuple: (image, target)
    images, targets = zip(*batch)
    # Stack images. (Ensure that each image is already a tensor.)
    images = torch.stack(images)
    targets = torch.tensor(targets)
    return images, targets

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,check:bool=False,
                    update_freq:int=1,update_batches:int =5, stats: bool = False, update_data_loader= None,
                    output_dir: str = None,scoring_type:str ="Conductance",same_batch = False,help_par:int =1) -> dict:
   
    # TODO fix this for finetuning
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200

    # Wrap one of them with the metric logger for training.
    logged_iter = metric_logger.log_every(data_loader, print_freq, header)
    #new_iter = iter(data_loader)

    # if check and (not same_batch) and (update_data_loader == None):
        # Create a new iterator for the data loader.
    new_iter = iter(data_loader)
    

    # print('check:', check)
    for batch_idx, (samples, targets) in enumerate(logged_iter):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        #print('batch_idx:', batch_idx)
        with torch.amp.autocast('cuda'):
            if check and (batch_idx % update_freq == 0):
                # Get the next update_batches batches.
                if update_data_loader == None:
                    helper = data_loader.batch_size//32
                    next_batches = []
                    if same_batch:
                        big_batches = [(samples.clone(), targets.clone())]
                    else:
                        big_batches = list(itertools.islice(new_iter, math.ceil(update_batches/helper)))
                    for i in range(update_batches):
                        b = i//helper
                        ig = i%helper
                        full_samples, full_targets = big_batches[b]
                        sub_samples = full_samples[32*ig:32*(ig+1)]
                        sub_targets = full_targets[32*ig:32*(ig+1)]
                        next_batches.append((sub_samples.clone(), sub_targets.clone()))
                    
                    #next_batches = list(itertools.islice(new_iter, update_batches))
                else:
                    next_batches = []
                    for _ in range(update_batches):
                        # Get a random batch from the preloaded cached_subdataset.
                        sub_samples, sub_targets = get_random_batch(update_data_loader, batch_size=32)  # Use desired sub batch size (e.g. 32)
                        # Move the subbatch to device.
                        sub_samples = sub_samples.to(device, non_blocking=True)
                        sub_targets = sub_targets.to(device, non_blocking=True)
                        next_batches.append((sub_samples, sub_targets))
                # Now, get the next "update_batches" batches from the peek iterator.
                #model.calculate_scores(next_batches,device,stats=stats)
                model.calculate_scores(next_batches,device,stats=stats,scoring_type=scoring_type)



            outputs = model(samples)
            loss = criterion(outputs, targets)
            if stats and batch_idx % 350 == 0:
                model.plot_current_stats(epoch+1,batch_idx, output_dir / f'plots/epoch_{epoch+1}')

                
            


        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        

        optimizer.zero_grad()
        is_second_order = False 
        if help_par == 1:
            loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order
        )
        else:
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)


    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    print_freq = 30

    for images, target in metric_logger.log_every(data_loader, print_freq, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.amp.autocast('cuda'):
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
