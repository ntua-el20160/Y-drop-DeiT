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

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils

import itertools


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,check:bool=False,
                    update_freq:int=1,update_batches:int =5, stats: bool = False) -> dict:
    # TODO fix this for finetuning
    model.train()
    criterion.train()
    #model.base_dropout()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200
    # Wrap one of them with the metric logger for training.
    logged_iter = metric_logger.log_every(data_loader, print_freq, header)
    new_iter = iter(data_loader)
    for batch_idx, (samples, targets) in enumerate(logged_iter):
    #for batch_idx, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        with torch.amp.autocast('cuda'):
            if check and (batch_idx % update_freq == 0):
                # Get the next update_batches batches.
                next_batches = list(itertools.islice(new_iter, update_batches))
                # Now, get the next "update_batches" batches from the peek iterator.
                model.calculate_scores(next_batches,device,stats=stats)

            #model.check_dead_neurons(samples)

            outputs = model(samples)
            loss = criterion(outputs, targets)


        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        

        optimizer.zero_grad()

        # loss_scaler(loss, optimizer, clip_grad=max_norm,
        #             parameters=model.parameters(), create_graph=is_second_order)
        fc1_weight_norm_before = model.fc1.weight.norm().item()
        #print(f"[Batch {batch_idx}] fc1 weight norm before update: {fc1_weight_norm_before:.4f}")

        optimizer.zero_grad()
        loss.backward()
        # if model.fc1.weight.grad is not None:
        #     fc1_grad_norm = model.fc1.weight.grad.norm().item()
        #     print(f"[Batch {batch_idx}] fc1 grad norm: {fc1_grad_norm:.4f}")
        # else:
        #     print(f"[Batch {batch_idx}] fc1 grad is None")
        
        optimizer.step()
        fc1_weight_norm_after = model.fc1.weight.norm().item()
        #print(f"[Batch {batch_idx}] fc1 weight norm after update: {fc1_weight_norm_after:.4f}")
        torch.cuda.synchronize()

        

             
        if batch_idx == 1500:
            baseline = torch.zeros_like(samples)
            baseline_act = model(baseline)
            real_act = model(samples)
            print("Baseline activation fc1:", baseline_act.mean().item())
            print("Real activation fc1:", real_act.mean().item())
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

    for images, target in metric_logger.log_every(data_loader, 10, header):
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
