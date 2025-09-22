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
import time	
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from updated_transformer.pruning_indices import calculate_scores,accumulated_scores_uncertainty,select_pruning_indices,expand_prune_indices,update_dropout_masks
from updated_transformer.pruning_masks import apply_linear_mask,enforce_all_masks,generate_prune_masks_transformer,generate_prune_masks_linear_layers
#added
from updated_transformer.dynamic_dropath import update_drop_path_rates
#added
import utils
import json
import itertools
import torch.distributed as dist

def _ddp_is_on():
    return dist.is_available() and dist.is_initialized()

@torch.no_grad()
def _ddp_avg_scores_(scores: dict):
    """
    In-place all-reduce (mean) of every tensor value in `scores`.
    Assumes same keys and shapes on all ranks.
    """
    if not _ddp_is_on():
        return scores
    ws = dist.get_world_size()
    for k, v in scores.items():
        dist.all_reduce(v, op=dist.ReduceOp.SUM)
        v.div_(ws)
    return scores

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

#added
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable,support_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,check:bool=False,
                    update_freq:int=1,update_batches:int =5, tracker =None, update_data_loader= None,
                    mode =None,scoring_type:str ="Conductance",same_batch = False,help_par:int =1,
                    noisy_dropout = False,min_dropout = 0.0,alt_attention_cond = False,mask_type = "sigmoid"
                    ,ypath = False,conductance_batch_size:int = 32,no_attn = True) -> dict:
   
    # TODO fix this for finetuning
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200
    tracker_normal = tracker[0] if (tracker is not None and len(tracker) >= 1) else None
    tracker_minmax = tracker[1] if (tracker is not None and len(tracker) >= 2) else None
    # Wrap one of them with the metric logger for training.
    logged_iter = metric_logger.log_every(data_loader, print_freq, header)


    #new_iter = iter(support_loader) if support_loader is not None else iter(data_loader)
    new_iter = iter(data_loader)    

    for batch_idx, (samples, targets) in enumerate(logged_iter):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        print("Batch idx:", batch_idx)
        with torch.amp.autocast('cuda'):
            if (check and (batch_idx % update_freq == 0)) or (tracker_normal is not None):
                # Get the next update_batches batches.
                if update_data_loader == None:
                    next_batches = []
                    if same_batch:
                        bs,bt = samples, targets
                    else:
                        bs, bt = next(new_iter)
                    # if mixup_fn is not None:
                    #     bs, bt = mixup_fn(bs, bt)

                    sample_chunks = bs.split(conductance_batch_size)
                    target_chunks = bt.split(conductance_batch_size)
                    nb = min(update_batches, len(sample_chunks))
                    next_batches = [(sample_chunks[i], target_chunks[i]) for i in range(nb)]

                else:
                    next_batches = []
                    for _ in range(update_batches):
                        # Get a random batch from the preloaded cached_subdataset.
                        sub_samples, sub_targets = get_random_batch(update_data_loader, batch_size=conductance_batch_size)  # Use desired sub batch size (e.g. 32)
                        # Move the subbatch to device.
                        sub_samples = sub_samples.to(device, non_blocking=True)
                        sub_targets = sub_targets.to(device, non_blocking=True)
                        next_batches.append((sub_samples, sub_targets))

                

                sm = False
                selected_layers = model.module.selected_layers if hasattr(model, 'module') else model.selected_layers
                if hasattr(model, 'module'):
                    print("Batch idx scores:", batch_idx)
                    print("Selected layers len:", len(selected_layers))
                    print("Next batches len:", len(next_batches))
                    print("Next batches 0 len:", len(next_batches[0][0]))
                    scores,_ = calculate_scores(model.module,
                                        next_batches, device, scoring_type=scoring_type, mode =mode,
                                        normalization=False, sm=sm, selected_layers=selected_layers,ypath = ypath)
                else:
                    print("Batch idx scores:", batch_idx)
                    print("Selected layers len:", len(selected_layers))
                    print("Next batches len:", len(next_batches))
                    print("Next batches 0 len:", len(next_batches[0]))
                    scores,_ = calculate_scores(model,
                                        next_batches, device, scoring_type=scoring_type, mode=mode,
                                        normalization=False, sm=sm, selected_layers=selected_layers,ypath = ypath)
                # if batch_idx % print_freq == 0:
                #     rank = dist.get_rank()
                #     for i,score in scores.items():
                #         if i<4:
                #             if i == 0:
                #                 print("Rank {} First layer scores before avg:{}".format(rank, score),flush=True,force=True)
                #             print("Rank {} Conductance Score shape layer {}: {}".format(rank, i, score.shape),flush=True,force=True)
                #             print("Rank {} Conductance Score mean layer {}: {}".format(rank, i, score.mean()),flush=True,force=True)
                #             print("Rank {} Conductance Score layer {}: {}".format(rank, i, score.std()),flush=True,force=True)

                _ddp_avg_scores_(scores)

                if tracker_normal is not None:
                    with torch.no_grad():
                        tracker_normal.update(scores)

                update_dropout_masks(model = model.module if hasattr(model, 'module') else model,
                                    scores=scores, drop_list=None,
                                    min_dropout=min_dropout, noisy_dropout=noisy_dropout,
                                stats=False, alt_attention_cond=alt_attention_cond,)
                
                if tracker_normal is not None:
                    drop_rates ={}
                    for i,drop in enumerate(model.module.drop_list if hasattr(model, 'module') else model.drop_list):
                        drop_rates[i] = drop.previous
                    with torch.no_grad():
                        tracker_minmax.update(drop_rates)
                if _ddp_is_on():
                    dist.barrier()
                    
            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)
            outputs = model(samples)
            loss = criterion(outputs, targets)
          


        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        

        optimizer.zero_grad()

        if help_par == 1:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

            loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order
            )
            torch.cuda.synchronize()
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
    torch.cuda.empty_cache()


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
        metric_logger.meters['loss'].update(loss.item(), n=batch_size)
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def prune_and_train(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable,data_loader_val :Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epochs: int, loss_scaler, max_norm: float = 0,lr_scheduler=None,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    update_freq:int=1,update_batches:int =5, update_data_loader= None,
                    output_dir: str = None,scoring_type:str ="Conductance",normalization:bool = True,transformer:bool = False,
                    uncertainty:bool = False,w_avg_rate : float = 0.05,pruning_rate: float = 0.2, 
                    pruning_type: str = "normalization",next_layer:bool = False,help_par:int =1,ypath_rates = None) -> dict:
   
    # TODO fix this for finetuning
    prune_indices = None
    prune_masks = None
    model.use_normal_dropout()
    cumulative_train_time = 0.0

    for epoch in range(epochs):
        epoch_start_time = time.time()
        acc_scores = None
        acc_means = None
        acc_uncertainty = None

        model.train()
        criterion.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = 2

        # Wrap one of them with the metric logger for training.
        logged_iter = metric_logger.log_every(data_loader, print_freq, header)

        for batch_idx, (samples, targets) in enumerate(logged_iter):

            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)
            #print('batch_idx:', batch_idx)
            with torch.amp.autocast('cuda'):
                if (batch_idx % update_freq == 0):
                    # Get the next update_batches batches.
                    if update_data_loader == None:
                        bs,bt = samples, targets
                        sample_chunks = bs.split(32)
                        target_chunks = bt.split(32)
                        nb = min(update_batches, len(sample_chunks))
                        next_batches = [(sample_chunks[i], target_chunks[i]) for i in range(nb)]

                    else:
                        next_batches = []
                        for _ in range(update_batches):
                            # Get a random batch from the preloaded cached_subdataset.
                            sub_samples, sub_targets = get_random_batch(update_data_loader, batch_size=32)  # Use desired sub batch size (e.g. 32)
                            # Move the subbatch to device.
                            sub_samples = sub_samples.to(device, non_blocking=True)
                            sub_targets = sub_targets.to(device, non_blocking=True)
                            next_batches.append((sub_samples, sub_targets))
        
                new_scores,new_means = calculate_scores(model,next_batches,device,scoring_type=scoring_type,transformer=transformer,
                                                        normalization= normalization,sm = True)
                if acc_means is None:
                    acc_means = new_means
                else:
                    for i,mean in enumerate(new_means):
                        acc_means [i] += mean

                acc_scores, acc_uncertainty = accumulated_scores_uncertainty(acc_scores,new_scores,w_avg_rate,uncertainty,acc_uncertainty)
                
                outputs = model(samples)
                loss = criterion(outputs, targets)
               


            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
            

            optimizer.zero_grad()
            is_second_order = False 
            if help_par == 1:
                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order
                )
                torch.cuda.synchronize()

            else:
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
            enforce_all_masks(model)

            if model_ema is not None:
                model_ema.update(model)

            metric_logger.update(loss=loss_value)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # gather the stats from all processes
        acc_means = [mean / len(data_loader) for mean in acc_means]
        if uncertainty:
            for i, uncert in acc_uncertainty.items():
                acc_scores[i] = acc_scores[i] * uncert
        if lr_scheduler is not None:
            lr_scheduler.step(epoch)

        epoch_time = time.time() - epoch_start_time
        cumulative_train_time += epoch_time

        # test_stats = evaluate(data_loader_val, model, device)
        # print(f"Before pruning: Accuracy of the network on the  test images: {test_stats['acc1']:.1f}%")
        # log_stats = {
        #     'epoch': epoch,
        #     'before_pruning': "True",
        #     'train_loss': metric_logger.loss.global_avg,
        #     'test_acc': test_stats.get('acc1', 0),
        #     'time': cumulative_train_time,
        #     'test_loss': test_stats.get('loss', 0),
        # }
        # if output_dir :
        #     with (output_dir / "log.txt").open("a") as f:
        #         f.write(json.dumps(log_stats) + "\n")
        prune_indices = select_pruning_indices(acc_scores,pruning_rate,pruning_type,prune_indices,acc_means)
        exp_prune_indices = expand_prune_indices(prune_indices,acc_scores)
        flat_list = [ prune_indices[i] for i in range(len(model.selected_layers)) ]

        if transformer:
            prune_masks = generate_prune_masks_transformer(model,flat_list,next_layer=next_layer)
        else:
            prune_masks = generate_prune_masks_linear_layers(model,flat_list,next_layer=next_layer)
        for layer, (wm, bm) in zip(model.selected_layers, prune_masks):
            apply_linear_mask(layer, wm, bm)

        
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Before pruning: Accuracy of the network on the  test images: {test_stats['acc1']:.1f}%")
        log_stats = {
            'epoch': epoch,
            'before_pruning': "False",
            'train_loss': metric_logger.loss.global_avg,
            'test_acc': test_stats.get('acc1', 0),
            'time': cumulative_train_time,
            'test_loss': test_stats.get('loss', 0),
        }
        if output_dir :
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)


    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


