# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import random
from pathlib import Path
from updated_transformer.dynamic_dropath import  DropPath
import copy as _copy
import os
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
#from augment import new_data_aug_generator
import torch.nn as nn
import torch.nn.functional as F
from updated_transformer.dynamic_dropout import MyDropout

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets2 import build_dataset, create_subdataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss
from samplers import RASampler
#from augment import new_data_aug_generator
import models
import utils
#import models_v2
from stats_logging import StreamingConductanceEpochTracker,build_reports
class PerSampleSoftTargetCE(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # targets: [B] (long) or [B, C] (float)
        if targets.ndim == 1 or targets.dtype == torch.long:
            targets = F.one_hot(targets, num_classes=logits.size(-1)).to(logits.dtype)
        else:
            if targets.size(-1) != logits.size(-1):
                raise RuntimeError(
                    f"targets.size(-1) = {targets.size(-1)} "
                    f"!= logits.size(-1) = {logits.size(-1)}"
                )
            targets = targets.to(logits.dtype)
        return (-targets * F.log_softmax(logits, dim=-1)).sum(dim=-1)  # [B]
class LabelSmoothingCrossEntropyNoRed(nn.Module):
    """Per-sample label-smoothed cross-entropy (no reduction)."""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = float(smoothing)
        self.confidence = 1.0 - self.smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        x:      [B, C] logits
        target: [B]    class indices (LongTensor)
        returns: [B]   per-sample losses
        """
        logprobs = F.log_softmax(x, dim=-1)                  # [B, C]
        nll_loss = -logprobs.gather(1, target.unsqueeze(1)).squeeze(1)  # [B]
        smooth_loss = -logprobs.mean(dim=-1)                 # [B]
        return self.confidence * nll_loss + self.smoothing * smooth_loss
def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='deit_tiny_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop_rate', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')
    
    
    parser.add_argument('--experiment_name', default='simpletransformer', type=str, help='experiment name')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    # parser.add_argument('--train-mode', action='store_true')
    # parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    # parser.set_defaults(train_mode=True)


    #parser.add_argument('--ThreeAugment', action='store_true') #3augment
    
    # parser.add_argument('--src', action='store_true') #simple random crop
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

# # Distillation parameters
    # parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
    #                     help='Name of teacher model to train (default: "regnety_160"')
    # parser.add_argument('--teacher-path', type=str, default='')
    # parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    # parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    # parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    
    # * Cosub params
    parser.add_argument('--cosub', action='store_true') 
    
    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--attn-only', action='store_true') 
    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01_101/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR10','CIFAR100', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    #parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    #parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument("--dist-eval",
    action="store_true",
    default=False,
    help="Use DistributedSampler for validation",
    )
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # --- Custom Dropout Hyperparameters ---
    parser.add_argument('--ydrop', action='store_true', default=True,
                    help='Enable Y-Drop (MyDropout) by default')
    parser.add_argument('--no-ydrop', dest='ydrop', action='store_false',
                    help='Disable Y-Drop (MyDropout)')
    #added
    parser.add_argument('--ypath', action='store_true', default=False,
                    help='Enable Y-Path (MyPath)')
    #added

    parser.add_argument('--elasticity', type=float, default=0.01,
                        help='Elasticity factor for custom dropout')

    parser.add_argument('--annealing_factor', type=float, default=5,
                        help='Annealing factor for custom dropout')
    parser.add_argument('--n_steps',type=int,default = 5,
                         help ='intermediate steps for conductance calculation')
    parser.add_argument('--update_batches',type=int,default = 1,
                         help ='intermediate steps for conductance calculation')
    parser.add_argument('--update_freq',type=int,default = 1,
                            help ='intermediate steps for conductance calculation')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Number of epochs with no improvement in eval loss before early stopping')
    parser.add_argument('--plot_freq', default=5, type=int, help='plot frequency')
    parser.add_argument('--scaler', default=1.0, type=float, help='Loss scaler for mixed precision training')
    parser.add_argument('--mask_type', default='sigmoid', type=str, help='Type of mask for dropout')
    
    parser.add_argument('--sub_dataset', default ='none',choices=['none','stratified', 'random'] ,type=str, help='Sub dataset to use for training')
    parser.add_argument('--sub_factor', default=10, type=int, help='Sub dataset factor')
    parser.add_argument('--update_scaling',choices=['no','increasing', 'decreasing'], default='no', type =str,
                        help='Scale update frequency  for custom dropout')
    parser.add_argument('--update_scaling_steps', default=5, type=int, help='Amount of frequency updates')
    parser.add_argument('--scoring-type', choices=['Conductance', 'Sensitivity',"Conductance_alt"], default='Conductance',
                        type=str, help='Scoring type for custom dropout')
    parser.add_argument('--same_batch', action='store_true', default=False,
                        help='Enable smooth scoring for custom dropout')
    parser.add_argument('--mode',type=str, default=None,choices=['cls', 'mean',"sum","topk"],
                        help='Enable smooth scoring for custom dropout')
    parser.add_argument('--noisy_score', action='store_true', default=False,
                        help='Noise addition to score')
    parser.add_argument('--noisy_dropout', action='store_true', default=False,
                        help='Noise addition to dropout')
    parser.add_argument('--min_dropout', type=float, default=0.0,
                help='Minimum allowed dropout rate')
    parser.add_argument('--after_norm', action='store_true', default=False,
                help='Calculate conductance after normalization')
    
    parser.add_argument('--alt_attention_cond', action='store_true', default=False,
                help='Calculate conductance after normalization')
    parser.add_argument('--rescaling_type',choices=['linear','projection', 'power_law'], default=None, type =str,
                    help='Method to rescale the the limits of the dropout masks')
    parser.add_argument('--stats', action='store_true', default=False,
                        help='Enable statistics logging')
    parser.add_argument('--no_attn', action='store_true', default=True,
                        help='Disable attention mechanism')
    parser.add_argument('--conductance_batch_size', type=int, default=32,
                        help='Batch size for conductance calculation')
    parser.add_argument('--switch_epochs', type=int, default=None,
                        help='Number of steps to accumulate gradients for conductance')
    parser.add_argument('--epoch-gap', type=int, default=1,
                        help='Save tracker stats/logs every N epochs (also used by build_reports)')
    parser.add_argument('--use-wds', action='store_true', default=False,
                    help='Use WebDataset shards for ImageNet-1k (timm/imagenet-1k-wds)')
    parser.add_argument('--wds-train', type=str,
                        default='/leonardo_work/EUHPC_A04_051/tdir/imagenet_data/train_wds/imagenet1k-train-*.tar',
                        help='Glob for train shards (WebDataset)')
    parser.add_argument('--wds-val', type=str,
                        default='/leonardo_work/EUHPC_A04_051/tdir/imagenet_data/val_wds/imagenet1k-validation-*.tar',
                        help='Glob for val shards (WebDataset)')
    parser.add_argument('--scaled_dropout', action='store_true', default=False,
                        help='Enable scaled dropout')
    return parser


def main(args):
    utils.init_distributed_mode(args)

    seed = args.seed + utils.get_rank()


    # 1. Python built-in RNG
    random.seed(seed)
    # 2. NumPy RNG
    np.random.seed(seed)
    # 3. Torch CPU RNG
    torch.manual_seed(seed)
    # 4. Torch CUDA RNGs (if you have GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 5. Enforce deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)



    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    
    def preload_subdataset(subdataset):
        """
        Given a small subdataset (a torch.utils.data.Subset),
        load all (data, target) pairs into memory as a list.
        """
        cached = [subdataset[i] for i in range(len(subdataset))]
        return cached
    
    if args.sub_dataset == 'stratified':
        sub_dataset = create_subdataset(dataset_train, batch_size=args.batch_size, sub_factor=args.sub_factor, stratified=True)
        cached_subdataset = preload_subdataset(sub_dataset)
    elif args.sub_dataset == 'random':
        sub_dataset = create_subdataset(dataset_train, batch_size=args.batch_size, sub_factor=args.sub_factor, stratified=False)
        cached_subdataset = preload_subdataset(sub_dataset)
    else:
        cached_subdataset = None




    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()

        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # if args.ThreeAugment:
    #     data_loader_train.dataset.transform = new_data_aug_generator(args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)


    print(f"Creating model: {args.model}")

    model = create_model(
    args.model,
    pretrained=False,
    num_classes=args.nb_classes,
    drop_rate=args.drop_rate,   # changed from --drop
    drop_path_rate=args.drop_path,
    drop_block_rate=args.drop_block,
    # pass our extra custom keys. You can add them here:
    ydrop=args.ydrop,
    mask_type=args.mask_type,
    elasticity=args.elasticity,
    scaler=args.scaler,
    n_steps=args.n_steps,
    transformer_mean=True,
    rescaling_type=args.rescaling_type,
)
                    
   
    ### TO CHECK: AFTER NORM
    if args.after_norm:
        for i,block in enumerate(model.blocks):
            model.selected_layers[i*4 + 1] = block.norm2

        for i in range(len(model.blocks)-1):
            model.selected_layers[i*4 + 3] = model.blocks[i+1].norm1
    if args.alt_attention_cond:
        for i, block in enumerate(model.blocks):
            model.selected_layers[i*4] = block.attn.qkv

    if args.ydrop:
        model.selected_layers = []
        model.drop_list = []
        for i, block in enumerate(model.blocks):
            block.attn.attn_drop = torch.nn.Dropout(args.drop_rate)  # Disable attention dropout
            #block.attn.proj_drop = torch.nn.Dropout(args.drop_rate)  # Disable projection dropout
            # model.selected_layers.append(block.attn.attention_identity_layer)
            if args.after_norm:
                model.selected_layers.append(block.norm2)
            else:
                model.selected_layers.append(block.attn.proj)
            model.selected_layers.append(block.mlp.fc1)

            if i < len(model.blocks) - 1 and args.after_norm:
                model.selected_layers.append(model.blocks[i+1].norm1)
            else:
                model.selected_layers.append(block.mlp.fc2)
            # model.drop_list.append(block.attn.attn_drop)
            model.drop_list.append(block.attn.proj_drop)
            model.drop_list.append(block.mlp.drop1)
            model.drop_list.append(block.mlp.drop2)


    if args.scaled_dropout and args.ydrop:
        rates = np.linspace(0, args.drop_rate, len(model.blocks))
        model.drop_list = []
        model.selected_layers = []
        for i,block in enumerate(model.blocks):
            if i == 0:
                block.attn.attn_drop = torch.nn.Identity() # Disable attention dropout
                block.attn.proj_drop = torch.nn.Identity()  # Disable projection dropout
                block.mlp.drop1 = torch.nn.Identity()  # Disable attention dropout
                block.mlp.drop2 = torch.nn.Identity()  # Disable projection dropout
            else:
                block.attn.attn_drop = torch.nn.Dropout(rates[i])  # Disable attention dropout
                block.attn.proj_drop = MyDropout(elasticity=args.elasticity, p=rates[i], tied_layer=block.attn.proj, mask_type=args.mask_type, scaler=args.scaler,
                                    transformer_mean=True,rescaling_type=args.rescaling_type)
                block.mlp.drop1 = MyDropout(elasticity=args.elasticity, p=rates[i], tied_layer=block.mlp.fc1, mask_type=args.mask_type, scaler=args.scaler,
                                    transformer_mean=True,rescaling_type=args.rescaling_type)  # Disable attention dropout
                block.mlp.drop2 = MyDropout(elasticity=args.elasticity, p=rates[i], tied_layer=block.mlp.fc2, mask_type=args.mask_type, scaler=args.scaler,
                                    transformer_mean=True,rescaling_type=args.rescaling_type)
                                    
                model.drop_list.append(block.attn.proj_drop)
                model.drop_list.append(block.mlp.drop1)
                model.drop_list.append(block.mlp.drop2) 
                if args.after_norm:
                    model.selected_layers.append(block.norm2)
                else:
                    model.selected_layers.append(block.attn.proj)
                model.selected_layers.append(block.mlp.fc1)
                if i < len(model.blocks) - 1 and args.after_norm:
                    model.selected_layers.append(model.blocks[i+1].norm1)
                else:
                    model.selected_layers.append(block.mlp.fc2)
    elif args.scaled_dropout:
        rates = np.linspace(0, args.drop_rate, len(model.blocks))
        for i,block in enumerate(model.blocks):
            if i == 0:
                block.attn.attn_drop = torch.nn.Identity() # Disable attention dropout
                block.attn.proj_drop = torch.nn.Identity()  # Disable projection dropout
                block.mlp.drop1 = torch.nn.Identity()  # Disable attention dropout
                block.mlp.drop2 = torch.nn.Identity()  # Disable projection dropout
            else:
                block.attn.attn_drop = torch.nn.Dropout(rates[i])  # Disable attention dropout
                block.attn.proj_drop =  torch.nn.Dropout(rates[i])
                block.mlp.drop1 = torch.nn.Dropout(rates[i])
                block.mlp.drop2 = torch.nn.Dropout(rates[i])




    
    # for i, block in enumerate(model.blocks):
        
    #     block.mlp.drop1 = torch.nn.Dropout(args.drop_rate)  # Disable attention dropout
    #     block.mlp.drop2 = torch.nn.Dropout(args.drop_rate)  # Disable projection dropout   
    #     model.selected_layers.append(block.attn.attention_identity_layer)
    #     model.selected_layers.append(block.norm2)
    #     model.drop_list.append(block.attn.attn_drop)
    #     model.drop_list.append(block.attn.proj_drop)


    #added


    # TODO: finetuning

    model.to(device)
    dummy_input = torch.randn(1, 3, args.input_size, args.input_size, device=device)
    model(dummy_input)
    
    
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        ema_device = torch.device('cpu') if args.model_ema_force_cpu else device
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device=ema_device,
            resume='')


    # model_without_ddp = model
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_without_ddp = model.module
    ##GPT CHANGES###
    model_without_ddp = model
    if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module
    ###############    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    print("base lr: %.2e" % (args.lr))
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()


        # model_without_ddp.criter = PerSampleSoftTargetCE()
        #model_without_ddp.criter = LabelSmoothingCrossEntropyNoRed(smoothing=args.smoothing)
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        model_without_ddp.criter = LabelSmoothingCrossEntropyNoRed(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        model_without_ddp.criter = torch.nn.CrossEntropyLoss(reduction='none')

    args.experiment_name = f"{args.experiment_name}_seed{args.seed}"


    output_dir = Path(args.output_dir)
    output_dir = output_dir / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    default_resume = True if not args.resume else False
    if default_resume:
        resume_path = output_dir / 'checkpoint.pth'
        if resume_path.exists():
            args.resume = str(resume_path)
            print(f"No --resume given, auto-resuming from {args.resume}")
        else:
            print(f"No --resume given and no checkpoint at {resume_path}, starting fresh.")


    if args.resume:
        try:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)

            model_without_ddp.load_state_dict(checkpoint['model'])
            model_without_ddp.to(device)
           
            if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                
              
                if args.model_ema:
                    utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                if 'loss_scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['loss_scaler'])
            
            best_loss = checkpoint.get('lowest_loss', float('inf'))
            cumulative_train_time = checkpoint.get('train_time', 0.0)
            saved_epoch = checkpoint.get('epoch', 0)
            if saved_epoch >0:
                saved_epoch+=1

            lr_scheduler.step(saved_epoch)

            best_acc = checkpoint.get('best_acc', 0.0)
            patience_counter = checkpoint.get('patience_counter', 0)
            best_epoch = checkpoint.get('best_epoch', 1)
        except Exception as e:
            print("Error loading:",e)  
            best_loss = float('inf')
            saved_epoch = 0
            cumulative_train_time = 0.0
            best_acc = 0.0
            patience_counter = 0
            best_epoch =1

    else:
        best_loss = float('inf')
        saved_epoch = 0
        cumulative_train_time = 0.0
        best_acc = 0.0
        patience_counter = 0
        best_epoch =1

    



    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    print("Start training")
    #initially normal dropout
    if args.ydrop:
        if hasattr(model, 'module'):
            model.module.use_normal_dropout()
        else:
            model.use_normal_dropout() 

    check = False
    if args.stats:
        stats_dir = os.path.join(output_dir, "stats")
        stats_dir2 = os.path.join(output_dir, "stats_minmax")

        tracker = StreamingConductanceEpochTracker(
            output_dir=stats_dir,
            transformer=True,   # or False
            block_mod=3,        # your 4-layers-per-block rule
            cv_mode="signed",
            sign_eps=0.0
        )
        tracker_post_minmax = StreamingConductanceEpochTracker(
            output_dir=stats_dir2,
            transformer=True,   # or False
            block_mod=3,        # your 4-layers-per-block rule
            cv_mode="signed",
            sign_eps=0.0
        )
    else:
        tracker = None
        tracker_post_minmax = None


    delta  = 0.1   # required improvement (use 10.0 if acc is in [0,100])
    alive = True            # your boolean that flips
    anchor_best = None      # best accuracy at the start of the current window
    anchor_epoch = None 

    for epoch in range(saved_epoch, args.epochs):
        save_this_epoch = args.stats and ((epoch % args.epoch_gap) == 0 or epoch == args.epochs - 1)

        if save_this_epoch:
            tracker.begin_epoch(epoch)
            tracker_post_minmax.begin_epoch(epoch)
        epoch_start_time = time.time()

        if not alive:
            check = False
            if args.ydrop:
                if hasattr(model, 'module'):
                    model.module.use_normal_dropout()
                else:
                    model.use_normal_dropout()
        elif (args.ydrop or args.ypath) and epoch >= args.annealing_factor:

            if args.ydrop:
                if hasattr(model, 'module'):
                    model.module.use_ydrop()
                else:
                    model.use_ydrop() 
            check = True


        
        if args.distributed:# and not using_wds:
            data_loader_train.sampler.set_epoch(epoch)
            #data_loader_train_clean.sampler.set_epoch(epoch)

        max_norm = None if (args.clip_grad is None or args.clip_grad <= 0) else float(args.clip_grad)

        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            max_norm=max_norm,
            model_ema=model_ema,
            mixup_fn=mixup_fn,
            check=check,
            update_freq=args.update_freq,
            update_batches=args.update_batches,
            tracker=[tracker, tracker_post_minmax] if save_this_epoch else None,   # NEW
            scoring_type = args.scoring_type,
            same_batch = args.same_batch,
            help_par = 1,
            noisy_dropout = args.noisy_dropout,
            update_data_loader = cached_subdataset,
            min_dropout=args.min_dropout,
            alt_attention_cond=args.alt_attention_cond,
            mask_type=args.mask_type,
            ypath= args.ypath,
            support_loader=None,
            conductance_batch_size =args.conductance_batch_size,
            mode = args.mode,
            no_attn = args.no_attn
        )

        

        lr_scheduler.step(epoch)
        epoch_time = time.time() - epoch_start_time
        cumulative_train_time += epoch_time


        test_stats = evaluate(data_loader_val, model, device)
        #val_count = IMAGENET_VAL_COUNT if using_wds else len(dataset_val)
        #print(f"Accuracy of the network on the {val_count} test images: {test_stats['acc1']:.1f}%")
        test_acc = test_stats.get('acc1', 0.0)
        test_loss = test_stats.get('loss', 0.0)
        

        if save_this_epoch:
            tracker.end_epoch()
            tracker_post_minmax.end_epoch()

        if test_stats.get('acc1', 0) > best_acc:
            best_acc = test_stats.get('acc1', 0)
            best_epoch = epoch + 1

        # if args.switch_epochs is not None and args.ydrop and epoch >= args.switch_epochs:
        #     alive = False
            
        checkpoint ={
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'model_ema': get_state_dict(model_ema),
                'loss_scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
                'args': args,
                'test_acc': test_acc,
                'test_loss': test_loss,
                'lowest_loss': best_loss,
                'train_time': cumulative_train_time,  # cumulative training time so far
                'best_acc': best_acc,
                'patience_counter': patience_counter,
                'best_epoch': best_epoch
                }

        if test_loss < best_loss:
            best_loss = test_loss
            checkpoint['lowest_loss'] = best_loss
            patience_counter = 0  # reset early stopping counter
            checkpoint['patience_counter'] = patience_counter


        else:
            patience_counter += 1
            checkpoint['patience_counter'] = patience_counter
                 
        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss {train_stats['loss']:.4f}, "
              f"Test Acc {test_stats.get('acc1', 0):.2f}%, Epoch Time {epoch_time:.2f}s, Patience Counter {patience_counter}")
        
        if args.output_dir:
            utils.save_on_master(checkpoint, output_dir / 'checkpoint.pth')
        

        
        log_stats = {
            'epoch': epoch,
            'train_loss': train_stats.get('loss', 0),
            'test_acc': test_stats.get('acc1', 0),
            'time': cumulative_train_time,
            'best_acc': best_acc,
            'test_loss': test_stats.get('loss', 0),
            'best_loss': best_loss,
            'patience_counter': patience_counter,
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if patience_counter >= args.early_stopping_patience:
            print(f"Early stopping triggered. No improvement in eval loss for {args.early_stopping_patience} epochs.")
            break


    total_time_str = str(datetime.timedelta(seconds=int(cumulative_train_time)))
    print(f"Training complete. Best Test Accuracy: {best_acc:.2f}% at epoch {best_epoch}. Total training time: {total_time_str}")
    if args.stats:
        print('Building stats reports')
        build_reports(
            output_dir=stats_dir,
            transformer=True,
            block_mod=4,
            cv_mode="signed",
            bins=200,
            epoch_gap=args.epoch_gap,   # NEW
        )
        print('Building minmax stats reports')
        build_reports(
            output_dir=stats_dir2,
            transformer=True,
            block_mod=4,
            cv_mode="signed",
            bins=200,
            epoch_gap=args.epoch_gap,   # NEW
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

