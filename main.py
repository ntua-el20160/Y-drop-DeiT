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
from updated_transformer.plots import plot_epoch_statistics
from updated_transformer.dynamic_dropath import  DropPath
import copy as _copy
import os


from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset, create_subdataset
from engine import train_one_epoch, evaluate
#from losses import DistillationLosss
from samplers import RASampler
#from augment import new_data_aug_generator
import models
import utils
#import models_v2
from stats_logging import StreamingConductanceEpochTracker,build_reports

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
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
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
#     parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
#                         help='Name of teacher model to train (default: "regnety_160"')
#     parser.add_argument('--teacher-path', type=str, default='')
#     parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
#     parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
#     parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    
#     # * Cosub params
#     parser.add_argument('--cosub', action='store_true') 
    
#     # * Finetuning params
#     parser.add_argument('--finetune', default='', help='finetune from checkpoint')
#     parser.add_argument('--attn-only', action='store_true') 
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
    parser.add_argument('--transformer_mean', action='store_true', default=False,
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
    parser.add_argument('--rescaling_type',choices=['linear','piecewise', 'power_law'], default=None, type =str,
                    help='Method to rescale the the limits of the dropout masks')
    parser.add_argument('--stats', action='store_true', default=False,
                        help='Enable statistics logging')
    return parser


def main(args):
    utils.init_distributed_mode(args)

    # print(args)
    
    # if args.distillation_type != 'none' and args.finetune and not args.eval:
    #     raise NotImplementedError("Finetuning with distillation not yet supported")
    # device = torch.device(args.device)

    # # fix the seed for reproducibility
    # seed = args.seed + utils.get_rank()
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # # random.seed(seed)

    # cudnn.benchmark = True
    seed = args.seed

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

    # random.seed(seed)

    #cudnn.benchmark = True
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
    
    effective_epochs = args.epochs - args.annealing_factor
    step_size = round(effective_epochs / args.update_scaling_steps)
    denom = max(1, args.update_scaling_steps - 1) 

    dataset_train_clean = _copy.copy(dataset_train)
    dataset_train_clean.transform = dataset_val.transform

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()

        sampler_train_clean = torch.utils.data.DistributedSampler(
        dataset_train_clean, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
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
        sampler_train_clean = torch.utils.data.SequentialSampler(dataset_train_clean)


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_train_clean = torch.utils.data.DataLoader(
        dataset_train_clean, sampler=sampler_train_clean,
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
    transformer_mean=args.transformer_mean,
    rescaling_type=args.rescaling_type,
)
                    
    # if args.finetune:
    #     if args.finetune.startswith('https'):
    #         checkpoint = torch.hub.load_state_dict_from_url(
    #             args.finetune, map_location='cpu', check_hash=True)
    #     else:
    #         checkpoint = torch.load(args.finetune, map_location='cpu')

    #     checkpoint_model = checkpoint['model']
    #     state_dict = model.state_dict()
    #     for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
    #         if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
    #             print(f"Removing key {k} from pretrained checkpoint")
    #             del checkpoint_model[k]

    #     # interpolate position embedding
    #     pos_embed_checkpoint = checkpoint_model['pos_embed']
    #     embedding_size = pos_embed_checkpoint.shape[-1]
    #     num_patches = model.patch_embed.num_patches
    #     num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    #     # height (== width) for the checkpoint position embedding
    #     orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    #     # height (== width) for the new position embedding
    #     new_size = int(num_patches ** 0.5)
    #     # class_token and dist_token are kept unchanged
    #     extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    #     # only the position tokens are interpolated
    #     pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    #     pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    #     pos_tokens = torch.nn.functional.interpolate(
    #         pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    #     pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    #     new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    #     checkpoint_model['pos_embed'] = new_pos_embed

    #     model.load_state_dict(checkpoint_model, strict=False)
        


    ### TO CHECK: AFTER NORM
    if args.after_norm:
        for i,block in enumerate(model.blocks):
            model.selected_layers[i*4 + 1] = block.norm2

        for i in range(len(model.blocks)-1):
            model.selected_layers[i*4 + 3] = model.blocks[i+1].norm1
    if args.alt_attention_cond:
        for i, block in enumerate(model.blocks):
            model.selected_layers[i*4] = block.attn.qkv

#### Remove layers portion
    # model.selected_layers = []
    # model.drop_list = []

    # for i, block in enumerate(model.blocks):
    #     block.attn.attn_drop = torch.nn.Dropout(args.drop_rate)  # Disable attention dropout
    #     block.attn.proj_drop = torch.nn.Dropout(args.drop_rate)  # Disable projection dropout
    #     # model.selected_layers.append(block.attn.attention_identity_layer)
    #     model.selected_layers.append(block.mlp.fc1)
    #     if i < len(model.blocks) - 1:
    #         model.selected_layers.append(model.blocks[i+1].norm1)
    #     else:
    #         model.selected_layers.append(block.mlp.fc2)
    #     # model.drop_list.append(block.attn.attn_drop)
    #     # model.drop_list.append(block.attn.proj_drop)
    #     model.drop_list.append(block.mlp.drop1)
    #     model.drop_list.append(block.mlp.drop2)

    
    # for i, block in enumerate(model.blocks):
        
    #     block.mlp.drop1 = torch.nn.Dropout(args.drop_rate)  # Disable attention dropout
    #     block.mlp.drop2 = torch.nn.Dropout(args.drop_rate)  # Disable projection dropout   
    #     model.selected_layers.append(block.attn.attention_identity_layer)
    #     model.selected_layers.append(block.norm2)
    #     model.drop_list.append(block.attn.attn_drop)
    #     model.drop_list.append(block.attn.proj_drop)


    #added
    if args.ypath:
        ypath_layers = []
        for i, block in enumerate(model.blocks):
            block.drop_path1 = DropPath(args.drop_path)
            block.drop_path2 = DropPath(args.drop_path)
            ypath_layers.append(block.attn)
            ypath_layers.append(block.mlp)
            #ypath_layers.append(block)
        model.ypath_layers = ypath_layers
    else:
        model.ypath_layers = None
    model.drop_path_rate = args.drop_path
    model.elasticity = args.elasticity

    #added
    for i, block in enumerate(model.blocks):
        print(f"Block {i}: {block.attn.attn_drop}, {block.attn.proj_drop}, {block.mlp.drop1}, {block.mlp.drop2}")
        print(f"Block {i}: {block.norm1}, {block.norm2}, {block.drop_path1}, {block.drop_path2}")


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


    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        model_without_ddp.criter = LabelSmoothingCrossEntropy(smoothing=args.smoothing, reduction='none')
    else:
        criterion = torch.nn.CrossEntropyLoss()
        model_without_ddp.criter = torch.nn.CrossEntropyLoss(reduction='none')
    # teacher_model = None
    # if args.distillation_type != 'none':
    #     assert args.teacher_path, 'need to specify teacher-path when using distillation'
    #     print(f"Creating teacher model: {args.teacher_model}")
    #     teacher_model = create_model(
    #         args.teacher_model,
    #         pretrained=False,
    #         num_classes=args.nb_classes,
    #         global_pool='avg',
    #     )
    #     if args.teacher_path.startswith('https'):
    #         checkpoint = torch.hub.load_state_dict_from_url(
    #             args.teacher_path, map_location='cpu', check_hash=True)
    #     else:
    #         checkpoint = torch.load(args.teacher_path, map_location='cpu')
    #     teacher_model.load_state_dict(checkpoint['model'])
    #     teacher_model.to(device)
    #     teacher_model.eval()
    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    # criterion = DistillationLoss(
    #     criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    # )
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
                
                history = checkpoint.get('history', {})
                if history:
                    for i, drop in enumerate(model_without_ddp.drop_list):
                        drop_history = history.get(f'drop{i}', {})
                        drop.progression_keep = drop_history.get('progression_keep', [])
                        drop.progression_scoring = drop_history.get('progression_scoring', [])
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

    

    if args.update_scaling == 'increasing':
        update_freq = 1
    else:
        update_freq = args.update_freq
    
    effective_epochs = args.epochs - args.annealing_factor
    step_size = round(effective_epochs / args.update_scaling_steps)
    denom = max(1, args.update_scaling_steps - 1) 

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
            block_mod=4,        # your 4-layers-per-block rule
            cv_mode="signed",
            sign_eps=0.0
        )
        tracker_post_minmax = StreamingConductanceEpochTracker(
            output_dir=stats_dir2,
            transformer=True,   # or False
            block_mod=4,        # your 4-layers-per-block rule
            cv_mode="signed",
            sign_eps=0.0
        )
    else:
        tracker = None
        tracker_post_minmax = None

    for epoch in range(saved_epoch, args.epochs):
        if args.stats:
            tracker.begin_epoch(epoch)
            tracker_post_minmax.begin_epoch(epoch)
        epoch_start_time = time.time()
        
        if (args.ydrop or args.ypath) and epoch >= args.annealing_factor:
            
            if args.ydrop:
                if hasattr(model, 'module'):
                    model.module.use_ydrop()
                else:
                    model.use_ydrop() 
            check = True

            if args.update_scaling!='no':
                i = (epoch - args.annealing_factor) // step_size
                i = min(i, args.update_scaling_steps - 1)

                if args.update_scaling == 'increasing':
                    step_index = i
                else:
                    step_index = args.update_scaling_steps - 1 - i
                update_freq = round(1 + (args.update_freq - 1) * step_index / denom)
                print(f"[Epoch {epoch}] Update frequency set to {update_freq}")
        
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            data_loader_train_clean.sampler.set_epoch(epoch)
    
        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            max_norm=args.clip_grad,
            model_ema=model_ema,
            mixup_fn=mixup_fn,
            check=check,
            update_freq=update_freq,
            update_batches=args.update_batches,
            tracker = [tracker, tracker_post_minmax],
            scoring_type = args.scoring_type,
            same_batch = args.same_batch,
            help_par = 1,
            noisy_dropout = args.noisy_dropout,
            update_data_loader = cached_subdataset,
            output_dir=output_dir,
            min_dropout=args.min_dropout,
            alt_attention_cond=args.alt_attention_cond,
            mask_type=args.mask_type,
            ypath= args.ypath,
            support_loader=data_loader_train_clean,
            conductance_batch_size =64
        )

        

        lr_scheduler.step(epoch)
        epoch_time = time.time() - epoch_start_time
        cumulative_train_time += epoch_time


        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        test_acc = test_stats.get('acc1', 0.0)
        test_loss = test_stats.get('loss', 0.0)
        
        # if check and stats:
        #     if hasattr(model, 'module'):
        #         model.module.update_progression(output_dir / 'plots')
        #         model.module.plot_progression_statistics(output_dir / 'plots',label = "")
        #         model.module.save_statistics(epoch_dir)
        #         plot_epoch_statistics(output_dir, epoch+1, epoch_dir,True)
        #         model.module.clear_progression()
        #     else:
        #         model.update_progression(output_dir / 'plots')
        #         model.plot_progression_statistics(output_dir / 'plots',label = "")
        #         model.save_statistics(epoch_dir)
        #         plot_epoch_statistics(output_dir, epoch+1, epoch_dir,True)
        #         model.clear_progression()

        if args.stats:
            tracker.end_epoch()
            tracker_post_minmax.end_epoch()

        if test_stats.get('acc1', 0) > best_acc:
            best_acc = test_stats.get('acc1', 0)
            best_epoch = epoch + 1
        
        # print(f"Epoch {epoch+1}/{args.epochs}: Train Loss {train_stats['loss']:.4f}, "
        #       f"Test Acc {test_stats.get('acc1', 0):.2f}%, Epoch Time {epoch_time:.2f}s")
        
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
        if args.ydrop:
            checkpoint['history'] = {}
            for i, drop in enumerate(model.module.drop_list if hasattr(model, 'module') else model.drop_list):
                checkpoint['history'][f'drop{i}'] = {
                    'progression_keep': drop.progression_keep,
                    'progression_scoring': drop.progression_scoring,
                }   
        if test_loss < best_loss:
            best_loss = test_loss
            checkpoint['lowest_loss'] = best_loss
            patience_counter = 0  # reset early stopping counter
            checkpoint['patience_counter'] = patience_counter

            # if args.output_dir:  
            #     utils.save_on_master(checkpoint, output_dir / 'best.pth')
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
        build_reports(
            output_dir=stats_dir,
            transformer=True,   # must match how you named the layers
            block_mod=4,
            cv_mode="signed",
            bins=200
        )
        build_reports(
            output_dir=stats_dir2,
            transformer=True,   # must match how you named the layers
            block_mod=4,
            cv_mode="signed",
            bins=200
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

