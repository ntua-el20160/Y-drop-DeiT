# finetune_deit.py
# Minimal finetune entrypoint that keeps your Y-Drop + training stack.
# Uses DeiT-style defaults as ARG DEFAULTS (you can override on CLI).

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved. CC-BY-NC (matches original)

import argparse
import datetime
import json
import time
from pathlib import Path
from samplers import RASampler
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models import create_model
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler, get_state_dict, ModelEma
from types import MethodType

# your project modules
import utils
from datasets2 import build_dataset
from engine import train_one_epoch, evaluate

# Y-Drop pieces you already use
from updated_transformer.dynamic_dropout import MyDropout
from main import crit_for
# -----------------------------
# args: expose DeiT-ish “good” values as defaults (NOT fixed)
# -----------------------------
def get_args():
    p = argparse.ArgumentParser("DeiT finetune", add_help=False)

    # --- core model/dataset ---
    p.add_argument('--model', default='deit_tiny_patch16_224', type=str)
    p.add_argument('--input-size', default=224, type=int)
    p.add_argument('--data-set', default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'IMNET'])
    p.add_argument('--data-path', required=True, type=str)
    p.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                    help='Color jitter factor (default: 0.3)')
    p.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". " + \
                            "(default: rand-m9-mstd0.5-inc1)'),
    p.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    p.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    p.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    p.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    p.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    p.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
# --- epochs/batch ---
    p.add_argument('--batch-size', default=128, type=int)          # typical for CIFAR on a single GPU
    p.add_argument('--ft_epochs', default=25, type=int,            # “couple” of epochs; paper uses ~25 for finetune @384
                   help='number of finetuning epochs')
    p.add_argument('--num_workers', default=8, type=int)

    # --- optimizer/scheduler (DeiT defaults as defaults) ---
    p.add_argument('--opt', default='adamw', type=str)
    p.add_argument('--weight-decay', default=0.05, type=float)
    p.add_argument('--sched', default='cosine', type=str)
    p.add_argument('--lr', default=5e-4, type=float,
                   help='base LR before linear scaling; will be rescaled by batch/world/512 unless --ft_lr is set')
    p.add_argument('--ft_lr', default=None, type=float,
                   help='override the scaled LR for finetuning (recommended smaller, e.g., 1e-4)')
    p.add_argument('--warmup-epochs', default=5, type=int)
    p.add_argument('--min-lr', default=1e-5, type=float)
    p.add_argument('--clip-grad', default=None, type=float)

    # --- aug/regularization (paper defaults as defaults) ---
    p.add_argument('--smoothing', default=0.1, type=float)
    p.add_argument('--mixup', default=0.8, type=float)
    p.add_argument('--cutmix', default=1.0, type=float)
    p.add_argument('--mixup-prob', default=1.0, type=float)
    p.add_argument('--mixup-switch-prob', default=0.5, type=float)
    p.add_argument('--mixup-mode', default='batch', type=str)
    p.add_argument('--drop_rate', default=0.0, type=float)         # DeiT uses no plain dropout
    p.add_argument('--drop_path', default=0.1, type=float)         # stochastic depth
    p.add_argument('--scaled_dropout', action='store_true', default=False)

    # --- Y-Drop (kept, with your defaults) ---
    p.add_argument('--ydrop', action='store_true', default=True)
    p.add_argument('--no-ydrop', dest='ydrop', action='store_false')
    p.add_argument('--elasticity', type=float, default=0.01)
    p.add_argument('--annealing_factor', type=float, default=5)
    p.add_argument('--n_steps', type=int, default=5)
    p.add_argument('--mask_type', default='rank', type=str)
    p.add_argument('--scaler', default=1.0, type=float)
    p.add_argument('--after_norm', action='store_true', default=False)
    p.add_argument('--rescaling_type', choices=['linear', 'projection', 'power_law', None],
                   default=None)
    p.add_argument('--mode',type=str, default=None,choices=['cls', 'mean',"sum","topk"],
                    help='Enable smooth scoring for custom dropout')
    p.add_argument('--scoring-type', choices=['Conductance', 'Sensitivity',"Conductance_alt"], default='Conductance',
                        type=str, help='Scoring type for custom dropout')
    p.add_argument('--conductance_batch_size', type=int, default=32,
                   help='Batch size for conductance calculation')
    # (optional) freeze backbone for quick adaptation
    p.add_argument('--freeze_backbone', action='store_true', default=False)

    # --- ema / device / io ---
    p.add_argument('--model-ema', action='store_true', default=False)
    p.add_argument('--model-ema-decay', type=float, default=0.99996)
    p.add_argument('--no-model-ema', action='store_false', dest='model_ema')

    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', default=0, type=int)
    p.add_argument('--pin-mem', action='store_true', default=True)
    p.add_argument('--output_dir', default='', type=str)
    p.add_argument('--experiment_name_baseline', default='simpletransformer', type=str, help='experiment name')
    p.add_argument('--experiment_name_output', default='ydrop', type=str, help='experiment name')
    # --- finetune checkpoint ---
    p.add_argument('--baseline_dir', required=True, type=str,
                   help='path or URL to pretrained checkpoint saved like DeiT main.py')
    p.add_argument('--update_freq',type=int,default = 1,
                    help ='intermediate steps for conductance calculation')
    # --- distributed toggles (harmless if single-GPU) ---
    p.add_argument('--distributed', action='store_true', default=False)
    p.add_argument('--world_size', default=1, type=int)
    p.add_argument('--dist_url', default='env://', type=str)

    p.add_argument('--repeated-aug', action='store_true')
    p.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    p.set_defaults(repeated_aug=True)
    p.add_argument('--cutmix-minmax', type=float, nargs='+', default=None)
    p.add_argument('--model-ema-force-cpu', action='store_true', default=False)
    # ---- Optimizer details (match main.py) ----
    p.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                help='Optimizer Epsilon (default: 1e-8)')
    p.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                help='Optimizer Betas (default: None, use opt default)')

    # ---- LR schedule extras (match main.py) ----
    p.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                help='learning rate noise on/off epoch percentages')
    p.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                help='learning rate noise limit percent (default: 0.67)')
    p.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                help='learning rate noise std-dev (default: 1.0)')
    p.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                help='warmup learning rate (default: 1e-6)')
    p.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                help='epoch interval to decay LR')
    p.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    p.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                help='patience epochs for Plateau LR scheduler (default: 10')
    p.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                help='LR decay rate (default: 0.1)')

    # ---- Model regularization knobs (present in main.py) ----
    p.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                help='Drop block rate (default: None)')

    # ---- Data loader / eval toggles (parity with main.py) ----
    p.add_argument('--no-pin-mem', action='store_false', dest='pin_mem', help='')
    p.add_argument('--dist-eval', action='store_true', default=False,
                help='Use DistributedSampler for validation')

    # ---- Misc feature toggles present in main.py ----
    p.add_argument('--cosub', action='store_true')  # even if unused in finetune, keep parity

    return p.parse_args()

# -----------------------------
# load + adapt pretrained weights (drop heads + interpolate pos_embed)
# -----------------------------
@torch.no_grad()
def load_for_finetune(model, finetune_ckpt):
    ckpt_str = str(finetune_ckpt)
    if ckpt_str.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            finetune_ckpt, map_location='cpu', check_hash=True
        )
    else:
        checkpoint = torch.load(finetune_ckpt, map_location='cpu', weights_only=False)

    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()

    # drop mismatched classifier (and distillation) heads
    for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
        if k in checkpoint_model and k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
            print(f'[finetune] Removing incompatible key: {k} {tuple(checkpoint_model[k].shape)} -> {tuple(state_dict[k].shape)}')
            del checkpoint_model[k]

    # interpolate position embeddings if patch grid changed
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']          # [1, N_ckpt, D]
        embed_dim = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches                   # N_new
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches    # 1 (CLS) or 2 (CLS+DIST)

        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size  = int(num_patches ** 0.5)
        if orig_size != new_size:
            print(f'[finetune] Interpolate pos_embed: {orig_size}x{orig_size} -> {new_size}x{new_size}')
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]    # [1, H*W, D]
            pos_tokens = pos_tokens.reshape(1, orig_size, orig_size, embed_dim).permute(0, 3, 1, 2)
            pos_tokens = F.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)  # [1, H'*W', D]
            checkpoint_model['pos_embed'] = torch.cat((extra_tokens, pos_tokens), dim=1)

    missing, unexpected = model.load_state_dict(checkpoint_model, strict=False)
    if missing:   print(f'[finetune] Missing keys (expected for head): {len(missing)}')
    if unexpected:print(f'[finetune] Unexpected keys: {unexpected}')

# -----------------------------
# wire Y-Drop exactly like your main.py (kept)
# -----------------------------


def main():
    args = get_args()
    utils.init_distributed_mode(args)

    # seeds/determinism (matches your style)
    seed = args.seed + utils.get_rank()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(args.device)
    if args.distributed and torch.cuda.is_available():
    # utils.init_distributed_mode usually sets args.gpu; if not, LOCAL_RANK does.
        local_rank = getattr(args, "gpu", None)
        if local_rank is None:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            args.gpu = local_rank
        torch.cuda.set_device(local_rank)

    # datasets (reuses your builder; CIFAR will be upsampled to 224 with IMAGENET mean/std inside)
    dataset_train, args.nb_classes = build_dataset(is_train=True,  args=args)
    dataset_val,   _               = build_dataset(is_train=False, args=args)

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
        
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=True
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size), num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False
    )
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    # model
    print(f"Creating model: {args.model}")

    model = create_model(
        args.model, pretrained=False, num_classes=args.nb_classes,
        drop_rate=args.drop_rate, drop_path_rate=args.drop_path
    ).to(device)
    
    model.n_steps = args.n_steps
    
    if args.ydrop:
        model.selected_layers = []
        model.drop_list = []
        for i, block in enumerate(model.blocks):
            block.attn.attn_drop = torch.nn.Dropout(args.drop_rate)  # Disable attention dropout
            block.attn.proj_drop = MyDropout(elasticity=args.elasticity, p=args.drop_rate, tied_layer=block.attn.proj, mask_type=args.mask_type, scaler=args.scaler,
                                transformer_mean=True,rescaling_type=args.rescaling_type)
            block.mlp.drop1 = MyDropout(elasticity=args.elasticity, p=args.drop_rate, tied_layer=block.mlp.fc1, mask_type=args.mask_type, scaler=args.scaler,
                                transformer_mean=True,rescaling_type=args.rescaling_type)  # Disable attention dropout
            block.mlp.drop2 = MyDropout(elasticity=args.elasticity, p=args.drop_rate, tied_layer=block.mlp.fc2, mask_type=args.mask_type, scaler=args.scaler,
                                transformer_mean=True,rescaling_type=args.rescaling_type)
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
    else:
        for i, block in enumerate(model.blocks):
            block.attn.attn_drop = torch.nn.Dropout(args.drop_rate)  # Disable attention dropout
            block.attn.proj_drop = torch.nn.Dropout(args.drop_rate)  # Disable projection dropout
            block.mlp.drop1 = torch.nn.Dropout(args.drop_rate)  # Disable attention dropout
            block.mlp.drop2 = torch.nn.Dropout(args.drop_rate)  # Disable projection dropout


    if args.scaled_dropout and args.ydrop:
        rates = np.linspace(0, args.drop_rate, len(model.blocks))
        model.drop_list = []
        model.selected_layers = []
        for i,block in enumerate(model.blocks):
            if i == 0:
                block.attn.attn_drop = torch.nn.Dropout(0.0) # Disable attention dropout
                block.attn.proj_drop = torch.nn.Dropout(0.0)  # Disable projection dropout
                block.mlp.drop1 = torch.nn.Dropout(0.0)  # Disable attention dropout
                block.mlp.drop2 = torch.nn.Dropout(0.0)  # Disable projection dropout
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
                block.attn.attn_drop = torch.nn.Dropout(0.0) # Disable attention dropout
                block.attn.proj_drop = torch.nn.Dropout(0.0)  # Disable projection dropout
                block.mlp.drop1 = torch.nn.Dropout(0.0)  # Disable attention dropout
                block.mlp.drop2 = torch.nn.Dropout(0.0)  # Disable projection dropout
            else:
                block.attn.attn_drop = torch.nn.Dropout(rates[i])  # Disable attention dropout
                block.attn.proj_drop =  torch.nn.Dropout(rates[i])
                block.mlp.drop1 = torch.nn.Dropout(rates[i])
                block.mlp.drop2 = torch.nn.Dropout(rates[i])

    # prime (some custom modules need a first forward)
    model.to(device)

    _ = model(torch.randn(1, 3, args.input_size, args.input_size, device=device))


    # load pretrained for finetune

    # optional: freeze backbone (head-only finetune)
    if args.freeze_backbone:
        for n, p in model.named_parameters():
            if 'head' not in n:
                p.requires_grad = False

    # EMA (optional)
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
            ddp_dev = args.gpu if getattr(args, "gpu", None) is not None else torch.cuda.current_device()
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[ddp_dev],
                output_device=ddp_dev,
                broadcast_buffers=False,  # <- critical: avoids NCCL broadcasting CPU buffers
            )
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    # loss
    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
        model_without_ddp.criter = torch.nn.CrossEntropyLoss(reduction='none')
        

        # model_without_ddp.criter = PerSampleSoftTargetCE()
        #model_without_ddp.criter = LabelSmoothingCrossEntropyNoRed(smoothing=args.smoothing)
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        model_without_ddp.criter = torch.nn.CrossEntropyLoss(reduction='none')
        # model_without_ddp.criter = LabelSmoothingCrossEntropyNoRed(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        model_without_ddp.criter = torch.nn.CrossEntropyLoss(reduction='none')

    model_without_ddp.crit_for = MethodType(crit_for, model_without_ddp)
    
    # LR scaling (DeiT rule): lr * batch * world / 512, unless ft_lr overrides
    world = utils.get_world_size()
    scaled_lr = args.lr * args.batch_size * world / 512.0
    if args.ft_lr is not None:
        scaled_lr = args.ft_lr
    args.lr = scaled_lr
    print(f'[finetune] Using LR={args.lr:.3e} (world={world}, batch={args.batch_size})')

    # optimizer/scheduler/amp
    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    # training loop (ft_epochs only)
    best_acc = 0.0
    args.experiment_name_output = f"{args.experiment_name_output}_seed{args.seed}"
    args.experiment_name_baseline = f"{args.experiment_name_baseline}_seed{args.seed}"
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    output_dir = output_dir / args.experiment_name_output

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    


    baseline_dir = Path(args.baseline_dir)
    baseline_dir = baseline_dir / args.experiment_name_baseline
    if not baseline_dir.exists():
        print(f'Error: baseline dir {baseline_dir} does not exist')
        return
    ckpt_path = baseline_dir / 'checkpoint.pth'
    if not ckpt_path.exists():
        print(f'Error: checkpoint {ckpt_path} does not exist')
        return
    load_for_finetune(model_without_ddp, ckpt_path)
    model.to(device)
    model_without_ddp.to(device)
    # IMPORTANT for your Y-Drop: start with normal dropout for first epochs if you follow annealing
    if args.ydrop:
        if hasattr(model, 'module'):
            for drop in model.module.drop_list:
                if isinstance(drop, MyDropout):
                    drop.use_normal_dropout()
        else:
            for drop in model.drop_list:
                if isinstance(drop, MyDropout):
                    drop.use_normal_dropout()
    check = False
    best_loss = float('inf')
    best_epoch = 0
    cumulative_train_time = 0.0
    patience_counter = 0
    # print("drop list length:", (model_without_ddp.drop_list))
    # print("selected layers length:", model_without_ddp.selected_layers)
    for epoch in range(args.ft_epochs):
        # switch to Y-Drop after annealing_factor epochs
        if args.ydrop and epoch >= args.annealing_factor:
            if hasattr(model, 'module'):
                for drop in model.module.drop_list:
                    if isinstance(drop, MyDropout):
                        drop.use_ydrop()
            else:
                for drop in model.drop_list:
                    if isinstance(drop, MyDropout):
                        drop.use_ydrop()
            check = True
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        epoch_start_time = time.time()
        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            max_norm=None if not args.clip_grad else float(args.clip_grad),
            model_ema=model_ema,
            mixup_fn=mixup_fn,
            check=check,
            update_freq=args.update_freq,
            update_batches=1,
            tracker = None,
            scoring_type = args.scoring_type,
            help_par = 1,
            noisy_dropout = False,
            update_data_loader = None,
            min_dropout=0.0,
            alt_attention_cond=False,
            mask_type=args.mask_type,
            ypath= False,
            conductance_batch_size =args.conductance_batch_size,
            mode = args.mode,
  # (your engine accepts extra knobs; we keep the minimal call here)
        )

        lr_scheduler.step(epoch)
        epoch_time = time.time() - epoch_start_time
        cumulative_train_time += epoch_time
        test_stats = evaluate(data_loader_val, model, device)
        test_acc = test_stats.get('acc1', 0.0)
        test_loss = test_stats.get('loss', 0.0)
        
        if test_stats.get('acc1', 0) > best_acc:
            best_acc = test_stats.get('acc1', 0)
            best_epoch = epoch + 1
        
        ema_state = get_state_dict(model_ema) if model_ema is not None else None

        checkpoint ={
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'model_ema': ema_state,
                'loss_scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
                'args': args,
                'test_acc': test_acc,
                'test_loss': test_loss,
                'lowest_loss': best_loss,
                'train_time': cumulative_train_time,  # cumulative training time so far
                'best_acc': best_acc,
                'best_epoch': best_epoch
                }
        if test_loss < best_loss:
            best_loss = test_loss
            checkpoint['lowest_loss'] = best_loss
            patience_counter = 0
        else:
            patience_counter += 1
        # light checkpoint
        print(f"Epoch {epoch+1}/{args.ft_epochs}: Train Loss {train_stats['loss']:.4f}, "
              f"Test Acc {test_stats.get('acc1', 0):.2f}%, Epoch Time {epoch_time:.2f}s")
        
        log_stats = {
            'epoch': epoch,
            'train_loss': train_stats.get('loss', 0),
            'test_acc': test_stats.get('acc1', 0),
            'time': cumulative_train_time,
            'best_acc': best_acc,
            'test_loss': test_stats.get('loss', 0),
            'best_loss': best_loss,
            "patience_counter": patience_counter
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        # if args.output_dir:
        #     utils.save_on_master(checkpoint, output_dir / 'checkpoint.pth')


    total_time_str = str(datetime.timedelta(seconds=int(cumulative_train_time)))
    print(f"Training complete. Best Test Accuracy: {best_acc:.2f}% at epoch {best_epoch}. Total training time: {total_time_str}")

if __name__ == '__main__':
    main()
