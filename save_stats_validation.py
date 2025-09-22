# Single-file script: load a saved model, run over the validation set,
# compute conductance-like scores (calculate_scores) per batch, and
# save ONE tracker epoch to disk.
#
# Keeps the overall structure similar to your main.py (argparse + main()).
# Uses the same imports you already rely on (datasets2.build_dataset, utils,
# timm.create_model, updated_transformer.pruning_indices.calculate_scores,
# and stats_logging.StreamingConductanceEpochTracker).

import argparse
import datetime
from email import parser
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from timm.models import create_model

# project-local imports (same as your codebase)
from datasets2 import build_dataset
import models

import utils
from stats_logging import StreamingConductanceEpochTracker
from updated_transformer.pruning_indices import calculate_scores


def _ddp_is_on():
    return dist.is_available() and dist.is_initialized()

@torch.no_grad()
def _broadcast_boolean(flag: bool) -> bool:
    """Helper to keep a boolean consistent across ranks when needed."""
    if not _ddp_is_on():
        return flag
    t = torch.tensor([1 if flag else 0], device=torch.device(f"cuda:{torch.cuda.current_device()}"))
    dist.broadcast(t, src=0)
    return bool(t.item())

@torch.no_grad()
def _ddp_avg_scores_(scores: dict):
    """In-place all-reduce (mean) of every tensor value in `scores`.
    Assumes same keys and shapes on all ranks.
    """
    if not _ddp_is_on():
        return scores
    ws = dist.get_world_size()
    for k, v in scores.items():
        dist.all_reduce(v, op=dist.ReduceOp.SUM)
        v.div_(ws)
    return scores


def get_args_parser():
    parser = argparse.ArgumentParser('Validation-only scoring & tracking', add_help=True)

    # Model / checkpoint
    parser.add_argument('--model', default='deit_tiny_patch16_224', type=str)
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--resume', required=True, type=str, help='Path to checkpoint with a saved model (checkpoint.pth)')

    # Dataset
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR10', 'CIFAR100', 'IMNET', 'INAT', 'INAT19'])
    parser.add_argument('--data-path', required=True, type=str)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true', default=True)
    parser.add_argument('--color_jitter', type=float, default=0.3)
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1')
    parser.add_argument('--train_interpolation', type=str, default='bicubic')
    parser.add_argument('--reprob', type=float, default=0.25)
    parser.add_argument('--remode', type=str, default='pixel')
    parser.add_argument('--recount', type=int, default=1)

    # Scoring/tracker config (mirrors your training flags where relevant)
    parser.add_argument('--scoring-type', choices=['Conductance', 'Sensitivity', 'Conductance_alt'], default='Conductance')
    parser.add_argument('--mode', type=str, default=None, choices=['cls', 'mean', 'sum', 'topk'])
    parser.add_argument('--epoch-id', type=int, default=0, help='Tracker epoch number to save under')

    # Y-Drop / selected layers wiring (copied from your main, simplified to only what scoring needs)
    parser.add_argument('--ydrop', action='store_true', default=True)
    parser.add_argument('--no-ydrop', dest='ydrop', action='store_false')
    parser.add_argument('--after_norm', action='store_true', default=False)
    parser.add_argument('--alt_attention_cond', action='store_true', default=False)  # not used here but kept for parity
    parser.add_argument('--mask_type', default='sigmoid', type=str)
    parser.add_argument('--rescaling_type', choices=['linear', 'projection', 'power_law'], default=None)
    parser.add_argument('--elasticity', type=float, default=0.01)
    parser.add_argument('--scaler', type=float, default=1.0)
    parser.add_argument('--n_steps', type=int, default=5)
    parser.add_argument('--dist_url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--distributed', action='store_true', default=False)

    # Output
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--experiment_name', default='val_scores')

    # Reproducibility
    parser.add_argument('--seed', default=0, type=int)

    return parser


def main(args):
    utils.init_distributed_mode(args)

    # Seeding (same style you used)
    seed = args.seed + utils.get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Device
    if args.distributed:
        if not hasattr(args, 'gpu') or args.gpu is None:
            args.gpu = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID', 0)))
        torch.cuda.set_device(args.gpu)
        args.device = f"cuda:{args.gpu}"
    device = torch.device(args.device)

    # ===== Dataset / val loader =====
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)  # to get nb_classes
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # ===== Model =====
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=0.1,
        drop_path_rate=0.0,
        drop_block_rate=None,
        ydrop=args.ydrop,
        mask_type=args.mask_type,
        elasticity=args.elasticity,
        scaler=args.scaler,
        n_steps=args.n_steps,
        transformer_mean=True,
        rescaling_type=args.rescaling_type,
    )

    # Build the layer list used for scoring (mirrors your training wiring)

    if args.ydrop:
        for i, block in enumerate(model.blocks):
            block.attn.attn_drop = torch.nn.Dropout(0.1)  # Disable attention dropout


    model.to(device)

    # Warm a forward pass to ensure shapes are materialized for any hooks
    with torch.no_grad():
        dummy = torch.randn(1, 3, args.input_size, args.input_size, device=device)
        model(dummy)

    # Load checkpoint (expects the dict layout saved by your training script)
    ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    if utils.is_main_process():
        print('[CKPT] Missing keys:', missing)
        print('[CKPT] Unexpected keys:', unexpected)

    # DDP wrap (eval-only but keeps parity with your stack)
    selected_layers = []
    for i, block in enumerate(model.blocks):

        # attention/proj or norm2 depending on after_norm
        if args.after_norm:
            selected_layers.append(block.norm2)
        else:
            selected_layers.append(block.attn.proj)
        # mlp fc1
        selected_layers.append(block.mlp.fc1)
        # next norm1 if after_norm and not last block else mlp fc2
        if i < len(model.blocks) - 1 and args.after_norm:
            selected_layers.append(model.blocks[i + 1].norm1)
        else:
            selected_layers.append(block.mlp.fc2)
    model.selected_layers = selected_layers
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model.eval()


    # ===== Tracker setup =====
    out_root = Path(args.output_dir) / args.experiment_name
    stats_dir = out_root / 'stats_val'
    stats_dir.mkdir(parents=True, exist_ok=True)
    tracker = StreamingConductanceEpochTracker(
        output_dir=str(stats_dir),
        transformer=True,
        block_mod=3,
        cv_mode='signed',
        sign_eps=0.0,
    )

    # Only rank 0 writes to disk to avoid collisions
    do_track = utils.is_main_process()

    if do_track:
        tracker.begin_epoch(args.epoch_id)

    # ===== Iterate over validation set and update tracker =====
    start = time.time()
    total_iters = 0

    amp_dtype = 'cuda' if device.type == 'cuda' else 'cpu'

    for it, (images, targets) in enumerate(data_loader_val):

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast(amp_dtype):
            next_batches = [(images, targets)]

            # choose which layer collection to use
        
            scores, _means = calculate_scores(
                model_without_ddp,
                next_batches,
                device,
                scoring_type=args.scoring_type,
                mode=args.mode,
                normalization=False,
                sm=False,
                selected_layers=selected_layers,
                ypath=False,
            )

        # average scores across processes so tracker sees global values
        _ddp_avg_scores_(scores)

        # Only rank 0 updates/writes the tracker to avoid duplicating counts
        if do_track:
            tracker.update(scores)

        total_iters += 1
        if utils.is_main_process() and (it % 50 == 0):
            print(f"[val] iter {it:05d} | batches used={len(next_batches)}")

    if do_track:
        tracker.end_epoch()
        elapsed = time.time() - start
        print(f"Saved tracker epoch {args.epoch_id} in {str(datetime.timedelta(seconds=int(elapsed)))}")
        print(f"Output: {stats_dir}/epoch_{args.epoch_id:04d}")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
