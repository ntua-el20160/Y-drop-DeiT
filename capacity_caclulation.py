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
from types import MethodType

import numpy as np
import torch
import torch.distributed as dist
from timm.models import create_model
from updated_transformer.dynamic_dropout import MyDropout
from capacity_functions.effective_rank import effective_rank_from_attn,effective_rank_from_acts
from capacity_functions.intrinsic_dimenesnion import estimate_attention_pattern_id_per_head,estimate_vit_layer_id_twonn
from capacity_functions.simple_metrics import per_head_entropy_from_attn, count_post_relu_nonzeros
# project-local imports (same as your codebase)
from datasets2 import build_dataset
from engine import train_one_epoch, evaluate

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
def crit_for(self, x, y_true):
    """
    Per-sample loss. Keeps gradients (no detach), returns [B].
    """
    logits = self(x)  # calls forward

    # Make sure the criterion is the per-sample variant and on the right device.
    loss_fn = self.criter
    if isinstance(loss_fn, torch.nn.Module):
        # move once at setup ideally; this is just a guard
        try:
            if next(loss_fn.parameters(), None) is not None:
                loss_fn.to(logits.device)
        except StopIteration:
            pass

    # Support hard labels [B] or soft labels [B, C]
    if y_true.ndim == 1 and y_true.dtype != torch.long and logits.size(-1) > 1:
        y_true = y_true.long()
    elif y_true.ndim == 2:
        y_true = y_true.to(logits.dtype)

    loss = loss_fn(logits, y_true)   # must be reduction='none' to get [B] or [B, ...]
    if loss.ndim > 1:
        loss = loss.mean(dim=tuple(range(1, loss.ndim)))  # -> [B]
    return loss

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
    parser.add_argument('--scaled_dropout', action='store_true', default=False,
                        help='whether to use scaled dropout')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--distributed', action='store_true', default=False)

    # Output
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--experiment_name', default='val_scores')

    # Reproducibility
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--only_cls', default=False, type=bool, help='Whether to only use cls token for scoring')

    return parser


def main(args):
    utils.init_distributed_mode(args)
    use_cuda = torch.cuda.is_available()

    # Decide the GPU for this rank early
    if dist.is_available() and dist.is_initialized():
        if not hasattr(args, "gpu") or args.gpu is None:
            args.gpu = int(os.environ.get("LOCAL_RANK", 0))
        if use_cuda:
            torch.cuda.set_device(args.gpu)
            args.device = f"cuda:{args.gpu}"
        else:
            args.device = "cpu"
        args.distributed = dist.get_world_size() > 1
    else:
        # single-process
        if use_cuda:
            # keep current device
            args.gpu = torch.cuda.current_device()
            args.device = f"cuda:{args.gpu}"
        else:
            args.gpu = None
            args.device = "cpu"

    device = torch.device(args.device)

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
    args.drop_rate = 0.1
    args.drop_path = 0.0
    # Device
    # if args.distributed:
    #     if not hasattr(args, 'gpu') or args.gpu is None:
    #         args.gpu = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID', 0)))
    #     torch.cuda.set_device(args.gpu)
    #     args.device = f"cuda:{args.gpu}"
    # device = torch.device(args.device)

    # ===== Dataset / val loader =====
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)  # to get nb_classes
    dataset_val, _ = build_dataset(is_train=False, args=args)

    # if args.distributed:
    #     num_tasks = utils.get_world_size()
    #     global_rank = utils.get_rank()
    #     sampler_val = torch.utils.data.distributed.DistributedSampler(
    #         dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
    #     )
    # else:
    #     sampler_val = torch.utils.data.SequentialSampler(dataset_val)
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
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path,
        # drop_block_rate=None,
        # img_size=args.input_size
    )
    
    model.n_steps = args.n_steps

    # Build the layer list used for scoring (mirrors your training wiring)

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

    for i, block in enumerate(model.blocks):
        block.attn.fused_attn = False
    # model_without_ddp = model
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_without_ddp = model.module
    model.to(device)

    model_without_ddp = model
    if args.distributed and dist.get_world_size() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu] if args.gpu is not None else None,
            output_device=args.gpu, broadcast_buffers=False
        )
        model_without_ddp = model.module
    model_without_ddp.criter = torch.nn.CrossEntropyLoss(reduction='none')
    model_without_ddp.crit_for = MethodType(crit_for, model_without_ddp)
    model.eval()
    # ===== Hook storage (global-ish, but you can also keep it inside main) =====
    HOOK_HANDLES = []
    HOOK_FEATURES = {
        "layers": {},      # outputs of model.selected_layers
        "attn": {},        # attention weights per block
    }
    # ====== HOOKS: selected layers, dropouts, attention weights ======

    # 1) Hook all selected layers (you already built model.selected_layers above)
    def _make_layer_hook(name):
        def hook(module, input, output):
            # store detached CPU tensors; you can keep on device if you prefer
            HOOK_FEATURES["layers"][name] = output.detach().cpu()
        return hook

    for idx, m in enumerate(model_without_ddp.selected_layers):
        h = m.register_forward_hook(_make_layer_hook(f"layer_{idx}"))
        HOOK_HANDLES.append(h)

    # 2) Hook all dropout / MyDropout modules in model.drop_list (if present)


    # 3) Hook attention weights via attn_drop
    #    In timm ViT, input to attn_drop is the softmaxed attention [B, H, N, N]
    def _make_attn_hook(block_idx):
        def hook(module, input, output):
            # input[0] is the attention matrix BEFORE dropout
            attn = input[0]
            HOOK_FEATURES["attn"][block_idx] = attn.detach().cpu()
        return hook

    for i, block in enumerate(model_without_ddp.blocks):
        h = block.attn.attn_drop.register_forward_hook(_make_attn_hook(i))
        HOOK_HANDLES.append(h)


    # ===== Tracker setup =====
    out_root = Path(args.output_dir) / args.experiment_name
    stats_dir = out_root / 'stats_val'
    stats_dir.mkdir(parents=True, exist_ok=True)


    # Only rank 0 writes to disk to avoid collisions
    do_track = utils.is_main_process()


    # ===== Iterate over validation set and update tracker =====
    start = time.time()
    total_iters = 0

    amp_dtype = 'cuda'

    # for it, (images, targets) in enumerate(data_loader_val):
    #     print("total iters:", it + 1)
    #     images = images.to(device, non_blocking=True)
    #     targets = targets.to(device, non_blocking=True)
    #     with torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype):
    #         outputs = model(images)
    #     if it +1 >1:
    #         break
    images, targets = next(iter(data_loader_val))
    images = images.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    outputs = model(images)
    
    for l in HOOK_FEATURES["layers"]:
        print(f"Layer {l} output shape: {HOOK_FEATURES['layers'][l].shape}")
        print(f"Layer {l} output stats: mean={HOOK_FEATURES['layers'][l].mean().item():.4f}, std={HOOK_FEATURES['layers'][l].std().item():.4f}")
    for b in HOOK_FEATURES["attn"]:
        print(f"Block {b} attention shape: {HOOK_FEATURES['attn'][b].shape}")
        print(f"Block {b} attention stats: mean={HOOK_FEATURES['attn'][b].mean().item():.4f}, std={HOOK_FEATURES['attn'][b].std().item():.4f}")
    
        # Clear hook buffers if you want per-batch stats
        # HOOK_FEATURES["layers"].clear()
        # HOOK_FEATURES["attn"].clear()
# ============= COMPUTE METRICS FOR THIS BATCH =============
# ===============================================================
#     COMPUTE ALL METRICS FOR A SINGLE BATCH (ONE ITERATION)
# ===============================================================

    single_run_metrics = {
        "layers": {},
        "attn": {}
    }

    # ------------ LAYER METRICS (FROM ACTIVATION HOOKS) ------------
    count = 0
    for lname, acts_cpu in HOOK_FEATURES["layers"].items():
        if count % 3 == 1:
            apply_relu = True
            print(f'Applying Relu for layer {lname}')
        else:
            apply_relu = False
        count += 1
        acts = acts_cpu.to(device)

        

        # Effective rank + participation
        effrank, participation = effective_rank_from_acts(
            acts, center=True, apply_relu=apply_relu, eps=1e-12,only_cls=args.only_cls
        )

        # TWO-NN intrinsic dimension
        id_twonn = estimate_vit_layer_id_twonn(
            acts_cpu,  # must be CPU
            max_points=200000,
            fraction_to_keep=0.9,
            apply_relu=apply_relu,
            only_cls=args.only_cls
        )

        # Sparsity (after ReLU)
        nonzero_count = count_post_relu_nonzeros(acts_cpu, only_cls=args.only_cls)

        single_run_metrics["layers"][lname] = {
            "effective_rank": float(effrank.item()),
            "participation_ratio": float(participation.item()),
            "intrinsic_dimension_twonn": float(id_twonn),
            "relu_nonzeros": int(nonzero_count),
        }

    # ------------ ATTENTION METRICS (PER BLOCK, PER HEAD) ----------
    for block_idx, attn_cpu in HOOK_FEATURES["attn"].items():
        attn = attn_cpu.to(device)

        # Per-head effective rank
        eranks, pr_heads = effective_rank_from_attn(attn)

        # Per-head intrinsic dimension TWO-NN
        id_heads = estimate_attention_pattern_id_per_head(
            attn_cpu,
            max_points_per_head=20000,
            fraction_to_keep=0.9,
            use_log=False
        )

        # Per-head entropy
        entropy_heads = per_head_entropy_from_attn(attn_cpu)

        single_run_metrics["attn"][f"block_{block_idx}"] = {
            "per_head_effective_rank": eranks.cpu().tolist(),
            "mean_effective_rank": float(eranks.mean().item()),

            "per_head_participation_ratio": pr_heads.cpu().tolist(),
            "mean_participation_ratio": float(pr_heads.mean().item()),

            "intrinsic_dimension_per_head_twonn": id_heads.cpu().tolist(),
            "intrinsic_dimension_mean_twonn": float(id_heads.mean().item()),

            "entropy_per_head": entropy_heads.cpu().tolist(),
            "entropy_mean": float(entropy_heads.mean().item())
        }

    # ------------ PRINT RESULTS (RANK 0 ONLY) ----------------------
    if utils.is_main_process():
        print("\n====== SINGLE BATCH METRICS ======")

        print("\n--- LAYERS ---")
        for lname, d in single_run_metrics["layers"].items():
            print(f"{lname}: ER={d['effective_rank']:.3f}, "
                f"PR={d['participation_ratio']:.3f}, "
                f"ID2NN={d['intrinsic_dimension_twonn']:.1f}, "
                f"NonZeros={d['relu_nonzeros']}")

        print("\n--- ATTENTION BLOCKS ---")
        for bname, d in single_run_metrics["attn"].items():
            print(f"{bname}: mean_ER={d['mean_effective_rank']:.3f}, "
                f"mean_PR={d['mean_participation_ratio']:.3f}, "
                f"mean_ID2NN={d['intrinsic_dimension_mean_twonn']:.1f}, "
                f"mean_entropy={d['entropy_mean']:.3f}")

    # ------------ SAVE JSON ----------------------------------------
    if utils.is_main_process():
        json_path = stats_dir / "single_batch_metrics.json"
        with open(json_path, "w") as f:
            json.dump(single_run_metrics, f, indent=2)

    print(f"\nSaved metrics → {json_path}")

    # ===============================================================
    # ============= COMPARE CONDUCTANCE & LOSS-CONDUCTANCE ============
    # ===============================================================
    print("\n=== CONDUCTANCE VS. LOSS-CONDUCTANCE PER LAYER CORRELATIONS ===")
    def pearson_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        a, b: 1D tensors of shape [N]
        returns: scalar tensor (Pearson correlation)
        """
        # ensure float
        a = a.float()
        b = b.float()
        
        # subtract mean
        a_centered = a - a.mean()
        b_centered = b - b.mean()
        
        # std (add small eps to avoid division by zero)
        eps = 1e-8
        a_std = a_centered.pow(2).mean().sqrt() + eps
        b_std = b_centered.pow(2).mean().sqrt() + eps
        
        # correlation = mean of elementwise product of z-scored vectors
        corr = (a_centered / a_std * b_centered / b_std).mean()
        return corr
    def ranks_from_values(x: torch.Tensor, descending: bool = False) -> torch.Tensor:
        """
        x: 1D tensor [N]
        returns: ranks [N] where 0 = smallest, N-1 = largest (or reverse if descending)
        """
        # argsort gives indices that would sort the tensor
        if descending:
            sorted_indices = torch.argsort(x, descending=True)
        else:
            sorted_indices = torch.argsort(x)
        
        # create empty rank tensor
        ranks = torch.empty_like(sorted_indices, dtype=torch.float)
        # ranks[sorted_indices[i]] = i
        ranks[sorted_indices] = torch.arange(len(x), dtype=torch.float, device=x.device)
        return ranks
    def spearman_rank_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        a, b: 1D tensors of shape [N]
        returns: scalar tensor (Spearman rank correlation)
        """
        # get ranks (largest neuron gets largest rank if descending=True)
        rank_a = ranks_from_values(a, descending=False)
        rank_b = ranks_from_values(b, descending=False)
        
        # now just Pearson on ranks
        return pearson_similarity(rank_a, rank_b)
    print("About to iterate over data_loader_val")
    print("len(data_loader_val) =", len(data_loader_val), flush=True)
    pearson_results = {}
    spearman_results = {}
    pearson_results_abs = {}
    spearman_results_abs = {}
    for batch, (images, targets) in enumerate(data_loader_val):
        print("Processing batch:", batch + 1, flush=True)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast('cuda'):
            next_batches = [(images, targets)]
            scores_conductance,_ = calculate_scores(
                    model.module if hasattr(model, 'module') else model,
                    next_batches, device, scoring_type="Conductance", mode='cls',
                    normalization=False, sm=False, selected_layers=selected_layers, ypath=False,
                    baseline= None,
                )
            scores_loss_conductance,_ = calculate_scores(
                    model.module if hasattr(model, 'module') else model,
                    next_batches, device, scoring_type="Conductance_alt", mode='cls',
                    normalization=False, sm=False, selected_layers=selected_layers, ypath=False,
                    baseline= None,
                )
            for key in scores_conductance:
                score1 = scores_conductance[key].cpu()
                score2 = scores_loss_conductance[key].cpu()
                score3 = torch.abs(score2)
                pearson_corr = pearson_similarity(score1, score2)
                spearman_corr = spearman_rank_similarity(score1, score2)
                pearson_corr_abs = pearson_similarity(score1, score3)
                spearman_corr_abs = spearman_rank_similarity(score1, score3)
                if key not in pearson_results:
                    pearson_results[key] = []
                    spearman_results[key] = []
                    pearson_results_abs[key] = []
                    spearman_results_abs[key] = []
                pearson_results[key].append(pearson_corr.item())
                spearman_results[key].append(spearman_corr.item())
                pearson_results_abs[key].append(pearson_corr_abs.item())
                spearman_results_abs[key].append(spearman_corr_abs.item())
    # Average correlations over batches
    print("Per-layer Pearson & Spearman means:")
    all_pearson_vals = []
    all_spearman_vals = []

    for key in pearson_results:
        layer_pearson = torch.tensor(pearson_results[key]).mean().item()
        layer_spearman = torch.tensor(spearman_results[key]).mean().item()
        layer_pearson_abs = torch.tensor(pearson_results_abs[key]).mean().item()
        layer_spearman_abs = torch.tensor(spearman_results_abs[key]).mean().item()

        all_pearson_vals.extend(pearson_results[key])
        all_spearman_vals.extend(spearman_results[key])
        all_pearson_vals.extend(pearson_results_abs[key])
        all_spearman_vals.extend(spearman_results_abs[key])

        print(f"  Layer {key}: Pearson mean = {layer_pearson:.4f}, "
            f"Spearman mean = {layer_spearman:.4f}")
        print(f"             Abs Pearson mean = {layer_pearson_abs:.4f}, "
            f"Abs Spearman mean = {layer_spearman_abs:.4f}")

    overall_pearson_mean = torch.tensor(all_pearson_vals).mean().item()
    overall_spearman_mean = torch.tensor(all_spearman_vals).mean().item()
    overall_pearson_mean_abs = torch.tensor(pearson_results_abs[key]).mean().item()
    overall_spearman_mean_abs = torch.tensor(spearman_results_abs[key]).mean().item()

    print("\nOverall means across all layers:")
    print(f"  Pearson overall mean  = {overall_pearson_mean:.4f}")
    print(f"  Spearman overall mean = {overall_spearman_mean:.4f}")
    print(f"  Abs Pearson overall mean  = {overall_pearson_mean_abs:.4f}")
    print(f"  Abs Spearman overall mean = {overall_spearman_mean_abs:.4f}")
    test_stats = evaluate(data_loader_val, model, device)
    test_acc = test_stats.get('acc1', 0.0)
    test_loss = test_stats.get('loss', 0.0)
    print(f"Validation accuracy: {test_acc:.2f}%, loss {test_loss:.4f}")
        # average scores across processes so tracker sees global values



if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
