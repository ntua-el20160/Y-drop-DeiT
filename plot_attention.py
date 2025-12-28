
#!/usr/bin/env python3
# (script content truncated for brevity in this cell; it's identical to the previous attempt)
# To keep things concise in this environment, we'll write the full content again below.

import argparse
import math
import os
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from timm.models import create_model
from datasets2 import build_dataset

def set_eval_with_dropout(m: torch.nn.Module, enable_mc: bool):
    if enable_mc:
        m.train()
        for mod in m.modules():
            if isinstance(mod, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d, torch.nn.LayerNorm)):
                mod.eval()
    else:
        m.eval()

def softmax_entropy(p: torch.Tensor, dim: int=-1, eps: float=1e-12) -> torch.Tensor:
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=dim)

def get_token_grid_size(seq_len: int) -> Optional[int]:
    n_patches = seq_len - 1
    r = int(round(math.sqrt(n_patches)))
    if r * r == n_patches:
        return r
    return None

def ece_score(probs: np.ndarray, labels: np.ndarray, n_bins: int=20) -> float:
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        m = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if m.any():
            acc_b = correct[m].mean()
            conf_b = conf[m].mean()
            ece += (m.mean()) * abs(acc_b - conf_b)
    return float(ece)

def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    n, k = probs.shape
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), labels] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))

import collections
import inspect
import torch
from typing import Dict, List

class AttnCapture:
    """
    Capture softmax attention (the tensor passed INTO block.attn.attn_drop).

    - Registers *both* forward_pre_hook and forward_hook on attn_drop (some PyTorch/timm combos
      only trigger one consistently). We ALWAYS record the *input* tensor.
    - Loud diagnostics: prints exactly what got hooked and what fired.
    """

    def __init__(self, model, debug: bool = True, max_prints_per_layer: int = 2):
        self.model = model
        self.debug = debug
        self.max_prints_per_layer = max_prints_per_layer

        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.buffers: Dict[int, List[torch.Tensor]] = {}
        self.fire_count = collections.Counter()     # how many times any hook fired per layer
        self.pre_fire_count = collections.Counter() # how many times pre-hook fired per layer
        self.fwd_fire_count = collections.Counter() # how many times fwd-hook fired per layer

        # enumerate target modules
        num_blocks = getattr(model, "blocks", None)
        if num_blocks is None:
            raise RuntimeError("Model has no attribute 'blocks' — not a ViT/DeiT?")

        if self.debug:
            print(f"[hook-reg] model has {len(model.blocks)} transformer blocks")

        for li, block in enumerate(model.blocks):
            if not hasattr(block, "attn"):
                if self.debug:
                    print(f"[hook-reg] L{li}: no 'attn' module, skipping")
                continue
            if not hasattr(block.attn, "attn_drop"):
                if self.debug:
                    print(f"[hook-reg] L{li}: no 'attn_drop' in Attention, skipping")
                continue

            mod = block.attn.attn_drop
            mod_cls = mod.__class__.__name__
            mod_id = id(mod)
            # Try to introspect a 'p' if it's Dropout-like
            p_val = getattr(mod, "p", None)

            if self.debug:
                print(f"[hook-reg] L{li}: attn_drop={mod_cls}(p={p_val})  id={mod_id}")

            # We record the INPUT to attn_drop from either hook.
            def make_pre(layer_index):
                def pre_hook(module, inputs):
                    # inputs: tuple of (attn_probs,)
                    if not inputs:
                        return
                    x = inputs[0]
                    if not torch.is_tensor(x):
                        return
                    self.pre_fire_count[layer_index] += 1
                    self.fire_count[layer_index] += 1
                    self.buffers.setdefault(layer_index, []).append(x.detach())
                    if self.debug and self.pre_fire_count[layer_index] <= self.max_prints_per_layer:
                        print(f"[hook-pre]  L{layer_index} fired "
                              f"shape={tuple(x.shape)} dtype={x.dtype} "
                              f"min={float(x.min()) if x.numel() else 'NA'} "
                              f"max={float(x.max()) if x.numel() else 'NA'}")
                return pre_hook

            def make_fwd(layer_index):
                def fwd_hook(module, inputs, output):
                    # Prefer INPUT (pre-dropout attention)
                    if inputs and torch.is_tensor(inputs[0]):
                        x = inputs[0]
                        self.fwd_fire_count[layer_index] += 1
                        self.fire_count[layer_index] += 1
                        self.buffers.setdefault(layer_index, []).append(x.detach())
                        if self.debug and self.fwd_fire_count[layer_index] <= self.max_prints_per_layer:
                            print(f"[hook-fwd] L{layer_index} fired "
                                  f"shape={tuple(x.shape)} dtype={x.dtype} "
                                  f"min={float(x.min()) if x.numel() else 'NA'} "
                                  f"max={float(x.max()) if x.numel() else 'NA'}")
                return fwd_hook

            # Register both; we want to SEE which one triggers
            self.hooks.append(mod.register_forward_pre_hook(make_pre(li), with_kwargs=False))
            self.hooks.append(mod.register_forward_hook(make_fwd(li)))

        if self.debug:
            print("[hook-reg] done registering hooks.")

    def clear(self):
        self.buffers.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def collect_batch(self) -> Dict[int, torch.Tensor]:
        """
        Stack per-layer captures along batch dim. Clears internal buffers.
        Returns { layer_index: tensor[B_total, H, T, T] } or {}
        """
        out = {}
        for li, lst in self.buffers.items():
            if lst:
                out[li] = torch.cat(lst, dim=0)
        self.clear()
        return out

    # Debug helper: call this after a batch if nothing fired.
    def debug_summary(self):
        if not self.debug:
            return
        print("\n===== ATTENTION HOOK DEBUG SUMMARY =====")
        total_fires = sum(self.fire_count.values())
        print(f"total fires: {total_fires}")
        for li in range(len(self.model.blocks)):
            pf = self.pre_fire_count[li]
            ff = self.fwd_fire_count[li]
            if pf or ff:
                print(f"  L{li}: pre={pf}  fwd={ff}")
        print("========================================\n")



def per_head_metrics(attn: torch.Tensor) -> Dict[str, torch.Tensor]:
    B, H, T, _ = attn.shape
    ent = softmax_entropy(attn, dim=-1).mean(dim=-1)
    cls_focus = attn[..., 0].mean(dim=-1)
    idx = torch.arange(T, device=attn.device)
    dist = (idx[None, None, None, :] - idx[None, None, :, None]).abs().to(attn.dtype)
    avg_dist = (attn * dist).sum(dim=-1).mean(dim=-1)
    grid = get_token_grid_size(T)
    if grid is not None:
        coords = torch.zeros((T, 2), device=attn.device, dtype=attn.dtype)
        if T > 1:
            ys, xs = torch.meshgrid(torch.arange(grid, device=attn.device),
                                    torch.arange(grid, device=attn.device),
                                    indexing="ij")
            coords[1:, 0] = ys.flatten().to(attn.dtype)
            coords[1:, 1] = xs.flatten().to(attn.dtype)
        c = coords
        diff = c[None, None, None, :, :] - c[None, None, :, None, :]
        dist2d = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-9)
        avg_dist2d = (attn * dist2d).sum(dim=-1).mean(dim=-1)
    else:
        avg_dist2d = torch.full_like(avg_dist, float('nan'))
    return {
        "entropy": ent,
        "cls_focus": cls_focus,
        "avg_dist": avg_dist,
        "avg_dist2d": avg_dist2d,
    }

def aggregate_metrics(layer_to_attn: Dict[int, torch.Tensor]) -> Dict[str, np.ndarray]:
    layers = sorted(layer_to_attn.keys())
    metrics = {k: [] for k in ["entropy", "cls_focus", "avg_dist", "avg_dist2d"]}
    for li in layers:
        m = per_head_metrics(layer_to_attn[li])
        for k in metrics:
            metrics[k].append(m[k].mean(dim=0).cpu().numpy())
    for k in metrics:
        metrics[k] = np.stack(metrics[k], axis=0)
    return metrics

def _finalize(ax, title, xlabel="", ylabel="", legend=False):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    if legend:
        ax.legend(frameon=True)

def heatmap(ax, data: np.ndarray, title: str, ylabel: str):
    im = ax.imshow(data, aspect='auto', interpolation='nearest')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Heads")
    ax.set_yticks(range(data.shape[0]))
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im, cbar
def toggle_fused_attention(model, enabled: bool):
    """Set fused attention flag across all blocks."""
    cnt = 0
    for i, blk in enumerate(getattr(model, "blocks", [])):
        if hasattr(blk, "attn") and hasattr(blk.attn, "fused_attn"):
            blk.attn.fused_attn = enabled
            cnt += 1
    print(f"[fused-attn] set fused_attn={enabled} on {cnt} blocks")

def compare_outputs_equivalence(model, images, atol=1e-5, rtol=1e-4):
    """Sanity-check outputs with fused on vs off in eval mode."""
    model.eval()
    # ensure deterministic behavior
    with torch.no_grad():
        toggle_fused_attention(model, True)
        y_fused = model(images).detach()
        toggle_fused_attention(model, False)
        y_manual = model(images).detach()
    max_abs = (y_fused - y_manual).abs().max().item()
    ok = torch.allclose(y_fused, y_manual, atol=atol, rtol=rtol)
    print(f"[equiv] allclose={ok}  max_abs_diff={max_abs:.3e}  (atol={atol}, rtol={rtol})")
    return ok, max_abs

def plot_metric_heatmaps(metrics_a, label_a, outdir, metrics_b=None, label_b=None, prefix=""):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for key in metrics_a.keys():
        if metrics_b is None:
            fig, ax = plt.subplots(figsize=(8, 5))
            heatmap(ax, metrics_a[key], f"{label_a} – {key}", "Layers")
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"{prefix}{key}_{label_a}.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)
        else:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            heatmap(axes[0], metrics_a[key], f"{label_a} – {key}", "Layers")
            heatmap(axes[1], metrics_b[key], f"{label_b} – {key}", "Layers")
            diff = metrics_b[key] - metrics_a[key]
            heatmap(axes[2], diff, f"Δ ({label_b} − {label_a}) – {key}", "Layers")
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"{prefix}{key}_{label_a}_vs_{label_b}.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)

def plot_token_token_maps(
    layer_to_attn: Dict[int, torch.Tensor],
    sample_indices: List[int],
    layers_to_plot: List[int],
    outdir: str,
    prefix: str,
    max_heads: int = 8
):
    """
    For selected layers, plot the full token×token attention matrix per head.
    - layer_to_attn[li]: [B, H, T, T] attention for layer li
    - sample_indices: indices within the current captured batch to visualize
    - layers_to_plot: layer ids to visualize
    - max_heads: cap how many heads per figure to keep layout readable
    """
    if not layer_to_attn:
        return
    Path(outdir).mkdir(parents=True, exist_ok=True)

    for li in layers_to_plot:
        if li not in layer_to_attn:
            continue
        A = layer_to_attn[li]  # [B, H, T, T]
        B, H, T, _ = A.shape
        n_heads = min(H, max_heads)
        n_samples = min(len(sample_indices), 4)

        fig, axes = plt.subplots(
            n_samples, n_heads,
            figsize=(2.6 * n_heads, 2.6 * n_samples),
            squeeze=False
        )

        for si, bidx in enumerate(sample_indices[:n_samples]):
            # A[b, h] is [T, T] with rows=query, cols=key
            A_b = A[bidx].detach().cpu().numpy()  # [H, T, T]
            for h in range(n_heads):
                ax = axes[si, h]
                ax.imshow(A_b[h], interpolation="nearest", aspect="auto")
                ax.set_xticks([]); ax.set_yticks([])
                if si == 0:
                    ax.set_title(f"H{h}", fontsize=9)
                # Optional: mark CLS row/col
                # ax.axhline(y=0, linewidth=0.5)
                # ax.axvline(x=0, linewidth=0.5)

        fig.suptitle(f"Token×Token attention – Layer {li}", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{prefix}tok_tok_layer{li}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

def plot_cls_attention_maps(layer_to_attn: Dict[int, torch.Tensor], sample_indices: List[int], layers_to_plot: List[int], outdir: str, prefix: str):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for li in layers_to_plot:
        if li not in layer_to_attn:
            continue
        attn = layer_to_attn[li]
        B, H, T, _ = attn.shape
        grid = get_token_grid_size(T)
        max_heads = min(H, 8)
        max_samples = min(len(sample_indices), 4)
        fig, axes = plt.subplots(max_samples, max_heads, figsize=(2.2*max_heads, 2.2*max_samples))
        axes = np.atleast_2d(axes)
        for si, bidx in enumerate(sample_indices[:max_samples]):
            cls_attn = attn[bidx, :, 0, 1:].detach().cpu().numpy()
            if grid is not None:
                cls_attn = cls_attn.reshape(H, grid, grid)
            for h in range(max_heads):
                ax = axes[si, h]
                im = ax.imshow(cls_attn[h], interpolation='nearest')
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"H{h}", fontsize=8)
        fig.suptitle(f"CLS attention maps – Layer {li}")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{prefix}cls_maps_layer{li}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

@torch.no_grad()
def evaluate_model(model, data_loader, device, max_batches: int, mc_samples: int=0):
    capt = AttnCapture(model, debug=True, max_prints_per_layer=2)
    set_eval_with_dropout(model, enable_mc=(mc_samples > 0))

    ce_losses = []
    probs_all = []
    labels_all = []

    aggregated: Dict[str, List[np.ndarray]] = {"entropy": [], "cls_focus": [], "avg_dist": [], "avg_dist2d": []}
    saved_layer_to_attn = None

    def forward_once(images):
        logits = model(images)
        return logits

    batches_run = 0
    for it, (images, targets) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        print(f"Processing batch {it+1} ...", end='\r')
        print("images shape:", images.shape )
        print("targets shape:", targets.shape )
        if mc_samples > 0:
            ps = []
            for _ in range(mc_samples):
                logits = forward_once(images)
                ps.append(F.softmax(logits, dim=-1))
            probs = torch.stack(ps, dim=0).mean(dim=0)
            logits_mean = probs.log()
        else:
            logits = forward_once(images)
            probs = F.softmax(logits, dim=-1)
            logits_mean = logits.log_softmax(dim=-1)
        print("logits shape:", logits.shape )
        ce = F.nll_loss(logits_mean, targets, reduction='none')
        ce_losses.append(ce.detach().cpu().numpy())
        probs_all.append(probs.detach().cpu().numpy())
        labels_all.append(targets.detach().cpu().numpy())

        layer_to_attn = capt.collect_batch()
        if not layer_to_attn:
            print("[HOOK-ERR] No attention captured this batch. Printing per-layer fire counts so far...")
            capt.debug_summary()
        print("Attention shapes by layer:", {k: v.shape for k, v in layer_to_attn.items()})

        if saved_layer_to_attn is None:
            saved_layer_to_attn = {k: v.clone().cpu() for k, v in layer_to_attn.items()}

        m = aggregate_metrics(layer_to_attn)
        for k in aggregated:
            aggregated[k].append(m[k])

        batches_run += 1
        if batches_run >= max_batches:
            break

    metrics = {}
    for k, lst in aggregated.items():
        if lst:
            metrics[k] = np.mean(np.stack(lst, axis=0), axis=0)
        else:
            metrics[k] = None

    labels = np.concatenate(labels_all, axis=0) if labels_all else np.array([])
    probs_np = np.concatenate(probs_all, axis=0) if probs_all else np.array([])
    acc = float((probs_np.argmax(axis=1) == labels).mean()) if probs_np.size else float('nan')
    nll = float(-np.log(np.maximum(probs_np[np.arange(len(labels)), labels], 1e-12)).mean()) if probs_np.size else float('nan')
    ece = ece_score(probs_np, labels) if probs_np.size else float('nan')
    brier = brier_score(probs_np, labels) if probs_np.size else float('nan')

    capt.remove()

    return saved_layer_to_attn, metrics, {"acc": acc, "nll": nll, "ece": ece, "brier": brier}

def main():
    p = argparse.ArgumentParser(description="Attention/head comparison for ViT/DeiT models.")
    p.add_argument("--data-set", default="IMNET", choices=["CIFAR10","CIFAR100","IMNET","INAT","INAT19"])
    p.add_argument("--data-path", required=True, type=str)
    p.add_argument("--batch-size", default=64, type=int)
    p.add_argument("--num-workers", default=4, type=int)
    p.add_argument("--model", default="deit_tiny_patch16_224", type=str)
    p.add_argument("--ckpt-a", required=True, type=str)
    p.add_argument("--ckpt-b", default=None, type=str)
    p.add_argument("--label-a", default="Model A", type=str)
    p.add_argument("--label-b", default="Model B", type=str)
    p.add_argument("--drop-rate", default=0.0, type=float)
    p.add_argument("--drop-path", default=0.0, type=float)
    p.add_argument("--num-batches", default=4, type=int)
    p.add_argument("--mc-samples", default=0, type=int)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--outdir", default="attn_figs", type=str)
    p.add_argument("--cls-layers", nargs="*", type=int, default=[0, 2, 4])



    p.add_argument('--input-size', default=224, type=int)
    p.add_argument('--color-jitter', type=float, default=0.0, metavar='PCT',
                    help='Color jitter factor (default: 0.0)')
    p.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". " + \
                            "(default: rand-m9-mstd0.5-inc1)'),
    p.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    p.add_argument('--reprob', type=float, default=0.0, metavar='PCT',
                        help='Random erase prob (default: 0.0)')
    p.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    p.add_argument('--recount', type=int, default=0,
                        help='Random erase count (default: 1)')
    p.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    p.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
# --- epochs/batch ---
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
    p.add_argument('--smoothing', default=0.0, type=float)
    p.add_argument('--mixup', default=0.0, type=float)
    p.add_argument('--cutmix', default=0.0, type=float)
    p.add_argument('--mixup-prob', default=0.0, type=float)
    p.add_argument('--mixup-switch-prob', default=0.0, type=float)
    p.add_argument('--mixup-mode', default='batch', type=str)
    p.add_argument('--drop_rate', default=0.0, type=float)         # DeiT uses no plain dropout
    p.add_argument('--drop_path', default=0.0, type=float)         # stochastic depth
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

    p.add_argument('--seed', default=0, type=int)
    p.add_argument('--pin-mem', action='store_true', default=True)
    p.add_argument('--output_dir', default='', type=str)
    p.add_argument('--experiment_name_baseline', default='simpletransformer', type=str, help='experiment name')
    p.add_argument('--experiment_name_output', default='ydrop', type=str, help='experiment name')
    # --- finetune checkpoint ---

    p.add_argument('--update_freq',type=int,default = 1,
                    help ='intermediate steps for conductance calculation')
    # --- distributed toggles (harmless if single-GPU) ---
    p.add_argument('--distributed', action='store_true', default=False)
    p.add_argument('--world_size', default=1, type=int)
    p.add_argument('--dist_url', default='env://', type=str)

    p.add_argument('--repeated-aug', action='store_true')
    p.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    p.set_defaults(repeated_aug=False)
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
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset_train, nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    def load_model(ckpt_path: str):
        model = create_model(args.model, pretrained=False, num_classes=nb_classes,
                             drop_rate=args.drop_rate, drop_path_rate=args.drop_path)
        model.to(device)
        model.eval()
        ckpt = torch.load(ckpt_path, map_location="cpu",weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        print(f"[load] {ckpt_path}\n  missing: {len(missing)} keys, unexpected: {len(unexpected)} keys")
        return model

    model_a = load_model(args.ckpt_a)
    toggle_fused_attention(model_a, False)
    model_b = load_model(args.ckpt_b) if args.ckpt_b else None
    if model_b is not None:
        toggle_fused_attention(model_b, False)

    print(f"Evaluating {args.label_a} ...")
    layer_attn_a, metrics_a, scalars_a = evaluate_model(model_a, loader, device, args.num_batches, args.mc_samples)
    if model_b is not None:
        print(f"Evaluating {args.label_b} ...")
        layer_attn_b, metrics_b, scalars_b = evaluate_model(model_b, loader, device, args.num_batches, args.mc_samples)
    else:
        layer_attn_b, metrics_b, scalars_b = None, None, None

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(args.outdir, "scalars.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label","acc","nll","ece","brier"])
        w.writerow([args.label_a, scalars_a["acc"], scalars_a["nll"], scalars_a["ece"], scalars_a["brier"]])
        if scalars_b is not None:
            w.writerow([args.label_b, scalars_b["acc"], scalars_b["nll"], scalars_b["ece"], scalars_b["brier"]])

    plot_metric_heatmaps(metrics_a, args.label_a, args.outdir, metrics_b, args.label_b, prefix="head_metrics_")

    # For model A
    if layer_attn_a is not None:
        sample_indices = list(range(min(4, layer_attn_a[next(iter(layer_attn_a))].shape[0])))
        plot_cls_attention_maps(layer_attn_a, sample_indices, args.cls_layers, args.outdir,
                                prefix=f"{args.label_a.replace(' ','_')}_")
        plot_token_token_maps(layer_attn_a, sample_indices, args.cls_layers, args.outdir,
                            prefix=f"{args.label_a.replace(' ','_')}_", max_heads=8)

    # For model B
    if layer_attn_b is not None:
        sample_indices = list(range(min(4, layer_attn_b[next(iter(layer_attn_b))].shape[0])))
        plot_cls_attention_maps(layer_attn_b, sample_indices, args.cls_layers, args.outdir,
                                prefix=f"{args.label_b.replace(' ','_')}_")
        plot_token_token_maps(layer_attn_b, sample_indices, args.cls_layers, args.outdir,
                            prefix=f"{args.label_b.replace(' ','_')}_", max_heads=8)

    print("Done. Figures saved to:", args.outdir)

if __name__ == "__main__":
    main()
