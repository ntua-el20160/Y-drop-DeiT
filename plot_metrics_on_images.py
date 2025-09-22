#!/usr/bin/env python3
"""
Attribution & Visualization script

- Supports multiple checkpoints: --resumes ckptA.pth ckptB.pth ...
  (falls back to --resume for a single path)
- Attribution methods: --attr {Conductance, Conductance_alt, LRP}
- Two outputs per checkpoint:
    1) Top % of pixels (absolute or signed-positive)
    2) Top % of patches (by mean |attr|)
- Pick dataset image by index: --img-index N
- Save directory: --save-dir DIR
"""

import argparse
import os
from pathlib import Path
from timm.utils import NativeScaler, get_state_dict, ModelEma

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # headless saving
import matplotlib.pyplot as plt
from PIL import Image

import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100

from timm.models import create_model
import models
import utils

# Your custom modules (as per your repo)
from evaluate_gradients.MultiLayerConductance import MultiLayerConductance
#from captum.attr import LRP


# ---------------------------
# CLI
# ---------------------------
def get_args_parser():
    parser = argparse.ArgumentParser("Attribution visualizer", add_help=True)

    # Data / basic
    parser.add_argument("--data-path", type=str, required=True,
                        help="Dataset root path")
    parser.add_argument("--data-set", type=str, default="CIFAR10",
                        choices=["CIFAR10", "CIFAR100"],
                        help="Dataset to use (for label count)")
    parser.add_argument("--img-index", type=int, default=1,
                        help="Dataset index to visualize")
    parser.add_argument("--input-size", type=int, default=224,
                        help="Model input size for resizing/cropping")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")

    # Models / checkpoints
    parser.add_argument("--model", default="deit_tiny_patch16_224", type=str,
                        help="timm model name (your ViT variant)")
    parser.add_argument("--resume", default="", type=str,
                        help="Single checkpoint path")
    parser.add_argument("--resumes", nargs="+", default=None,
                        help="One or more checkpoint paths; overrides --resume if given")
    
    # ViT creation args you use elsewhere (kept minimal but extensible)
    parser.add_argument("--drop_rate", type=float, default=0.1)
    parser.add_argument("--drop_path", type=float, default=0.0)
    parser.add_argument("--drop_block", type=float, default=None)
    parser.add_argument("--ydrop", action="store_true", default=False)
    parser.add_argument("--mask_type", type=str, default="sigmoid")
    parser.add_argument("--elasticity", type=float, default=0.01)
    parser.add_argument("--scaler", type=float, default=1.0)
    parser.add_argument("--n_steps", type=int, default=5)
    parser.add_argument("--transformer_mean", action="store_true", default=False)
    parser.add_argument("--rescaling_type", type=str, default=None,
                        choices=[None, "linear", "piecewise", "power_law"])

    # Attribution
    parser.add_argument("--attr", choices=["Conductance", "Conductance_alt", "LRP"],
                        default="Conductance", help="Attribution method to use")
    parser.add_argument("--abs-pixels", action="store_true", default=False,
                        help="Select top-% pixels by |attr|; otherwise signed-positive only")
    parser.add_argument("--top-pct-pixels", type=float, default=5.0,
                        help="Percent of pixels for pixel overlay")
    parser.add_argument("--top-pct-patches", type=float, default=10.0,
                        help="Percent of patches for patch overlay")

    # Output
    parser.add_argument("--save-dir", type=str, default="",
                        help="Directory to save images (created if missing)")

    # Repro
    parser.add_argument("--seed", type=int, default=0)

    return parser


# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(data_set: str, data_path: str):
    if data_set == "CIFAR10":
        ds = CIFAR10(root=data_path, train=True, download=False, transform=None)
        num_classes = 10
    elif data_set == "CIFAR100":
        ds = CIFAR100(root=data_path, train=True, download=False, transform=None)
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {data_set}")
    return ds, num_classes


def build_model(args, num_classes: int, device: torch.device):
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=num_classes,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        ydrop=args.ydrop,
        mask_type=args.mask_type,
        elasticity=args.elasticity,
        scaler=args.scaler,
        n_steps=args.n_steps,
        transformer_mean=False,
        rescaling_type=args.rescaling_type,
    ).to(device)
    return model


def load_checkpoint_into_model(model: nn.Module, ckpt_path: str):
    if ckpt_path.startswith("https://") or ckpt_path.startswith("http://"):
        checkpoint = torch.hub.load_state_dict_from_url(
            ckpt_path, map_location="cpu", check_hash=True
        )
    else:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    return checkpoint


def prepare_selected_layers_if_present(model: nn.Module):
    # Your repo uses `model.selected_layers` and `block.drop_list`.
    # We mirror your earlier mapping, but guard for availability.
    if hasattr(model, "blocks"):
        blocks = model.blocks
        if hasattr(model, "selected_layers"):
            for i, block in enumerate(blocks):
                if hasattr(block, "norm2"):
                    model.selected_layers[i * 4 + 1] = block.norm2
                if hasattr(block, "drop_list"):
                    for drop in block.drop_list:
                        if hasattr(drop, "tied_layer"):
                            drop.tied_layer = None
            for i in range(len(blocks) - 1):
                if hasattr(blocks[i + 1], "norm1"):
                    model.selected_layers[i * 4 + 3] = blocks[i + 1].norm1
    model.eval()


def to_model_input(img_pil: Image.Image, size: int, device: torch.device):
    tf = T.Compose([
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    inp = tf(img_pil).unsqueeze(0).to(device).requires_grad_(True)
    return inp


def compute_input_attr(args, model: nn.Module, inp: torch.Tensor, label: int):
    # Conductance / Conductance_alt via MultiLayerConductance
    if args.attr == "Conductance":
        cond = MultiLayerConductance(model, model.selected_layers)
    elif args.attr == "Conductance_alt":
        cond = MultiLayerConductance(model.crit_for, model.selected_layers)
    else:
        raise ValueError(f"Unknown attr method: {args.attr}")
    baseline = torch.zeros_like(inp)
    layer_attr, input_attr, delta = cond.attribute(
        inp,
        baselines=baseline,
        target=label,
        n_steps=args.n_steps,
        return_convergence_delta=True,
        return_input_attributions=True,
    )
    return input_attr  # list with one tensor [1,C,H,W]
    # LRP
    # elif args.attr == "LRP":
    #     lrp = LRP(model)
    #     input_attr = lrp.attribute(inp, target=label)
    #     return [input_attr]
    


def pixel_overlay(orig_rgb: np.ndarray,
                  attr_hw: np.ndarray,
                  top_pct: float,
                  use_abs: bool,
                  title_suffix: str = ""):
    """
    orig_rgb: [H,W,3] in [0,1]
    attr_hw: [H,W] signed attribution
    """
    H, W = attr_hw.shape
    score = np.abs(attr_hw) if use_abs else np.clip(attr_hw, a_min=0, a_max=None)
    cutoff = np.percentile(score, 100.0 - top_pct)
    mask = score >= cutoff

    overlay = np.zeros((H, W, 4), dtype=np.float32)
    overlay[..., 0] = 1.0          # red
    overlay[..., 3] = mask.astype(float) * 0.6

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(orig_rgb)
    ax.imshow(overlay, interpolation="none")
    ax.set_title(f"Top {top_pct:.1f}% Pixels{title_suffix}")
    ax.axis("off")
    return fig


def patch_overlay(orig_rgb: np.ndarray,
                  abs_map: np.ndarray,
                  patch_size: int,
                  top_pct: float):
    """
    abs_map: [H,W] in [0,1]
    """
    H, W = abs_map.shape
    n_h, n_w = H // patch_size, W // patch_size
    HH, WW = n_h * patch_size, n_w * patch_size
    if HH == 0 or WW == 0:
        raise ValueError(f"Patch size {patch_size} too large for map {H}x{W}")
    pm = abs_map[:HH, :WW].reshape(n_h, patch_size, n_w, patch_size).mean(axis=(1, 3))
    cutoff = np.percentile(pm, 100.0 - top_pct)
    patch_mask = pm >= cutoff

    mask = np.zeros((HH, WW), dtype=bool)
    for i in range(n_h):
        for j in range(n_w):
            if patch_mask[i, j]:
                y0, y1 = i * patch_size, (i + 1) * patch_size
                x0, x1 = j * patch_size, (j + 1) * patch_size
                mask[y0:y1, x0:x1] = True

    overlay = np.zeros((H, W, 4), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 3] = 0.0
    overlay[:HH, :WW, 3] = mask.astype(float) * 0.6

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(orig_rgb)
    ax.imshow(overlay, interpolation="none")
    ax.set_title(f"Top {top_pct:.1f}% Patches (mean |attr|)")
    ax.axis("off")
    return fig


def save_fig(fig, path: str):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------------------------
# Main
# ---------------------------
def main(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    # Dataset & pick image
    ds, num_classes = load_dataset(args.data_set, args.data_path)
    idx = max(0, min(args.img_index, len(ds) - 1))
    img_sample, label_sample = ds[idx]
    img_pil = T.ToPILImage()(img_sample) if isinstance(img_sample, torch.Tensor) else img_sample

    # Model input tensor and original RGB for plotting
    inp = to_model_input(img_pil, args.input_size, device)
    orig_rgb = np.array(img_pil.resize((args.input_size, args.input_size))).astype(np.float32) / 255.0
    label = int(label_sample)

    # Collect checkpoints
    resume_list = args.resumes if args.resumes is not None else ([args.resume] if args.resume else [])
    if not resume_list:
        raise RuntimeError("Please provide at least one checkpoint via --resume or --resumes")

    save_dir = args.save_dir if args.save_dir else "./attr_outputs"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for ckpt_path in resume_list:
        print(f"[ATTR] Loading checkpoint: {ckpt_path}")

        # Build fresh model per checkpoint (avoids any lingering state)
        model = build_model(args, num_classes=num_classes, device=device)
        _ = load_checkpoint_into_model(model, ckpt_path)
        prepare_selected_layers_if_present(model)

        # Attribution
        input_attr_list = compute_input_attr(args, model, inp, label)
        attr_tensor = input_attr_list[0].detach().squeeze(0).cpu()  # [C,H,W]

        # Collapse channels -> [H,W] signed
        attr_hw = attr_tensor.sum(0).numpy()
        abs_map = np.abs(attr_hw)
        abs_map /= (abs_map.max() + 1e-8)

        # Pixel overlay (top-X%)
        pix_suffix = " (abs)" if args.abs_pixels else " (signed +)"
        fig_pix = pixel_overlay(orig_rgb, attr_hw,
                                top_pct=args.top_pct_pixels,
                                use_abs=args.abs_pixels,
                                title_suffix=pix_suffix)

        # Patch overlay (top-Y% by mean |attr|)
        # Get patch size
        if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "patch_size"):
            ps = model.patch_embed.patch_size
            patch_size = ps if isinstance(ps, int) else int(ps[0])
        else:
            # Fallback (common ViT tiny/16)
            patch_size = 16
        fig_patch = patch_overlay(orig_rgb, abs_map,
                                  patch_size=patch_size,
                                  top_pct=args.top_pct_patches)

        # Save both
        stem = Path(ckpt_path).stem
        pix_name = f"{stem}_{args.attr}_top{int(args.top_pct_pixels)}pct_pixels{'_abs' if args.abs_pixels else '_signed'}.png"
        pat_name = f"{stem}_{args.attr}_top{int(args.top_pct_patches)}pct_patches.png"
        save_fig(fig_pix, os.path.join(save_dir, pix_name))
        save_fig(fig_patch, os.path.join(save_dir, pat_name))
        print(f"[ATTR] Saved:\n  - {os.path.join(save_dir, pix_name)}\n  - {os.path.join(save_dir, pat_name)}")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
