#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CIFAR-100 training with M1 / M2 architectures (Krizhevsky-style 3×conv stem).
Implements:
- M1: FC head with [2048, 2048] on top of conv stem
- M2: FC head with [4096, 4096, 4096] on top of conv stem
- Y-Drop placeholder applied ONLY to fully-connected layers (as requested)
- CIFAR-100 data pipeline with standard augmentation (random crop + flip)
- 25% train->val split, early stopping (patience=10), 5 runs averaging by default
- SGD + momentum=0.9, StepLR for M1/M2, batch size 64 by default

Note on Y-Drop:
The paper snippet specifies "apply Y-Drop only to the fully connected layers."
Because the exact Y-Drop algorithmic details weren't provided here, this script
includes a YDrop module that currently behaves like nn.Dropout (same interface),
limited to FC layers. Swap in your actual Y-Drop implementation inside YDrop.
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from updated_transformer.dynamic_dropout import MyDropout  # Your custom dropout
from updated_transformer.pruning_indices import calculate_scores,update_dropout_masks
# ---------------------------
# Utilities
# ---------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make results a bit more deterministic (may slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def accuracy(topk_logits: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[torch.Tensor]:
    """Compute top-k accuracy for specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = topk_logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# ---------------------------
# Y-Drop placeholder
# ---------------------------

# ---------------------------
# Model definitions: Conv stem + FC heads (M1 / M2)
# ---------------------------

class ConvStem(nn.Module):
    """
    3 conv layers with channels [96, 128, 256], each 5x5, stride=1, padding=2 to preserve spatial dims.
    Each conv is followed by BN, ReLU, and 3x3 max pool stride=2.
    Input: 3x32x32
    Output: 256x3x3 (assuming 32->15->7->3 after 3 pools)
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 32 -> 15

            nn.Conv2d(96, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 15 -> 7

            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 7 -> 3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class MHead(nn.Module):
    """
    Fully-connected head with optional YDrop/Dropout between layers.
    regularizer_type: 'ydrop' | 'dropout' | 'none'
    """
    def __init__(self, in_features: int, hidden_dims: List[int], num_classes: int,
                 regularizer_type: str = 'dropout', dropout_p: float = 0.5,
                 mask_type: str = "robust_logistic",
                 # expose MyDropout knobs if you want
                 elasticity: float = 0.001, tied_layer=None, scaler: float = 1.0,
                 transformer_mean: bool = False, rescaling_type=None):
        super().__init__()
        layers: List[nn.Module] = []
        dims = [in_features] + hidden_dims

        def reg_factory():
            if regularizer_type == 'ydrop':
                return MyDropout(
                    elasticity=elasticity, p=dropout_p, tied_layer=tied_layer,
                    mask_type=mask_type, scaler=scaler,
                    transformer_mean=transformer_mean, rescaling_type=rescaling_type
                )
            elif regularizer_type == 'dropout':
                return nn.Dropout(p=dropout_p)
            else:
                return None

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            reg = reg_factory()
            if reg is not None:
                layers.append(reg)

        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



class M1(nn.Module):
    """
    M1: ConvStem + FC [2048, 2048] -> num_classes
    """
    def __init__(self, num_classes: int = 100, regularizer_type: str = 'ydrop', dropout_p: float = 0.5,
                 mask_type="robust_logistic"):
        super().__init__()
        self.stem = ConvStem()
        # Infer flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            out = self.stem(dummy)
            flat_dim = out.view(1, -1).shape[1]
        self.flatten = nn.Flatten()
        self.head = MHead(flat_dim, [2048, 2048], num_classes,
                          regularizer_type=regularizer_type, dropout_p=dropout_p, mask_type=mask_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.flatten(x)
        x = self.head(x)
        return x


class M2(nn.Module):
    """
    M2: ConvStem + FC [4096, 4096, 4096] -> num_classes
    """
    def __init__(self, num_classes: int = 100, regularizer_type: str = 'ydrop', dropout_p: float = 0.5,
                 mask_type="robust_logistic"):
        super().__init__()
        self.stem = ConvStem()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            out = self.stem(dummy)
            flat_dim = out.view(1, -1).shape[1]
        self.flatten = nn.Flatten()
        self.head = MHead(flat_dim, [4096, 4096, 4096], num_classes,
                          regularizer_type=regularizer_type, dropout_p=dropout_p,mask_type=mask_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.flatten(x)
        x = self.head(x)
        return x


# ---------------------------
# Data
# ---------------------------

def get_cifar100_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    val_split: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns train/val/test loaders for CIFAR-100.
    Uses 25% of training set for validation (val_split=0.25).
    """
    # CIFAR-100 mean/std (widely used estimates)
    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2762)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full_train = datasets.CIFAR100(root=data_dir, train=True, transform=train_tf, download=True)
    test_set = datasets.CIFAR100(root=data_dir, train=False, transform=test_tf, download=True)

    # Build a deterministic split
    n_train = len(full_train)  # 50,000
    n_val = int(n_train * val_split)
    n_subtrain = n_train - n_val
    g = torch.Generator().manual_seed(seed)
    subtrain_set, val_set = random_split(full_train, [n_subtrain, n_val], generator=g)

    train_loader = DataLoader(subtrain_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


# ---------------------------
# Training / Evaluation
# ---------------------------

@dataclass
class EarlyStopper:
    patience: int
    best_loss: float = float("inf")
    epochs_no_improve: int = 0
    best_state: Optional[dict] = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """
        Returns True if we should stop early.
        """
        if val_loss < self.best_loss - 1e-9:
            self.best_loss = val_loss
            self.epochs_no_improve = 0
            self.best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            return False
        else:
            self.epochs_no_improve += 1
            return self.epochs_no_improve >= self.patience


def train_one_epoch(model, loader, optimizer, device, scaler=None,selected_layers=None,drop_list=None,ydrop =False) -> Tuple[float, float]:
    model.train()
    loss_meter, acc_meter = 0.0, 0.0
    counter = 0
    for images, targets in loader:
        counter+=1
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        if ydrop is True:
            k = min(12, images.size(0))
            idx = torch.randperm(images.size(0), device=images.device)[:k]
            sub_images = images.index_select(0, idx)
            sub_targets = targets.index_select(0, idx)
            next_batches = [(sub_images, sub_targets)]
            
        optimizer.zero_grad(set_to_none=True)

        # Mixed precision if scaler is provided
        if ydrop is True:
            scores,_ = calculate_scores(model, next_batches,scoring_type ="Conductance",mode =None,selected_layers=selected_layers,device=device)
            if counter % 300 ==0:
                for i,score in scores.items():
                    print("Score shape layer {}: {}".format(i, score.shape))
                    print("Score mean layer {}: {}".format(i, score.mean()))
                    print("Score layer {}: {}".format(i, score.std()))
            update_dropout_masks(model = model,scores = scores,drop_list = drop_list,min_dropout =0.0,
                                    stats = False,alt_attention_cond=False)
            if counter % 300 ==0:
                for i,drop in enumerate(drop_list):
                    print("Dropout prev mean  {}: {}".format(i, drop.previous.mean()))
                    print("Dropout scaling mean {}: {}".format(i, drop.scaling.mean()))
                    
                    print("Dropout mask layer {}: {}".format(i, drop.previous))
                    print("Dropout mask layer {}: {}".format(i, drop.scaling))
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss = F.cross_entropy(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()

        acc1 = accuracy(logits, targets, topk=(1,))[0].item()
        loss_meter += loss.item() * images.size(0)
        acc_meter += acc1 * images.size(0)

    n = len(loader.dataset)
    return loss_meter / n, acc_meter / n


@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, float, float]:
    model.eval()
    loss_meter, acc1_meter, acc5_meter = 0.0, 0.0, 0.0
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, targets)
        top1, top5 = accuracy(logits, targets, topk=(1, 5))
        loss_meter += loss.item() * images.size(0)
        acc1_meter += top1.item() * images.size(0)
        acc5_meter += top5.item() * images.size(0)
    n = len(loader.dataset)
    return loss_meter / n, acc1_meter / n, acc5_meter / n


def build_model(name: str, num_classes: int, regularizer: str, dropout_p: float, mask_type: str) -> nn.Module:
    name = name.upper()
    if name == "M1":
        return M1(num_classes=num_classes, regularizer_type=regularizer, dropout_p=dropout_p, mask_type=mask_type)
    elif name == "M2":
        return M2(num_classes=num_classes, regularizer_type=regularizer, dropout_p=dropout_p, mask_type=mask_type)
    else:
        raise ValueError(f"Unknown model name: {name}")

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="CIFAR-100 training with M1/M2 models (PyTorch)")
    # Core setup
    parser.add_argument("--data-dir", type=str, default="./data", help="Dataset root directory")
    parser.add_argument("--model", type=str, choices=["M1", "M2"], default="M1",
                        help="Model architecture (default: M1)")
    parser.add_argument("--epochs", type=int, default=200, help="Max epochs (early stopping enabled)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs to average (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--val-split", type=float, default=0.25, help="Validation split ratio")

    # Optimization
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum (default: 0.9)")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay (default: 5e-4)")
    parser.add_argument("--step-size", type=int, default=60, help="StepLR step_size (epochs)")
    parser.add_argument("--gamma", type=float, default=0.1, help="StepLR gamma")

    # Regularization
    parser.add_argument("--regularizer", type=str, choices=["ydrop", "dropout", "none"],
                        default="ydrop", help="Regularizer for FC layers (default: ydrop)")
    parser.add_argument("--dropout-p", type=float, default=0.5,
                        help="Dropout/Y-Drop probability (default: 0.5; tuned in [0.1,0.6])")
    parser.add_argument('--annealing_factor', type=float, default=5,
                        help='Annealing factor for custom dropout')
    parser.add_argument('--mask_type', default='sigmoid', type=str, help='Type of mask for dropout')

    # Early stopping
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")

    # Misc
    parser.add_argument("--out-dir", type=str, default="./outputs", help="Directory to save checkpoints/logs")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training (CUDA only)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("===== Configuration =====")
    print(json.dumps(vars(args), indent=2))
    print("=========================")

    # Fixed split seed to keep the same train/val across runs
    set_seed(args.seed)
    train_loader, val_loader, test_loader = get_cifar100_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
    )

    all_run_metrics = []
    for run_idx in range(args.runs):
        run_seed = args.seed + run_idx
        set_seed(run_seed)

        model = build_model(args.model, num_classes=100,
                    regularizer=args.regularizer, dropout_p=args.dropout_p,
                    mask_type=args.mask_type).to(device)
        model.n_steps =5
        drop_list =[]
        selected_layers = []
        if args.regularizer == 'ydrop':
            # for m in model.head.net:
            #     if isinstance(m, nn.ReLU):
            #         selected_layers.append(m)
            #     # Only MyDropout supports use_ydrop()/use_normal_dropout()
            #     if isinstance(m, MyDropout):
            #         drop_list.append(m)
            selected_layers = [model.head.net[1], model.head.net[4]]
            drop_list = [model.head.net[2], model.head.net[5]]
           # selected_layers = selected_layers[:-1]  # Exclude final classifier
            print(drop_list)
            print(selected_layers)
            #print(len(model.head.net))

        n_params = count_params(model)
        print(f"\nRun {run_idx+1}/{args.runs} | Model: {args.model} | Params: {n_params/1e6:.2f}M")

        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

        scaler = torch.amp.GradScaler() if (args.amp and device.type == "cuda") else None

        stopper = EarlyStopper(patience=args.patience)
        best_epoch = -1

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            check = False
            if args.regularizer == 'ydrop' and epoch >= args.annealing_factor:
                for drop in drop_list:
                    drop.use_ydrop()
                check = True
            elif args.regularizer == 'ydrop':
                for drop in drop_list:
                    drop.use_normal_dropout()
                check = False
            train_loss, train_acc1 = train_one_epoch(model, train_loader, optimizer, device, scaler,selected_layers,drop_list,ydrop=check)
            val_loss, val_acc1, val_acc5 = evaluate(model, val_loader, device)
            scheduler.step()
            dt = time.time() - t0

            print(f"Epoch {epoch:03d} | "
                  f"Train Loss {train_loss:.4f} Acc@1 {train_acc1:.2f}% | "
                  f"Val Loss {val_loss:.4f} Acc@1 {val_acc1:.2f}% Acc@5 {val_acc5:.2f}% | "
                  f"{dt:.1f}s")

            if stopper.step(val_loss, model):
                print(f"Early stopping triggered at epoch {epoch}. Best val loss: {stopper.best_loss:.4f}")
                break
            else:
                best_epoch = epoch

        # Load best weights (by val loss)
        if stopper.best_state is not None:
            model.load_state_dict(stopper.best_state)

        # Evaluate on test
        test_loss, test_acc1, test_acc5 = evaluate(model, test_loader, device)
        print(f"[Run {run_idx+1}] Test Loss {test_loss:.4f} | Test Acc@1 {test_acc1:.2f}% | Acc@5 {test_acc5:.2f}%")

        # Save checkpoint and metadata
        ckpt_path = os.path.join(
            args.out_dir,
            f"cifar100_{args.model}_run{run_idx+1}_{args.regularizer}.pt"
        )
        torch.save({
            "model_state": model.state_dict(),
            "config": vars(args),
            "test_metrics": {"loss": test_loss, "acc1": test_acc1, "acc5": test_acc5},
            "best_val_loss": stopper.best_loss,
            "best_epoch": best_epoch,
            "params_millions": n_params / 1e6,
            "seed": run_seed,
        }, ckpt_path)
        print(f"Saved checkpoint to: {ckpt_path}")

        all_run_metrics.append((test_loss, test_acc1, test_acc5))

    # Aggregate over runs
    if args.runs > 1:
        import numpy as np
        arr = np.array(all_run_metrics)  # shape [runs, 3]
        mean = arr.mean(0)
        std = arr.std(0)
        print("\n===== Averaged over runs =====")
        print(f"Test Loss: {mean[0]:.4f} ± {std[0]:.4f}")
        print(f"Test Acc@1: {mean[1]:.2f}% ± {std[1]:.2f}%")
        print(f"Test Acc@5: {mean[2]:.2f}% ± {std[2]:.2f}%")
        # Save summary
        with open(os.path.join(args.out_dir, f"summary_{args.model}_{args.regularizer}.json"), "w") as f:
            json.dump({
                "runs": args.runs,
                "mean": {"loss": float(mean[0]), "acc1": float(mean[1]), "acc5": float(mean[2])},
                "std": {"loss": float(std[0]), "acc1": float(std[1]), "acc5": float(std[2])},
                "config": vars(args),
            }, f, indent=2)


if __name__ == "__main__":
    main()
