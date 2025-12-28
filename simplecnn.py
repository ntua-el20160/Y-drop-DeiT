import argparse
import datetime
import json
import os
import time
from pathlib import Path
import utils
import numpy as np
#CHANGE TO CHECK: import random
import random
from types import MethodType


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from datasets2 import build_dataset, create_subdataset
# Import your custom dropout module.
from updated_transformer.dynamic_dropout import MyDropout
from updated_transformer.plots import plot_epoch_statistics
from engine import train_one_epoch, evaluate
from timm.utils import NativeScaler
import copy
from typing import Iterable
from captum.attr import LayerConductance
from evaluate_gradients.MultiLayerConductance import MultiLayerConductance
from evaluate_gradients.MultiLayerSensitivity import MultiLayerSensitivity

# class CNN6_S1(nn.Module):
#     def __init__(self, num_classes=10, use_custom_dropout=True, elasticity=1.0, p=0.1, n_steps=5,
#                  mask_type = 'sigmoid',scaler = 1.0,rescaling_type = None):
        
#         super(CNN6_S1, self).__init__()
#         self.n_steps = n_steps
#         # Convolutional layers (unchanged)
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.bn1   = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.bn2   = nn.BatchNorm2d(64)
#         self.pool  = nn.MaxPool2d(2, 2)
        
#         # After two poolings: 32x32 -> 16x16 -> 8x8 with 64 channels, so flattened dim = 64*8*8 = 4096.
#         self.flatten = nn.Flatten()
        
#         # Fully connected layers for S1: 2×1024.
#         self.fc1 = nn.Linear(4096, 1024)
#         self.relu1 = nn.ReLU()    # Explicit ReLU for fc1
#         self.drop1 = nn.Dropout(p)  # Dropout for fc1
#         self.fc2 = nn.Linear(1024, 1024)
#         self.relu2 = nn.ReLU()    # Explicit ReLU for fc2
#         self.drop2 = nn.Dropout(p)  # Dropout for fc2

#         self.fc3 = nn.Linear(1024, num_classes)
        
#         # Use MyDropout for fc1 and fc2 outputs.
#         self.selected_layers = [self.fc1, self.fc2]
#         if use_custom_dropout:
#             self.drop_list = nn.ModuleList([
#                 MyDropout(elasticity=elasticity, p=p, tied_layer=layer, mask_type=mask_type, scaler=scaler,rescaling_type=rescaling_type)
#                 for layer in self.selected_layers
#             ])
#         else:
#             self.drop_list = nn.ModuleList([nn.Dropout(p) for _ in self.selected_layers])
#         self.scores ={}
        
#     def forward(self, x):
#         # Convolutional layers
#         x = self.conv1(x)
#         x = F.relu(self.bn1(x))
#         x = self.pool(x)
#         x = self.conv2(x)
#         x = F.relu(self.bn2(x))
#         x = self.pool(x)
        
#         # Flatten
#         x = self.flatten(x)  # Shape: [B, 4096]
        
#         # FC1 with dropout
#         x = self.fc1(x)
#         x = self.relu1(x)  # relu1
#         x = self.drop_list[0](x)
        
#         # FC2 with dropout
#         x = self.fc2(x)
#         x = self.relu2(x)  # relu2
#         x = self.drop_list[1](x)
        
#         # Output layer
#         x = self.fc3(x)
#         return x

#     def calculate_scores(self, batches: Iterable, device: torch.device,stats = True,
#                          scoring_type = "Conductance",noisy_score = False,noisy_dropout = False,
#                          min_dropout = 0.0,alt_attention_cond = False,sm = True) -> None:
#         # Create a detached copy of the model for IG computation.

#         self.eval()
#         orig_reqs = []
#         for p in self.parameters():
#             orig_reqs.append(p.requires_grad)
#             p.requires_grad_(False)
#         # Also make sure no stale weight‐grads are sitting around
#         self.zero_grad()  
        
#         # Initialize conductances for each layer
#         for i, _ in enumerate(self.selected_layers):
#             self.scores[f'drop_{i}'] = None
#         if scoring_type == "Conductance":
#             mlc = MultiLayerConductance(self, self.selected_layers)
#         elif scoring_type == "Sensitivity":
#             mlc = MultiLayerSensitivity(self, self.selected_layers)
#         else:
#             mlc = MultiLayerConductance(self, self.selected_layers)


#         for x, _ in batches:

#             x_captum = x.detach().clone().requires_grad_()
#             x_captum = x_captum.to(device, non_blocking=True)
#             baseline = torch.zeros_like(x_captum)

#             # Get model predictions
#             outputs = self(x_captum)
#             pred = outputs.argmax(dim=1)

#             captum_out = mlc.attribute(
#                 x_captum, baselines=baseline, target=pred,
#                 n_steps=self.n_steps,
#                 internal_batch_size=None,
#                 return_convergence_delta=False,
#                 attribute_to_layer_input=False,
#                 grad_kwargs={"retain_graph": False},
#             )
#             if isinstance(captum_out, list):
#                 captum_attrs = [t.detach() for t in captum_out]
#             elif isinstance(captum_out, tuple):
#                 captum_attrs = tuple(t.detach() for t in captum_out)
#             else:
#                 captum_attrs = [captum_out.detach()]
#             # Average out the conductance across the batch and add it
#             for i, score in enumerate(captum_attrs):
#                 #score_mean = score.mean(dim=0)
#                 #score_mean = score if scoring_type == "Sensitivity" else score.sum(dim=0)
#                 if scoring_type == "Sensitivity":
#                     score_mean = score
#                 elif sm:
#                     score_mean = score.sum(dim =0)
#                 else:
#                     score_mean = score.mean(dim=0)

#                 if self.scores[f'drop_{i}'] is None:
#                     # First time: initialize with the computed score_mean
#                     self.scores[f'drop_{i}'] = score_mean.clone()
#                     #print(f"Initialized scores for drop_{i} {score_mean.mean().item()}")
#                 else:
#                     # Accumulate the score_mean
#                     self.scores[f'drop_{i}'] += score_mean
#                     #print(f"Initialized scores for drop_{i} {score_mean.mean().item()}")
        
#         #update the masks based on the scores
#         for i, drop_layer in enumerate(self.drop_list):

#             score = self.scores[f'drop_{i}'] / float(len(batches))
            

#             if noisy_score:
                
#                 noise = (torch.rand_like(score) - 0.5) * 2 * (score.abs()+10**-4)*0.1#+-10% maximum
#                 mask = (torch.rand_like(score) < 0.3).float()

#                 score = score + (mask*noise)
#             #print(f"Drop {i} after division {score}")
#             drop_layer.update_dropout_masks(
#                 #CHANGE TO CHECK: diving by the number of batches to get the average score
#                 score,
#                 #model_clone.scores[f'drop_{i}'],
#                 stats=stats,
#                 noisy = noisy_dropout,
#                 min_dropout=min_dropout
#             )

#         for p, req in zip(self.parameters(), orig_reqs):
#             p.requires_grad_(req)
#         torch.cuda.empty_cache()

#         self.train()
    
#     def use_normal_dropout(self):
#         for drop in self.drop_list:
#             drop.use_normal_dropout()

#     def use_ydrop(self):
#         for drop in self.drop_list:
#             drop.use_ydrop()
#     def plot_aggregated_statistics(self, epoch_label, save_dir=None):
#         for i,_ in enumerate(self.drop_list):
#             self.drop_list[i].plot_aggregated_statistics(epoch_label+f"layer {i}", save_dir)
    
#     def plot_current_stats(self, epoch,batch_idx, save_dir):
#         for i,_ in enumerate(self.drop_list):
#             self.drop_list[i].plot_current_stats(epoch,batch_idx, save_dir, i,False)

#     def update_progression(self,save_dir):
#         for i,_ in enumerate(self.drop_list):
#             self.drop_list[i].update_progression(save_dir, label = f"layer{i}")
 
#     def plot_progression_statistics(self, save_dir=None,label =''):
#         for i,_ in enumerate(self.drop_list):
#             self.drop_list[i].plot_progression_statistics(save_dir,label =label + f" layer {i}")
#     def plot_random_node_histograms_scoring(self, epoch_label, save_dir=None):
#         for i,_ in enumerate(self.drop_list):
#             self.drop_list[i].plot_random_node_histograms_scoring(epoch_label+f"layer {i}", save_dir)
#     def plot_random_node_histograms_keep(self, epoch_label, save_dir=None):
#         for i,_ in enumerate(self.drop_list):
#             self.drop_list[i].plot_random_node_histograms_keep(epoch_label+f"layer {i}", save_dir)

#     def save_statistics(self, save_dir):
#         for i,_ in enumerate(self.drop_list):
#             self.drop_list[i].save_statistics(save_dir, layer_label = f"layer{i}")
#     def clear_progression(self):
#         for drop in self.drop_list:
#             drop.clear_progression()
    
import torch
import torch.nn as nn

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
# ---------------------------
# Shared 3-layer CNN backbone (M)
# ---------------------------
class CIFAR100_Backbone_M(nn.Module):
    """
    3 conv layers with 96, 128, 256 filters; 5x5 kernel, stride=1, padding=2.
    Each block: Conv -> BN -> ReLU -> MaxPool(3x3, stride=2).
    CIFAR (32x32): 32->16->8->4, flatten = 256*4*4 = 4096.
    """
    def __init__(self):
        super().__init__()
        # Block 1
        self.conv1   = nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1     = nn.BatchNorm2d(96)
        self.relu1   = nn.ReLU(inplace=True)
        self.pool1   = nn.MaxPool2d(kernel_size=3, stride=2)

        # Block 2
        self.conv2   = nn.Conv2d(96, 128, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2     = nn.BatchNorm2d(128)
        self.relu2   = nn.ReLU(inplace=True)
        self.pool2   = nn.MaxPool2d(kernel_size=3, stride=2)

        # Block 3
        self.conv3   = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn3     = nn.BatchNorm2d(256)
        self.relu3   = nn.ReLU(inplace=True)
        self.pool3   = nn.MaxPool2d(kernel_size=3, stride=2)

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.flatten(x)  # (N, 256*4*4=4096)
        return x


# ---------------------------
# M1: Backbone M + FC(2048) x 2 + classifier
# ---------------------------
class CIFAR100_M1(nn.Module):
    def __init__(self, num_classes=100, p=0.5):
        super().__init__()
        self.backbone = CIFAR100_Backbone_M()

        # Fully connected head (2 × 2048)
        self.fc1      = nn.Linear(256 * 3 * 3, 2048)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.drop1    = nn.Dropout(p)

        self.fc2      = nn.Linear(2048, 2048)
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.drop2    = nn.Dropout(p)

        self.fc_out   = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.backbone(x)            # -> 4096
        x = self.drop1(self.relu_fc1(self.fc1(x)))
        x = self.drop2(self.relu_fc2(self.fc2(x)))
        x = self.fc_out(x)
        return x


# ---------------------------
# M2: Backbone M + FC(4096) x 3 + classifier
# ---------------------------
class CIFAR100_M2(nn.Module):
    def __init__(self, num_classes=100, p=0.5):
        super().__init__()
        self.backbone = CIFAR100_Backbone_M()

        # Fully connected head (3 × 4096)
        self.fc1      = nn.Linear(256 * 3 * 3, 4096)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.drop1    = nn.Dropout(p)

        self.fc2      = nn.Linear(4096, 4096)
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.drop2    = nn.Dropout(p)

        self.fc3      = nn.Linear(4096, 4096)
        self.relu_fc3 = nn.ReLU(inplace=True)
        self.drop3    = nn.Dropout(p)

        self.fc_out   = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.backbone(x)            # -> 4096
        x = self.drop1(self.relu_fc1(self.fc1(x)))
        x = self.drop2(self.relu_fc2(self.fc2(x)))
        x = self.drop3(self.relu_fc3(self.fc3(x)))
        x = self.fc_out(x)
        return x

def get_args_parser():
    parser = argparse.ArgumentParser('SimpleCNNMLP Training Script', add_help=False)
    parser.add_argument('--batch-size', default=128, type=int, help='Batch size for training')
    parser.add_argument('--epochs', default=50, type=int, help='Number of training epochs')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--device', default='cuda', type=str, help='Device to use for training (cuda or cpu)')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--output_dir', default='./output', type=str, help='Directory to save checkpoints and logs')
    parser.add_argument('--ydrop', action='store_true', default=False,
                    help='Enable Y-Drop (MyDropout) by default')
    parser.add_argument('--no-ydrop', dest='ydrop', action='store_false',
                    help='Disable Y-Drop (MyDropout)')
    parser.set_defaults(ydrop=True)
    parser.add_argument('--data-path', default='/datasets01_101/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR10','CIFAR100', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    # Custom dropout hyperparameters.
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
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
    parser.add_argument('--experiment_name', default='simplecnnmlp', type=str, help='experiment name')
    parser.add_argument('--model', default='M1', type=str, help='Model to use')
    parser.add_argument('--plot_freq', default=5, type=int, help='plot frequency')
    parser.add_argument('--resume', default='', type=str, help='Path to checkpoint to resume training or load a model')
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
    parser.add_argument('--noisy_score', action='store_true', default=False,
                        help='Noise addition to score')
    parser.add_argument('--noisy_dropout', action='store_true', default=False,
                        help='Noise addition to dropout')
    parser.add_argument('--min_dropout', type=float, default=0.0,
                    help='Minimum allowed dropout rate')
    parser.add_argument('--max_dropout', type=float, default=0.5,
                    help='Maximum allowed dropout rate')
    parser.add_argument('--rescaling_type',choices=['linear','piecewise', 'power_law'], default=None, type =str,
                    help='Method to rescale the the limits of the dropout masks')
    parser.add_argument('--after_relu', action='store_true', default=False,
                        help='Apply dropout after ReLU activation')
    parser.add_argument('--conductance_batch_size', default=64, type=int, help='Batch size for conductance calculation')
    return parser

def main(args):
    # Set the random seed for reproducibility.
    # torch.manual_seed(args.seed)
    
    # device = torch.device(args.device)
    # seed = args.seed + utils.get_rank()
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    #CHANGE TO CHECK: how to set the random seed for random module
    seed = args.seed

    # # 1. Python built-in RNG
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
    device = torch.device(args.device)

    ############################################################

    # Data augmentation and normalization for CIFAR10.
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    
    # train_dataset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True,
    #                                              transform=transform_train)
    # test_dataset  = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True,
    #                                              transform=transform_test)
    
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
    #                                            shuffle=True, num_workers=4, pin_memory=True,drop_last=True)
    # test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
    #                                            shuffle=False, num_workers=4, pin_memory=True)
    if args.data_set == 'CIFAR100':
        DatasetClass = torchvision.datasets.CIFAR100
        num_classes = 100
    else:
        DatasetClass = torchvision.datasets.CIFAR10
        num_classes = 10

    train_dataset = DatasetClass(
        root=args.data_path, train=True, download=True,
        transform=transform_train
    )
    test_dataset = DatasetClass(
        root=args.data_path, train=False, download=True,
        transform=transform_test
    )

        
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True,drop_last=True)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                               shuffle=False, num_workers=4, pin_memory=True)
    


    #print(sub_dataset.shape())
    def preload_subdataset(subdataset):
        """
        Given a small subdataset (a torch.utils.data.Subset),
        load all (data, target) pairs into memory as a list.
        """
        cached = [subdataset[i] for i in range(len(subdataset))]
        return cached
    
    if args.sub_dataset == 'stratified':
        sub_dataset = create_subdataset(train_dataset, batch_size=args.batch_size, sub_factor=args.sub_factor, stratified=True)
        cached_subdataset = preload_subdataset(sub_dataset)
    elif args.sub_dataset == 'random':
        sub_dataset = create_subdataset(train_dataset, batch_size=args.batch_size, sub_factor=args.sub_factor, stratified=False)
        cached_subdataset = preload_subdataset(sub_dataset)
    else:
        cached_subdataset = None

    if args.model == 'M1':
        model = CIFAR100_M1(num_classes=num_classes, p=args.drop)
    elif args.model == 'M2':
        model = CIFAR100_M2(num_classes=num_classes, p=args.drop)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = model.to(device)
    if args.ydrop:
        # Replace standard Dropout with MyDropout in the model.
        if args.after_relu:
            selected_layers = [model.relu_fc1, model.relu_fc2] if args.model == 'M1' else [model.relu_fc1, model.relu_fc2, model.relu_fc3]
        else:
            selected_layers = [model.fc1, model.fc2] if args.model == 'M1' else [model.fc1, model.fc2, model.fc3]
        drop_list = []
        model.drop1 = MyDropout(elasticity=args.elasticity, p=args.drop, tied_layer=model.fc1, transformer_mean=False,
                                 mask_type=args.mask_type, scaler=args.scaler, rescaling_type=args.rescaling_type)
        model.drop2 = MyDropout(elasticity=args.elasticity, p=args.drop, tied_layer=model.fc2,transformer_mean=False,
                                 mask_type=args.mask_type, scaler=args.scaler, rescaling_type=args.rescaling_type)
        drop_list.append(model.drop1)
        drop_list.append(model.drop2)
        if args.model == 'M2':
            model.drop3 = MyDropout(elasticity=args.elasticity, p=args.drop, tied_layer=model.fc3, transformer_mean=False,
                                     mask_type=args.mask_type, scaler=args.scaler, rescaling_type=args.rescaling_type)
            drop_list.append(model.drop3)

        model.drop_list = drop_list
        model.selected_layers = selected_layers
        model.n_steps = args.n_steps

    model.to(device)

    # Set up optimizer and loss function.
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    model.criter = nn.CrossEntropyLoss(reduction='none')
    model.crit_for = MethodType(crit_for, model)
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

    # dummy_input = torch.randn(1, 3, args.input_size, args.input_size, device=device)
    # model(dummy_input)
    
    if args.resume:
        try:
            print(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device,weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"Resumed at epoch {start_epoch}")
            cumulative_train_time = checkpoint['train_time']
            best_acc = checkpoint['best_acc']
            history = checkpoint.get('history', {})
            best_loss = checkpoint.get('lowest_loss', float('inf'))
            best_epoch = checkpoint.get('best_epoch', 0)

            if history:
                for i, drop in enumerate(model.drop_list):
                    drop_history = history.get(f'drop{i}', {})
                    drop.progression_keep = drop_history.get('progression_keep', [])
                    drop.progression_scoring = drop_history.get('progression_scoring', [])
        except Exception as e:
            print(f"Warning: could not resume from '{args.resume}' ({e}). Starting from scratch.")
            start_epoch = 0
            cumulative_train_time = 0.0
            best_acc = 0.0
            history = {}
            best_loss = float('inf')
            best_epoch = 0
    else:
        start_epoch = 0
        cumulative_train_time = 0.0
        best_acc = 0.0
        history = {}
        best_loss = float('inf')
        best_epoch = 0
    # print("Start")
    # for drop in model.drop_list:
    #     print(f"scaling: {drop.scaling}")
    # print("Rest")
    # if args.update_scaling == 'increasing':

    #     update_freq = 1
    # else:
    #     update_freq = args.update_freq
    
    # effective_epochs = args.epochs - args.annealing_factor
    # step_size = round(effective_epochs / args.update_scaling_steps)
    # denom = max(1, args.update_scaling_steps - 1)    

    print("Start training")
    start_time = time.time()
    patience_counter = 0
    #initially normal dropout
    loss_scaler = NativeScaler()
    if args.ydrop:
        if hasattr(model, 'module'):
            for drop in model.module.drop_list:
                if isinstance(drop, MyDropout):
                    drop.use_normal_dropout()
            # model.module.use_normal_dropout()
        else:
            for drop in model.drop_list:
                if isinstance(drop, MyDropout):
                    drop.use_normal_dropout()
    check = False
    for epoch in range(start_epoch, args.epochs):
        stats = False
        start_time = time.time()

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
            

        # Train for one epoch.
        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            max_norm= None,
            model_ema=None,
            mixup_fn=None,
            check=check,
            update_freq=args.update_freq,
            update_batches=args.update_batches,
            tracker=None,
            update_data_loader = None,
            scoring_type = args.scoring_type,
            help_par = 0,
            noisy_dropout = args.noisy_dropout,
            min_dropout = args.min_dropout,
            mask_type = args.mask_type,
            ypath = False,
            conductance_batch_size =args.conductance_batch_size,
            mode = 'nothing',
            max_dropout = args.max_dropout,
        )

        epoch_time = time.time() - start_time
        cumulative_train_time += epoch_time

        
        # Evaluate on the test set.
        test_stats = evaluate(test_loader, model, device)
        test_acc = test_stats.get('acc1', 0.0)
        test_loss = test_stats.get('loss', 0.0)
        
        # print(f"Epoch {epoch+1}/{args.epochs}: Train Loss {train_stats['loss']:.4f}, "
        #       f"Test Acc {test_stats.get('acc1', 0):.2f}%, Epoch Time {epoch_time:.2f}s")
        
        # if check and stats:
        #     model.update_progression(output_dir / 'plots')
        #     model.save_statistics(epoch_dir)

        #     plot_epoch_statistics(output_dir, epoch+1, epoch_dir_image)
        #     model.plot_progression_statistics(output_dir / 'plots',label = "")
        #     #model.plot_aggregated_statistics(f'Epoch {epoch+1} ', epoch_dir)
        #     #model.plot_random_node_histograms_scoring(f'Epoch {epoch+1} ', epoch_dir)
        #     #model.plot_random_node_histograms_keep(f'Epoch {epoch+1} ', output_dir / 'plots')
            
        #     model.clear_progression()

        checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'test_acc': test_acc,

                # 'test_acc': best_acc,
                'train_stats': train_stats,
                'test_stats': test_stats,
                'train_time': cumulative_train_time,
                'best_acc': best_acc,
                'test_loss': test_loss,
                'lowest_loss': best_loss,  
                'best_epoch': best_epoch,
            }
        # if args.ydrop:
        #     checkpoint['history'] = {}
        #     for i, drop in enumerate(model.drop_list):
        #             checkpoint['history'][f'drop{i}'] = {
        #                 'progression_keep': drop.progression_keep,
        #                 'progression_scoring': drop.progression_scoring,
        #             }    
        if test_loss < best_loss:
            best_loss = test_loss
            checkpoint['lowest_loss'] = best_loss
            patience_counter = 0  # reset early stopping counter


        else:
            patience_counter += 1
                 
        # Save checkpoint if a new best accuracy is reached.
        if test_stats.get('acc1', 0) > best_acc:
            best_acc = test_stats.get('acc1', 0)
            best_epoch = epoch +1

            checkpoint['best_acc'] = best_acc
            checkpoint['best_epoch'] = best_epoch
            torch.save(checkpoint, output_dir /"best_model.pth")
     
        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss {train_stats['loss']:.4f}, "
              f"Test Acc {test_stats.get('acc1', 0):.2f}%, Epoch Time {epoch_time:.2f}s")
        
        torch.save(checkpoint, output_dir / "checkpoint.pth")

        # Log epoch statistics.
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

        with (output_dir/ "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")
    
    total_time_str = str(datetime.timedelta(seconds=int(cumulative_train_time)))
    print(f"Training complete. Best Test Accuracy: {best_acc:.2f}% at epoch {best_epoch}. Total training time: {total_time_str}")
        

    
    # Optionally update custom dropout hyperparameters for each dropout layer.
# Example usage:
if __name__ == '__main__':
    parser = argparse.ArgumentParser('SimpleCNNMLP Training Script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)