import argparse
import datetime
import json
import os
import time
from pathlib import Path
import utils
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from datasets import build_dataset
# Import your custom dropout module.
from updated_transformer.dynamic_dropout import MyDropout
from engine import train_one_epoch, evaluate
from timm.utils import NativeScaler
import copy
from typing import Iterable
from captum.attr import LayerConductance


class CNN6_S1(nn.Module):
    def __init__(self, num_classes=10, use_custom_dropout=True, elasticity=1.0, p=0.1, n_steps=5,mask_type = 'sigmoid',scaler = 1.0):
        
        super(CNN6_S1, self).__init__()
        self.n_steps = n_steps
        # Convolutional layers (unchanged)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2, 2)
        
        # After two poolings: 32x32 -> 16x16 -> 8x8 with 64 channels, so flattened dim = 64*8*8 = 4096.
        self.flatten = nn.Flatten()
        
        # Fully connected layers for S1: 2×1024.
        self.fc1 = nn.Linear(4096, 1024)
        self.relu1 = nn.ReLU()    # Explicit ReLU for fc1

        self.fc2 = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU()    # Explicit ReLU for fc2

        self.fc3 = nn.Linear(1024, num_classes)
        
        # Use MyDropout for fc1 and fc2 outputs.
        if use_custom_dropout:
            self.drop1 = MyDropout(elasticity=elasticity, p=p, num_channels=1024,mask_type = mask_type, scaler = scaler)
            self.drop2 = MyDropout(elasticity=elasticity, p=p, num_channels=1024,mask_type = mask_type, scaler = scaler)
        else:
            self.drop1 = nn.Dropout(p)
            self.drop2 = nn.Dropout(p)
        
        # Integrated gradients placeholders.
        self._saved = {}
        self._hooks = []
        self.conductance = {'fc1': None, 'fc2': None}
        
    def forward(self, x):
        # Convolutional layers.
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool(x)
        
        # Flatten.
        x = self.flatten(x)  # Shape: [B, 4096]
        
        # fc1: raw output, then ReLU, then dropout.
        pre_fc1 = self.fc1(x)
        # Save pre-activation if needed for integrated gradients:
        x1 = self.relu1(pre_fc1)  # This is the post-ReLU activation for fc1.

        if 'fc1' in self._saved:
            x1.retain_grad()
            self._saved['fc1'] = x1
        x1 = self.drop1(x1)
        
        # fc2: raw output, then ReLU, then dropout.
        pre_fc2 = self.fc2(x1)
        x2 = self.relu2(pre_fc2)  # Post-ReLU activation for fc2.
        if 'fc2' in self._saved:
            x2.retain_grad()
            self._saved['fc2'] = x2
        x2 = self.drop2(x2)
        
        # Final classification layer.
        out = self.fc3(x2)
        return out

    def split_images(self, x: torch.Tensor, n_steps: int = None) -> torch.Tensor:
        """
        Create interpolation paths from a baseline (zeros) to x.
        Returns a tensor of shape [B, n_steps+1, C, H, W] where B is the batch size.
        """
        if n_steps is None:
            n_steps = self.n_steps
        baseline = torch.zeros_like(x)  # shape: [B, C, H, W]
        alphas = torch.linspace(0, 1, steps=n_steps + 1, device=x.device).view(1, n_steps + 1, 1, 1, 1)
        x_exp = x.unsqueeze(1)          # shape: [B, 1, C, H, W]
        baseline_exp = baseline.unsqueeze(1)  # shape: [B, 1, C, H, W]
        interpolated = baseline_exp + alphas * (x_exp - baseline_exp)
        return interpolated  # shape: [B, n_steps+1, C, H, W]

    def save_output_gradients(self, x: torch.Tensor, n_steps: int):
        """
        Run forward pass with dropout disabled to capture outputs at:
          - after activation (fc1+act) and
          - after fc2.
        """
        self._saved = {'fc1': None, 'fc2': None}
        

        orig_drop1, orig_drop2 = self.drop1, self.drop2
        self.drop1 = nn.Identity()
        self.drop2 = nn.Identity()
        # def hook_fn(name):
        #     def hook(module, input, output):
        #         if isinstance(output, torch.Tensor):
        #             output.retain_grad()
        #         self._saved[name] = output
        #     return hook
        # h1 = self.fc1.register_forward_hook(hook_fn('fc1'))  
        # h2 = self.fc2.register_forward_hook(hook_fn('fc2'))
        def hook_fc1(module, input, output):
            if isinstance(output, torch.Tensor):
                output.retain_grad()
            self._saved['fc1'] = output
        h1 = self.relu1.register_forward_hook(hook_fc1)
        
        # Register a hook on the ReLU module for fc2.
        def hook_fc2(module, input, output):
            if isinstance(output, torch.Tensor):
                output.retain_grad()
            self._saved['fc2'] = output
        h2 = self.relu2.register_forward_hook(hook_fc2)
        
        self._hooks = [h1, h2]
        y= self.forward(x)
        self.drop1 = orig_drop1
        self.drop2 = orig_drop2
        return y
    def calculate_conductance(self, n_steps: int, n_batches: int = None):
        """
        Compute integrated gradients for the saved outputs and update the dropout masks.
        """
        for key in ['fc1', 'fc2']:
            act = self._saved.get(key)
            if act is None:
                print(f"[calculate_conductance] No saved activation for {key}.")
                continue
            if act.grad is None:
                print(f"[calculate_conductance] No gradient for {key}.")
                continue
            # Print out basic stats for debugging.
            #print(f"[calculate_conductance] {key}: activation mean = {act.mean().item():.4f}, std = {act.std().item():.4f}")
            #print(f"[calculate_conductance] {key}: grad mean = {act.grad.mean().item():.4f}, std = {act.grad.std().item():.4f}")
            
            B = act.shape[0] // (n_steps + 1)
            new_shape = (n_steps + 1, B) + act.shape[1:]
            acts = act.reshape(new_shape)
            grads = act.grad.reshape(new_shape)
            #print(f"[calculate_conductance] {key}: acts shape: {acts.shape}, grads shape: {grads.shape}")
            
            # For debugging, print the mean absolute difference and gradient in the first step.
            diffs = acts[1:] - acts[:-1]
            grad_seg = grads[:-1]
            simple_attr = (acts[-1] * grads[-1]).mean(dim=0)
            print("Simple attr last step mean:", simple_attr.mean().item())
            #print(f"[calculate_conductance] {key}: mean absolute diff (step 0): {diffs[0].abs().mean().item():.4f}")
            #print(f"[calculate_conductance] {key}: mean grad (step 0): {grad_seg[0].abs().mean().item():.4f}")
            
            # integrated = (diffs * grad_seg).sum(dim=0) / n_steps  # shape: [B, ...]
            avg_conductance = (diffs * grad_seg).sum(dim=0).mean(dim=0)


            #print(f"[calculate_conductance] {key}: integrated shape: {avg_conductance.shape}")
            #avg_conductance = integrated.mean(dim=0)  # average over batch
            #print(f"[calculate_conductance] {key}: avg_conductance: {avg_conductance}")
            
            if n_batches is None:
                self.conductance[key] = avg_conductance
            elif self.conductance[key] is None:
                self.conductance[key] = avg_conductance / n_batches
            else:
                self.conductance[key] += avg_conductance / n_batches

        self._saved.clear()
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def update_dropout_masks(self, print_stats=True):
        # Only try to print zeros if we actually got a tensor.
        if 'fc1' in self.conductance and isinstance(self.conductance['fc1'], torch.Tensor):
            #zeros_count = (self.conductance['fc1'] == 0).sum().item()
            #print(f"Total conductance fc1: {self.conductance['fc1'].sum().item()}")
            #print("Amount of zeros in conductance fc1:", zeros_count)
            self.drop1.update_dropout_masks(self.conductance['fc1'], print_stats)
        else:
            print("Conductance for fc1 not available or not a tensor:", self.conductance.get('fc1'))
        if 'fc2' in self.conductance and isinstance(self.conductance['fc2'], torch.Tensor):
            #zeros_count = (self.conductance['fc2'] == 0).sum().item()
            #print(f"Total conductance fc2:{self.conductance['fc2'].sum().item()}")

            #print("Amount of zeros in conductance fc2:", zeros_count)
            self.drop2.update_dropout_masks(self.conductance['fc2'], print_stats)
        else:
            print("Conductance for fc2 not available or not a tensor:", self.conductance.get('fc2'))
        
        self.conductance = {'fc1': None, 'fc2': None}
        return



    def calculate_scores(self, batches: Iterable, device: torch.device,stats = True) -> None:
        # Create a detached copy of the model for IG computation.
        # model_clone = copy.deepcopy(self)
        # model_clone.to(device)
        model_clone = copy.deepcopy(self)
        model_clone.to(device)
        model_clone.eval()  
        model_clone.conductance = {'fc1': 0.0, 'fc2': 0.0}
        for batch in batches:
            x, _ = batch  # assuming batch is (samples, targets)
            x_captum = x.detach().clone().requires_grad_()
            x_captum = x_captum.to(device, non_blocking=True)
            baseline = torch.zeros_like(x_captum)
            outputs = model_clone(x_captum)
            pred = outputs.argmax(dim=1)
            lc_fc1 = LayerConductance(model_clone, model_clone.relu1)
            captum_attr_fc1 = lc_fc1.attribute(x_captum, baselines=baseline, target=pred,n_steps =model_clone.n_steps)
            # Compute layer conductance for fc2 using the ReLU after fc2.
            lc_fc2 = LayerConductance(model_clone, model_clone.relu2)
            captum_attr_fc2 = lc_fc2.attribute(x_captum, baselines=baseline, target=pred,n_steps = model_clone.n_steps)
            # Average out the conductance across the batch.
            model_clone.conductance['fc1'] += captum_attr_fc1.mean(dim=0)
            model_clone.conductance['fc2'] += captum_attr_fc2.mean(dim=0)

            # avg_attr_fc1 = captum_attr_fc1.mean(dim=0)
            # avg_attr_fc2 = captum_attr_fc2.mean(dim=0)
            # print("Batch Captum conductance average for fc1:", avg_attr_fc1.sum().item())
            # print("Batch Captum conductance average for fc2:", avg_attr_fc2.sum().item())


            # x, _ = batch  # assuming batch is (samples, targets)
            # x = x.to(device, non_blocking=True)
            # interp = model_clone.split_images(x, n_steps=model_clone.n_steps)
            # B, steps, C, H, W = interp.shape
            # interp_flat = interp.view(-1, C, H, W)
            # out = model_clone.save_output_gradients(interp_flat, model_clone.n_steps)
            # loss = out.sum()
            # loss.backward()
            # model_clone.calculate_conductance(n_steps=model_clone.n_steps, n_batches=n_batches)
            # baseline = torch.zeros_like(x)
            # output_diff = (model_clone(x) - model_clone(baseline)).sum().item()
            # print(f"Model output difference: {output_diff:.4f}")
            model_clone.update_dropout_masks(stats)
            self.drop1.load_state_dict(model_clone.drop1.state_dict())
            self.drop2.load_state_dict(model_clone.drop2.state_dict())
            self.drop1.scaling = model_clone.drop1.scaling.detach().clone()
            self.drop2.scaling = model_clone.drop2.scaling.detach().clone()
            self.drop1.previous = model_clone.drop1.previous.detach().clone()
            self.drop2.previous = model_clone.drop2.previous.detach().clone()
            self.drop1.stats = model_clone.drop1.stats.copy()
            self.drop2.stats = model_clone.drop2.stats.copy()
            self.drop1.avg_scoring = model_clone.drop1.avg_scoring
            self.drop2.avg_scoring = model_clone.drop2.avg_scoring
            self.drop1.avg_dropout = model_clone.drop1.avg_dropout
            self.drop2.avg_dropout = model_clone.drop2.avg_dropout
            self.drop1.var_scoring = model_clone.drop1.var_scoring
            self.drop2.var_scoring = model_clone.drop2.var_scoring
            self.drop1.var_dropout = model_clone.drop1.var_dropout
            self.drop2.var_dropout = model_clone.drop2.var_dropout
            self.train()
    def effecient_conductance_calculation(self, batches: Iterable, device: torch.device,stats = True) -> None:

        model_clone = copy.deepcopy(self)
        model_clone.to(device)
        model_clone.eval()  # Make sure it’s in eval mode to freeze things like batchnorm statistics.
        model_clone.conductance = {'fc1': 0.0, 'fc2': 0.0}
        for batch in batches:
            x, _ = batch  # assuming batch is (samples, targets)
            x_captum = x.detach().clone().requires_grad_()
            x_captum = x_captum.to(device, non_blocking=True)
            # baseline = torch.zeros_like(x_captum)
            # outputs = model_clone(x_captum)
            # pred = outputs.argmax(dim=1)
            # lc_fc1 = LayerConductance(model_clone, model_clone.relu1)
            # captum_attr_fc1 = lc_fc1.attribute(x_captum, baselines=baseline, target=pred,n_steps =model_clone.n_steps)
            # # Compute layer conductance for fc2 using the ReLU after fc2.
            # lc_fc2 = LayerConductance(model_clone, model_clone.relu2)
            # captum_attr_fc2 = lc_fc2.attribute(x_captum, baselines=baseline, target=pred,n_steps = model_clone.n_steps)
            # # Average out the conductance across the batch.
            # model_clone.conductance['fc1'] += captum_attr_fc1.mean(dim=0)
            # model_clone.conductance['fc2'] += captum_attr_fc2.mean(dim=0)

            interp = model_clone.split_images(x_captum, n_steps=model_clone.n_steps)
            interp
            B, steps, C, H, W = interp.shape
            interp_flat = interp.view(-1, C, H, W)
            out = model_clone.save_output_gradients(interp_flat, model_clone.n_steps)
            loss = out.sum()
            loss.backward()
            model_clone.calculate_conductance(n_steps=model_clone.n_steps, n_batches=n_batches)
            baseline = torch.zeros_like(x)
            output_diff = (model_clone(x) - model_clone(baseline)).sum().item()
            print(f"Model output difference: {output_diff:.4f}")
            model_clone.update_dropout_masks(stats)

        
        # Now, copy the updated dropout mask parameters from model_clone back to self.
        # (The exact code here depends on how MyDropout stores its mask/hyperparameters.)
        self.drop1.load_state_dict(model_clone.drop1.state_dict())
        self.drop2.load_state_dict(model_clone.drop2.state_dict())
        self.drop1.scaling = model_clone.drop1.scaling.detach().clone()
        self.drop2.scaling = model_clone.drop2.scaling.detach().clone()
        self.drop1.previous = model_clone.drop1.previous.detach().clone()
        self.drop2.previous = model_clone.drop2.previous.detach().clone()
        self.drop1.stats = model_clone.drop1.stats.copy()
        self.drop2.stats = model_clone.drop2.stats.copy()
        self.drop1.avg_scoring = model_clone.drop1.avg_scoring
        self.drop2.avg_scoring = model_clone.drop2.avg_scoring
        self.drop1.avg_dropout = model_clone.drop1.avg_dropout
        self.drop2.avg_dropout = model_clone.drop2.avg_dropout
        self.drop1.var_scoring = model_clone.drop1.var_scoring
        self.drop2.var_scoring = model_clone.drop2.var_scoring
        self.drop1.var_dropout = model_clone.drop1.var_dropout
        self.drop2.var_dropout = model_clone.drop2.var_dropout
        self.train()


    def base_dropout(self):
        self.drop1.base_dropout()
        self.drop2.base_dropout()
    def custom_dropout(self):
        self.drop1.custom_dropout()
        self.drop2.custom_dropout()
    def compute_and_plot_history_statistics(self, epoch_label, save_dir=None):
        self.drop1.compute_and_plot_history_statistics(epoch_label+' fc1 ', save_dir)
        self.drop2.compute_and_plot_history_statistics(epoch_label+" fc2 ", save_dir)
    def plot_progression_statistics(self, save_dir=None):
        self.drop1.plot_progression_statistics(save_dir,label = "fc1")
        self.drop2.plot_progression_statistics(save_dir,label ="fc2")

    def clear_update_history(self):
        self.drop1.clear_update_history()
        self.drop2.clear_update_history()
    def check_dead_neurons(self, x):
        with torch.no_grad():
            activations = F.relu(self.fc1(self.flatten(self.pool(F.relu(self.bn2(self.conv2(
                self.pool(F.relu(self.bn1(self.conv1(x)))))))))))
            active_neurons = (activations > 0).float().mean().item()
            print(f"Fraction of active fc1 neurons: {active_neurons:.4f}")



def get_args_parser():
    parser = argparse.ArgumentParser('SimpleCNNMLP Training Script', add_help=False)
    parser.add_argument('--batch-size', default=128, type=int, help='Batch size for training')
    parser.add_argument('--epochs', default=50, type=int, help='Number of training epochs')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--device', default='cuda', type=str, help='Device to use for training (cuda or cpu)')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--output_dir', default='./output', type=str, help='Directory to save checkpoints and logs')
    parser.add_argument('--use_custom_dropout', action='store_true', default=True,
                        help='Enable custom dropout (MyDropout) instead of standard dropout')
    parser.add_argument('--no_custom_dropout', action='store_false', dest='use_custom_dropout',
                        help='Disable custom dropout')
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
    parser.add_argument('--input-size', default=32, type=int, help='images input size')
    parser.add_argument('--experiment_name', default='simplecnnmlp', type=str, help='experiment name')
    parser.add_argument('--plot_freq', default=5, type=int, help='plot frequency')
    parser.add_argument('--resume', default='', type=str, help='Path to checkpoint to resume training or load a model')
    parser.add_argument('--scaler', default=1.0, type=float, help='Loss scaler for mixed precision training')
    parser.add_argument('--mask_type', default='sigmoid', type=str, help='Type of mask for dropout')




    return parser

def main(args):
    # Set the random seed for reproducibility.
    torch.manual_seed(args.seed)
    
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
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
    
    train_dataset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True,
                                                 transform=transform_train)
    test_dataset  = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True,
                                                 transform=transform_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True,drop_last=True)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                               shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize the model.
    # Initialize the model using the CNN6_S1 architecture (smallest configuration).
    model = CNN6_S1(num_classes=10, use_custom_dropout=args.use_custom_dropout,
                elasticity=args.elasticity, p=args.drop, n_steps=args.n_steps,mask_type = args.mask_type,scaler = args.scaler)
    model = model.to(device)

    
    # Set up optimizer and loss function.
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    output_dir = Path(args.output_dir)
    output_dir = output_dir / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device,weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed at epoch {start_epoch}")
        cumulative_train_time = checkpoint['train_time']
        best_acc = checkpoint['best_acc']
        history = checkpoint.get('history', {})
        if history:
            drop1_history = history.get('drop1', {})
            drop2_history = history.get('drop2', {})
            model.drop1.avg_scoring = drop1_history.get('avg_scoring', [])
            model.drop1.avg_dropout = drop1_history.get('avg_dropout', [])
            model.drop1.var_scoring = drop1_history.get('var_scoring', [])
            model.drop1.var_dropout = drop1_history.get('var_dropout', [])
            
            model.drop2.avg_scoring = drop2_history.get('avg_scoring', [])
            model.drop2.avg_dropout = drop2_history.get('avg_dropout', [])
            model.drop2.var_scoring = drop2_history.get('var_scoring', [])
            model.drop2.var_dropout = drop2_history.get('var_dropout', [])
    else:
        start_epoch = 0
        cumulative_train_time = 0.0
        best_acc = 0.0
    
    
    print("Start training")
    start_time = time.time()
    best_loss = float('inf')
    #initially normal dropout
    loss_scaler = NativeScaler()
    if args.use_custom_dropout:
        model.base_dropout()
    check = False
    for epoch in range(start_epoch, args.epochs):
        stats = False
        stats2 = False
        start_time = time.time()
        if args.use_custom_dropout and (epoch) >= args.annealing_factor:
            model.custom_dropout()
            check = True
            if (epoch+1)%args.plot_freq == 0:
                stats = True
            if (epoch+1)%(args.plot_freq) == 0:
                stats2 = True

            

        # Train for one epoch.
        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            max_norm=0,
            model_ema=None,
            mixup_fn=None,
            check=check,
            update_freq=args.update_freq,
            update_batches=args.update_batches,
            stats = stats
        )

        epoch_time = time.time() - start_time
        cumulative_train_time += epoch_time
        print(stats)
        
        
        # Evaluate on the test set.
        test_stats = evaluate(test_loader, model, device)
        
        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss {train_stats['loss']:.4f}, "
              f"Test Acc {test_stats.get('acc1', 0):.2f}%, Epoch Time {epoch_time:.2f}s")

        if stats:
            model.compute_and_plot_history_statistics(f'Epoch {epoch+1}', output_dir / 'plots')
            model.clear_update_history()
        checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'test_acc': best_acc,
                'train_stats': train_stats,
                'test_stats': test_stats,
                'train_time': cumulative_train_time,
                'best_acc': best_acc,  
            }
        if args.use_custom_dropout:
            checkpoint['history'] = {
                
                'drop1': {
                    'avg_scoring': model.drop1.avg_scoring,
                    'avg_dropout': model.drop1.avg_dropout,
                    'var_scoring': model.drop1.var_scoring,
                    'var_dropout': model.drop1.var_dropout,
                },
                'drop2': {
                    'avg_scoring': model.drop2.avg_scoring,
                    'avg_dropout': model.drop2.avg_dropout,
                    'var_scoring': model.drop2.var_scoring,
                    'var_dropout': model.drop2.var_dropout,
                }
            }
            
        torch.save(checkpoint, output_dir / "checkpoint.pth")

        # Save checkpoint if a new best accuracy is reached.
        if test_stats.get('acc1', 0) > best_acc:
            best_acc = test_stats.get('acc1', 0)

            torch.save(checkpoint, output_dir /"best_model.pth")
        
        # Log epoch statistics.
        log_stats = {
            'epoch': epoch,
            'train_loss': train_stats.get('loss', 0),
            'test_acc': test_stats.get('acc1', 0),
            'time': cumulative_train_time,
            'best_acc': best_acc,
        }
        if stats2:
            model.plot_progression_statistics(output_dir / 'plots')
        with (output_dir/ "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")
    
    total_time_str = str(datetime.timedelta(seconds=int(cumulative_train_time)))
    print(f"Training complete. Best Test Accuracy: {best_acc:.2f}%. Total training time: {total_time_str}")
        

    
    # Optionally update custom dropout hyperparameters for each dropout layer.
# Example usage:
if __name__ == '__main__':
    parser = argparse.ArgumentParser('SimpleCNNMLP Training Script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)