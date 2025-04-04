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
from evaluate_gradients.MultiLayerConductance import MultiLayerConductance


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
        self.selected_layers = [self.relu1, self.relu2]
        if use_custom_dropout:
            self.drop_list = nn.ModuleList([
                MyDropout(elasticity=elasticity, p=p, tied_layer=layer, mask_type=mask_type, scaler=scaler)
                for layer in self.selected_layers
            ])
        else:
            self.drop_list = nn.ModuleList([nn.Dropout(p) for _ in self.selected_layers])
        self.scores ={}
        
        
    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool(x)
        
        # Flatten
        x = self.flatten(x)  # Shape: [B, 4096]
        
        # FC1 with dropout
        x = self.fc1(x)
        x = self.selected_layers[0](x)  # relu1
        x = self.drop_list[0](x)
        
        # FC2 with dropout
        x = self.fc2(x)
        x = self.selected_layers[1](x)  # relu2
        x = self.drop_list[1](x)
        
        # Output layer
        x = self.fc3(x)
        return x

    def calculate_scores(self, batches: Iterable, device: torch.device,stats = True) -> None:
        # Create a detached copy of the model for IG computation.
        model_clone = copy.deepcopy(self)
        model_clone.to(device)
        model_clone.eval()  
        
        # Initialize conductances for each layer
        for i, _ in enumerate(self.selected_layers):
            model_clone.scores[f'drop_{i}'] = 0.0

        for batch in batches:
            x, _ = batch  # Batch is (samples, targets)
            x_captum = x.detach().clone().requires_grad_()
            x_captum = x_captum.to(device, non_blocking=True)
            baseline = torch.zeros_like(x_captum)

            # Get model predictions
            outputs = model_clone(x_captum)
            pred = outputs.argmax(dim=1)

            #calculate conductunce for batch
            mlc = MultiLayerConductance(model_clone, model_clone.selected_layers)
            captum_attrs = mlc.attribute(x_captum, baselines=baseline, target=pred, n_steps=model_clone.n_steps)

            # Average out the conductance across the batch and add it
            for i, score in enumerate(captum_attrs):
                model_clone.scores[f'drop_{i}'] += score.mean(dim=0)

        #update the masks based on the scores
        for i,_ in enumerate(model_clone.drop_list):
            model_clone.drop_list[i].update_dropout_masks(model_clone.scores[f'drop_{i}'],stats)
        # Update the dropout masks based on the accumulated conductances
        for i, drop_layer in enumerate(model_clone.drop_list):
            drop_layer.update_dropout_masks(
                model_clone.scores[f'drop_{i}'],
                stats=stats
            )


        #load the update on the model from the copy
        for i,_ in enumerate(model_clone.drop_list):
            self.drop_list[i].load_state_dict(model_clone.drop_list[i].state_dict())
            self.drop_list[i].scaling = model_clone.drop_list[i].scaling.detach().clone()
            self.drop_list[i].previous = model_clone.drop_list[i].previous.detach().clone()
            self.drop_list[i].stats = model_clone.drop_list[i].stats.copy()
            self.drop_list[i].avg_scoring = model_clone.drop_list[i].avg_scoring
            self.drop_list[i].avg_dropout = model_clone.drop_list[i].avg_dropout
            self.drop_list[i].var_scoring = model_clone.drop_list[i].var_scoring
            self.drop_list[i].var_dropout = model_clone.drop_list[i].var_dropout
        self.train()
    
    def base_dropout(self):
        for drop in self.drop_list:
            drop.base_dropout()

    def custom_dropout(self):
        for drop in self.drop_list:
            drop.custom_dropout()
    def compute_and_plot_history_statistics(self, epoch_label, save_dir=None):
        for i,_ in enumerate(self.drop_list):
            self.drop_list[i].compute_and_plot_history_statistics(epoch_label+f" fc{i}", save_dir)

 
    def plot_progression_statistics(self, save_dir=None,label =''):
        for i,_ in enumerate(self.drop_list):
            self.drop_list[i].plot_progression_statistics(save_dir,label =label + f" drop{i}")

    def clear_update_history(self):
        for drop in self.drop_list:
            drop.clear_update_history()
# \
#     def check_dead_neurons(self, x):
#         with torch.no_grad():
#             activations = F.relu(self.fc1(self.flatten(self.pool(F.relu(self.bn2(self.conv2(
#                 self.pool(F.relu(self.bn1(self.conv1(x)))))))))))
#             active_neurons = (activations > 0).float().mean().item()
#             print(f"Fraction of active fc1 neurons: {active_neurons:.4f}")



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
            for i, drop in enumerate(model.drop_list):
                drop_history = history.get(f'drop{i}', {})
                drop.avg_scoring = drop_history.get('avg_scoring', [])
                drop.avg_dropout = drop_history.get('avg_dropout', [])
                drop.var_scoring = drop_history.get('var_scoring', [])
                drop.var_dropout = drop_history.get('var_dropout', [])

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
            checkpoint['history'] = {}
            for i, drop in enumerate(model.drop_list):
                checkpoint['history'][f'drop{i}'] = {
                    'avg_scoring': drop.avg_scoring,
                    'avg_dropout': drop.avg_dropout,
                    'var_scoring': drop.var_scoring,
                    'var_dropout': drop.var_dropout,
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