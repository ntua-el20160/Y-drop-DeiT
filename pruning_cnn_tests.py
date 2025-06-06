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


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from datasets import build_dataset, create_subdataset
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
from simplecnn import CNN6_S1
from updated_transformer.pruning_indices import select_pruning_indices
from updated_transformer.pruning import prune_selected_layers


def get_args_parser():
    parser = argparse.ArgumentParser('SimpleCNNMLP Training Script', add_help=False)
    parser.add_argument('--batch-size', default=128, type=int, help='Batch size for training')
    parser.add_argument('--epochs', default=50, type=int, help='Number of training epochs')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')

    parser.add_argument('--device', default='cuda', type=str, help='Device to use for training (cuda or cpu)')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--output_dir', default='./output', type=str, help='Directory to save checkpoints and logs')

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
    parser.add_argument('--plot_freq', default=5, type=int, help='plot frequency')
    parser.add_argument('--resume', default='', type=str, help='Path to checkpoint to resume training or load a model')
    parser.add_argument('--scaler', default=1.0, type=float, help='Loss scaler for mixed precision training')
    parser.add_argument('--mask_type', default='sigmoid', type=str, help='Type of mask for dropout')
    parser.add_argument('--sub_dataset', default ='none',choices=['none','stratified', 'random'] ,type=str, help='Sub dataset to use for training')
    parser.add_argument('--sub_factor', default=10, type=int, help='Sub dataset factor')
    parser.add_argument('--update_scaling',choices=['no','increasing', 'decreasing'], default='no', type =str,
                        help='Scale update frequency  for custom dropout')
    parser.add_argument('--update_scaling_steps', default=5, type=int, help='Amount of frequency updates')
    parser.add_argument('--scoring-type', choices=['Conductance', 'Sensitivity'], default='Conductance',
                        type=str, help='Scoring type for custom dropout')
    parser.add_argument('--same_batch', action='store_true', default=False,
                        help='Enable smooth scoring for custom dropout')
    parser.add_argument('--pruning_type',choices=['normalization','quota', 'quotweighted','quotaweighted','hybrid'], default='normalization', type =str,
                        help='how to prune the model')
    parser.add_argument('--pruning_rate', type=float, default=0.2, help='Pruning rate for custom dropout')  
    return parser

def main(args):
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

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    print(f"Using device: {device}")
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
    print(f"Number of training samples: {len(train_dataset)}")
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

  
    model = CNN6_S1(num_classes=num_classes, use_custom_dropout=True,
                elasticity=args.elasticity, p=args.drop, n_steps=args.n_steps,mask_type = args.mask_type,scaler = args.scaler)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    output_dir = Path(args.output_dir)
    output_dir = output_dir / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
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

        if history:
            for i, drop in enumerate(model.drop_list):
                drop_history = history.get(f'drop{i}', {})
                drop.progression_keep = drop_history.get('progression_keep', [])
                drop.progression_scoring = drop_history.get('progression_scoring', [])
    except Exception as e:
        print(f"Failed to resume from checkpoint: {e}")
        raise RuntimeError(f"Failed to resume from checkpoint: {e}")
    print(f"Starting training from epoch {start_epoch}")
    prune_indices = select_pruning_indices(model =model,
                              data_loader = train_loader
                                , device = device
                                ,scoring_type= args.scoring_type,
                                batches_num= args.update_batches,
                                pruning_rate= args.pruning_rate,
                                pruning_type=args.pruning_type)
    # for key, tuple_list in prune_indices.items():
    #     # tuple_list is something like [(203,), (950,), …]
    #     prune_indices[key] = [tup[0] for tup in tuple_list]


        
    print(f"Pruning indices: {prune_indices}")
    print(max(prune_indices[0]), max(prune_indices[1]))
    print(min(prune_indices[0]), min(prune_indices[1]))
    print(f"Pruning rate: {len(prune_indices[0]),len(prune_indices[1])}")
    test_stats = evaluate(test_loader, model, device)
    test_acc = test_stats.get('acc1', 0.0)
    test_loss = test_stats.get('loss', 0.0)
    print(f"Initial test accuracy: {test_acc:.2f}, Initial test loss: {test_loss:.4f}")
    print("Pruning model...")
    # x = model.fc1.weight.clone()
    # x.to(device)
    prune_selected_layers(model, prune_indices)
    model.to(device)
    print("Model after pruning:")
    # offset = 0
    # for i in range(1024):
    #     if (i,) in prune_indices[0]:
    #         print(f"Pruned weight at index {i}")
    #         offset += 1
    #     else:
    #         print(x[i, :] == model.fc1.weight[i-offset, :])
    # print(model.fc1.weight)

    model = model.to(device)
    test_stats = evaluate(test_loader, model, device)
    test_acc = test_stats.get('acc1', 0.0)
    test_loss = test_stats.get('loss', 0.0)
    print(f"Test accuracy: {test_acc:.2f}, Test loss: {test_loss:.4f}")
    loss_scaler = NativeScaler()

    for i in range(10):
        model.use_normal_dropout()

        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=i + start_epoch,
            loss_scaler=loss_scaler,
            max_norm=0,
            model_ema=None,
            mixup_fn=None,
            check=False,
            update_freq=1,
            update_batches=1,
            stats = False,
            update_data_loader = cached_subdataset,
            output_dir = output_dir,
            scoring_type = args.scoring_type,
            same_batch = args.same_batch,
            help_par = 0
        )
        test_stats = evaluate(test_loader, model, device)
        test_acc = test_stats.get('acc1', 0.0)
        test_loss = test_stats.get('loss', 0.0)
    print(f"Initial test accuracy: {test_acc:.2f}, Initial test loss: {test_loss:.4f}")



        

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SimpleCNNMLP Training Script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)