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

class CNN6_S1(nn.Module):
    def __init__(self, num_classes=10, use_custom_dropout=True, elasticity=1.0, p=0.1, n_steps=5,
                 mask_type = 'sigmoid',scaler = 1.0,rescaling_type = None):
        
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
        self.selected_layers = [self.fc1, self.fc2]
        if use_custom_dropout:
            self.drop_list = nn.ModuleList([
                MyDropout(elasticity=elasticity, p=p, tied_layer=layer, mask_type=mask_type, scaler=scaler,rescaling_type=rescaling_type)
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
        x = self.relu1(x)  # relu1
        x = self.drop_list[0](x)
        
        # FC2 with dropout
        x = self.fc2(x)
        x = self.relu2(x)  # relu2
        x = self.drop_list[1](x)
        
        # Output layer
        x = self.fc3(x)
        return x
    
def get_args_parser():
        parser = argparse.ArgumentParser('SimpleCNNMLP Training Script', add_help=False)
        parser.add_argument('--data-set', default='CIFAR10', choices=['CIFAR10', 'CIFAR100'])
        return parser

def main(args):
    if args.data_set == 'CIFAR100':
        DatasetClass = torchvision.datasets.CIFAR100
        num_classes = 100
    else:
        DatasetClass = torchvision.datasets.CIFAR10
        num_classes = 10

    model = CNN6_S1(num_classes=num_classes, use_custom_dropout=False,
            elasticity=0.01, p=0.1, n_steps=4,mask_type ='softmax',scaler = 1.0
            , rescaling_type = 'linear')