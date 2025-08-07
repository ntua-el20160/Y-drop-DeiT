# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
from functools import partial
from torch.jit import Final
from typing import Type, Optional
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os


from timm.models.vision_transformer import VisionTransformer, _cfg, LayerScale
from timm.models import register_model
from timm.layers import PatchEmbed,use_fused_attn,DropPath, trunc_normal_


"""DropPath and LayerScale may need changes"""
def linear_compression(x, a, b):
    μ = x.mean()
    α = torch.min((μ - a) / μ, (b - μ) / (1 - μ))
    return μ + α * (x - μ)



def update_drop_path_rates(scoring,drop_rate = 0.1,
                           mask_type = "softmax", min_dropout = 0.0,rescaling_type = None):    

    dr_rate = torch.tensor(drop_rate, device=scoring.device, dtype=scoring.dtype)
    beta = torch.log(dr_rate / (1 - dr_rate))
    #Different mask types
    if mask_type == "sigmoid":
        # Original approach
        normalized = (scoring - scoring.mean()) / scoring.std()
        raw_drop = torch.sigmoid(beta + normalized)
    
    elif mask_type == "sigmoid_inverse":
        normalized = (scoring - scoring.mean()) / scoring.std()
        normalized = - normalized

        # Example smaller slope + random noise
        raw_drop = torch.sigmoid(beta + normalized)
    
    
    elif mask_type == "softmax":
  
        epsilon = torch.finfo(scoring.dtype).eps
        s_min, s_max = scoring.min(), scoring.max()
        normalized = 2 * (scoring - s_min) / (s_max - s_min + epsilon) - 1
    
        flat = normalized.view(-1)
        softmax_flat = torch.softmax(flat, dim=0)
        probs = softmax_flat.view(scoring.shape)

        raw_drop = probs * scoring.numel() * dr_rate
                


    elif mask_type == "softmax_inverse":
        epsilon = torch.finfo(scoring.dtype).eps
        s_min, s_max = scoring.min(), scoring.max()
        normalized = 2 * (scoring - s_min) / (s_max - s_min + epsilon) - 1
        normalized = - normalized

    
        flat = normalized.view(-1)
        softmax_flat = torch.softmax(flat, dim=0)
        probs = softmax_flat.view(scoring.shape)

       
        raw_drop = probs * scoring.numel() * dr_rate

    else:
        normalized = (scoring - scoring.mean()) / scoring.std()
        normalized = - normalized
        raw_drop = torch.sigmoid(beta + normalized)

    raw_drop = raw_drop.clamp(0.0, 1.0)


    if rescaling_type == "linear":
        drop_prob = linear_compression(raw_drop, min_dropout, 0.7)
    else:
        drop_prob = raw_drop.clamp(min_dropout, 0.7)

    return drop_prob

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True,newprob: Optional[float] = None):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    if newprob is not None:
        new_keep_prob = 1 - newprob
        random_tensor = x.new_empty(shape).bernoulli_(new_keep_prob)
    else:
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
        self.newprob = None

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep, self.newprob)
    def update_params(self, drop_prob: float = 0., elasticity: float = 0., curr = False):
        self.drop_prob = (1- elasticity) * self.drop_prob + drop_prob * elasticity
        print(f"New rate {drop_prob}, Updated drop_prob: {self.drop_prob}")
        if curr:
            self.newprob = drop_prob
    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'