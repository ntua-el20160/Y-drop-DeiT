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



def update_drop_path_rates(self,scoring,previous=None,elasticity =0.01,drop_rate = 0.1,
                           mask_type = "softmax", min_dropout = 0.0,recaling_type = None):    

    keep_rate = 1.0 - drop_rate
    beta = torch.log(keep_rate / (1 - keep_rate))
   
    #Different mask types
    if mask_type == "sigmoid":
        # Original approach
        normalized = (scoring - scoring.mean()) / scoring.std()
        normalized = - normalized
        raw_keep = torch.sigmoid(beta + normalized)
    
    elif self.mask_type == "sigmoid_inverse":
        normalized = (scoring - scoring.mean()) / scoring.std()
        # Example smaller slope + random noise
        raw_keep = torch.sigmoid(beta + normalized)
    
    
    elif self.mask_type == "softmax":
  
        epsilon = torch.finfo(scoring.dtype).eps
        s_min, s_max = scoring.min(), scoring.max()
        normalized = 2 * (scoring - s_min) / (s_max - s_min + epsilon) - 1
        normalized = -normalized  
    
        flat = normalized.view(-1)
        softmax_flat = torch.softmax(flat, dim=0)
        probs = softmax_flat.view(scoring.shape)

        raw_keep = probs * drop_rate * scoring.numel() * self.base_keep
                

        #keep_prob = raw_keep.clamp(min=0.3, max=1.0 - min_dropout)

    elif self.mask_type == "softmax_inverse":
        epsilon = torch.finfo(scoring.dtype).eps
        s_min, s_max = scoring.min(), scoring.max()
        normalized = 2 * (scoring - s_min) / (s_max - s_min + epsilon) - 1
    
        flat = normalized.view(-1)
        softmax_flat = torch.softmax(flat, dim=0)
        probs = softmax_flat.view(scoring.shape)

       
        raw_keep = probs * drop_rate * scoring.numel() * self.base_keep


    else:
        normalized = (scoring - scoring.mean()) / scoring.std()
        normalized = - normalized
        raw_keep = torch.sigmoid(beta + normalized)

    raw_keep = raw_keep.clamp(0.0, 1.0)


    if recaling_type == "linear":
        keep_prob = linear_compression(raw_keep, 0.6, 1.0 - min_dropout)
    else:
        keep_prob = raw_keep.clamp(0.6, 1.0 - min_dropout)


    if previous is not None:
        keep_prob = previous * (1 - elasticity) + keep_prob * elasticity
    
    return keep_prob
