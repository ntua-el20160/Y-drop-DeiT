# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from sklearn.preprocessing import scale
import torch
import torch.nn as nn
from functools import partial
from torch.jit import Final
from typing import Type, Optional
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F 

from timm.models.vision_transformer import VisionTransformer, _cfg, LayerScale
from timm.models import register_model
from timm.layers import PatchEmbed,use_fused_attn,DropPath, trunc_normal_
from updated_transformer.masks import (
    winsorized_minmax_map,
    robust_logistic_map,
    rank_power_map,
    isotonic_regression_map,
    zscore_clamp_map,
    rank_even_blend_map,
    yeo_johnson_map,
)



"""DropPath and LayerScale may need changes"""
# def linear_compression(x, a, b, target_mean=None, eps=1e-10):
#     # ensure a < b
#     if not isinstance(a, torch.Tensor): a = torch.tensor(a, dtype=x.dtype, device=x.device)
#     if not isinstance(b, torch.Tensor): b = torch.tensor(b, dtype=x.dtype, device=x.device)
#     b = torch.maximum(b, a + torch.tensor(1e-6, dtype=x.dtype, device=x.device))

#     mu = x.mean()
#     # If you want a specific mean (e.g., base_keep), use it, otherwise use current mu
#     mu = torch.tensor(target_mean, dtype=x.dtype, device=x.device) if (target_mean is not None) else mu
#     # Project mu into (a, b) to avoid negative/undefined slopes
#     mu = torch.clamp(mu, a + eps, b - eps)

#     # slopes that map x=0 -> >= a and x=1 -> <= b
#     alpha_lo = (mu - a) / (mu + eps)
#     alpha_hi = (b - mu) / (1 - mu + eps)
#     alpha = torch.minimum(alpha_lo, alpha_hi)
#     alpha = torch.clamp(alpha, min=0.0)  # keep monotonic, avoid flipping

#     y = mu + alpha * (x - mu)
#     return torch.clamp(y, a, b)
def linear_compression(x, a, b):
    μ = x.mean()
    α = torch.min((μ - a) / μ, (b - μ) / (1 - μ))
    return μ + α * (x - μ)
def half_rank_scores(x: torch.Tensor, top_val: float = 0.6, bot_val: float = 0.1) -> torch.Tensor:
    """
    Given a 1D tensor x, return a tensor of same length where:
      - the highest ceil(N/2) values get `top_val`
      - the rest get `bot_val`
    NaNs are treated as -inf for ranking (i.e., bottom group).
    """
    if x.ndim != 1:
        raise ValueError("half_rank_scores expects a 1D tensor (vector).")

    n = x.numel()
    if n == 0:
        return x.new_empty(0)

    # Treat NaNs as -inf for sorting
    vals = x.clone()
    neg_inf = torch.tensor(float('-inf'), device=x.device, dtype=x.dtype)
    vals = torch.where(torch.isnan(vals), neg_inf, vals)

    # Sort descending, take top ceil(n/2)
    idx_sorted = torch.argsort(vals, descending=True)
    k = (n + 1) // 2  # ceil(n/2)

    out_dtype = x.dtype if x.is_floating_point() else torch.float32
    out = torch.full((n,), bot_val, device=x.device, dtype=out_dtype)
    out[idx_sorted[:k]] = top_val
    return out

def project_to_capped_simplex(v, target_mean, lo=0.0, hi=1.0, iters=40):
    # v: Tensor; we want y in [lo, hi] with mean == target_mean
    N      = v.numel()
    device = v.device
    dtype  = v.dtype

    lo_t = torch.as_tensor(lo, dtype=dtype, device=device)
    hi_t = torch.as_tensor(hi, dtype=dtype, device=device)

    # desired sum in the unit box
    t = torch.as_tensor(target_mean, dtype=dtype, device=device) * N
    t = torch.clamp(t, lo_t * N, hi_t * N)  # <-- tensor clamp

    # bisection on tau for y = clamp(v - tau, lo, hi), s.t. sum(y) == t
    tau_lo = (v - hi_t).min()
    tau_hi = (v - lo_t).max()
    for _ in range(iters):
        tau = (tau_lo + tau_hi) * 0.5
        y   = torch.clamp(v - tau, lo_t, hi_t)
        s   = y.sum()
        # keep it branchless to avoid CPU/GPU sync
        gt  = (s > t).to(v.dtype)
        tau_lo = gt * tau + (1 - gt) * tau_lo
        tau_hi = (1 - gt) * tau + gt * tau_hi

    return torch.clamp(v - (tau_lo + tau_hi) * 0.5, lo_t, hi_t)


class MyDropout(nn.Module):
    def __init__(self,elasticity = 1.0,p=0.1,tied_layer: Optional[nn.Module] = None,scaler =1.0,
                 mask_type = "sigmoid",transformer_mean = False,rescaling_type = None):
        """
        p: dropout probability.
        elasticity: how quickly the dropout mask changes.
        tied_layer: the module whose output is tied to this dropout.
        scaler: scaling factor used in computing keep probability.
        mask_type: determines which method to use for computing the keep probability.
        """

        super(MyDropout, self).__init__()
      

        # self.register_buffer("previous", torch.full((num_channels,), 1 - p))
        # self.register_buffer("scaling", torch.full((num_channels,), 1 - p))
        # self.register_buffer("scoring", torch.zeros(num_channels))

        #self.beta = torch.log(torch.tensor(self.base_keep / (1 - self.base_keep), dtype=self.previous.dtype, device=self.previous.device))
  
        self.p = p
        self.elasticity = elasticity
        self.scaler = scaler
        self.mask_type = mask_type
        self.base = False
        self.base_keep = 1 - self.p
        self.tied_layer = tied_layer
        self.transformer_mean = transformer_mean
        self.rescaling_type = rescaling_type

        # Lazily shaped buffers; keep them registered!
        self.register_buffer("previous", torch.empty(0))
        self.register_buffer("scaling",  torch.empty(0))
        self.register_buffer("beta",     torch.tensor(0.0))  # logit(base_keep) set on first init
        #self.beta = 0.5
        self.initialized = False


        
    
    def initialize_buffers(self, feature_shape, device):
        dtype = None
        try:
            # forward() knows the input dtype; thread it in via a call-site arg if needed
            # but since you're calling from forward, you can just grab input.dtype there
            # and pass it here; simplest minimal change: store it on self before calling
            dtype = self._last_input_dtype   # <-- set this in forward() below
        except AttributeError:
            dtype = torch.get_default_dtype()
        new_prev = torch.full(feature_shape, 1 - self.p, device=device, dtype=dtype)
        new_scaling = torch.full(feature_shape, 1 - self.p, device=device, dtype=dtype)
        new_beta = torch.log(torch.tensor(self.base_keep / (1 - self.base_keep),
                                            dtype=new_prev.dtype,
                                            device=device))
        # Update the registered buffers.
        self.previous = new_prev
        self.scaling = new_scaling
        self.beta = new_beta
        self.initialized = True
    

    def update_dropout_masks(self, scoring, stats=True,noisy = False,min_dropout = 0.0):    

       # Normalize scoring

        a = 0.5 #max dropout
        b = 1.0 - min_dropout #min dropout

        if self.mask_type.endswith("_inverse"):
            scoring_final = scoring
        elif self.mask_type.endswith("_abs"):
            scoring_final = torch.abs(scoring)
        else:
            scoring_final = -scoring
        #Different mask types

        if self.mask_type.startswith("winsor"):
            keep_prob = winsorized_minmax_map(scoring_final, a, b, self.base_keep,lower_q=0.00,upper_q=1.0)
        elif self.mask_type.startswith("robust_logistic"):
            keep_prob = robust_logistic_map(scoring_final, a, b, self.base_keep)
        elif self.mask_type.startswith("isotonic"):
            keep_prob = isotonic_regression_map(scoring_final, a, b, self.base_keep)
        elif self.mask_type.startswith("rank_power"):
            keep_prob = rank_power_map(scoring_final, a, b, self.base_keep, gamma=1.0)
        else:
            # Fallback or default
            keep_prob = winsorized_minmax_map(scoring_final, a, b, self.base_keep, lower_q=0.00, upper_q=1.0)

        keep_prob = keep_prob.clamp(min=0.0, max=1.0).to(self.previous.dtype)

        # Step 3: Update scaling buffer and stats if needed
        if self.scaling.numel() == 0 or self.scaling.shape != keep_prob.shape:
            self.scaling = torch.full_like(keep_prob, self.base_keep)
            self.previous.resize_as_(keep_prob).zero_()


        self.scaling = self.scaling * (1 - self.elasticity) + keep_prob * self.elasticity
        self.previous.copy_(keep_prob)




    def forward(self, input):
        
        if not self.initialized:
            self._last_input_dtype = input.dtype
            if self.transformer_mean:
                feature_shape =input.shape[2:] #Exclude batch dimension and patch dimension
            else:
                feature_shape = input.shape[1:]  # Exclude the batch dimension.
            self.initialize_buffers(feature_shape, input.device)

        if not self.training:
            return input
        
        #Initialuze buffers if not done yet
        if self.base or self.previous is None:
            return F.dropout(input, p=self.p, training=True)
        else:
            
            probs = self.previous  # shape: (x, y)

            # Expand to input shape
            expanded_probs = probs.expand_as(input)  # input shape: (b, x, y) or (b, p, x, y)
            expanded_scaling = self.scaling.expand_as(input)  # shape: (b, x, y) or (b, p, x, y)
            mask  = torch.bernoulli(expanded_probs.to(dtype=input.dtype))
            denom = (expanded_scaling.to(dtype=input.dtype) + 1e-12)

            return mask * input / denom
            # mask = torch.bernoulli(expanded_probs)
            # return mask * input / (expanded_scaling + 1e-12)  # Avoid division by zero with a small epsilon

    def switch(self):
        if self.mask_type == "softmax_inverse":
            self.mask_type = "softmax"
        elif self.mask_type == "softmax":
                self.mask_type = "softmax_inverse"
        elif self.mask_type == "sigmoid_inverse":
            self.mask_type = "sigmoid"
        elif self.mask_type == "sigmoid":
            self.mask_type = "sigmoid_inverse"

        if not self.initialized:
        # nothing to do until buffers are created
            return

        with torch.no_grad():
            if self.previous.numel() > 1:
                self.previous.copy_(_switch_tensor(self.previous))
            if self.scaling.numel() > 1:
                self.scaling.copy_(_switch_tensor(self.scaling))
            
  

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # For lazy-initialized buffers, if they are still empty, override them with the checkpoint values.
        for key in ['previous', 'scaling', 'scoring']:
            full_key = prefix + key
            if full_key in state_dict:
                checkpoint_val = state_dict[full_key]

                # Replace the uninitialized buffer with the checkpoint tensor.
                setattr(self, key, checkpoint_val)
                # Remove the key so that the parent's loader doesn't attempt to load it.
                del state_dict[full_key]
        super(MyDropout, self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                                     strict, missing_keys, unexpected_keys, error_msgs)
        # Remove our keys from missing_keys since we've already loaded them.
        for key in ['previous', 'scaling', 'scoring']:
            full_key = prefix + key
            if full_key in missing_keys:
                missing_keys.remove(full_key)


    def reset_dropout_masks(self):
        """Reset the dropout masks to their default values."""
        if self.initialized:
            # Instead of removing the buffers, reset them to the default (1 - p)
            self.previous.fill_(1 - self.p)
            self.scaling.fill_(1 - self.p)
        return

    def use_normal_dropout(self):
        """Use the standard dropout."""
        self.base = True
        return
    def use_ydrop(self):
        """Use the custom dropout."""
        self.base = False
        return

def to_2d(arr):
    """
    Convert an input numpy array into a 2D array for plotting.
    - If the array is 1D, reshape it to have shape (1, N)
    - If the array is already 2D, return it as is.
    - If the array has >2 dimensions, flatten all dimensions except the last one.
    """
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    elif arr.ndim == 2:
        return arr
    else:
        # Flatten all dimensions except the last one.
        return arr.reshape(-1, arr.shape[-1])
def _switch_tensor(t: torch.Tensor) -> torch.Tensor:
    """
    Return a tensor with values reverse-mapped by rank.
    NaN/Inf entries are left in place and not involved in the switch.
    """
    flat = t.reshape(-1)
    out = flat.clone()

    # operate only on finite entries
    finite_mask = torch.isfinite(flat)
    if finite_mask.sum() <= 1:
        return t  # nothing to reorder (0 or 1 finite values)

    idx = torch.nonzero(finite_mask, as_tuple=False).squeeze(1)   # original indices of finite values
    vals = flat[idx]

    # argsort within the finite subset (ascending)
    order = torch.argsort(vals, dim=0)                # positions in 'vals' from smallest -> largest
    pos_asc = idx[order]                              # original indices of ascending ranks
    vals_desc = vals[order.flip(0)]                   # values in descending order

    # write back: smallest index gets largest value, etc.
    out[pos_asc] = vals_desc

    return out.view_as(t)   


    # if self.mask_type == "sigmoid":
    #     scoring_final = -scoring_final
    #     normalized = (scoring_final - scoring_final.mean()) / scoring_final.std()
    #     raw_keep = torch.sigmoid(self.beta + self.scaler * normalized)
    
    # elif self.mask_type == "sigmoid_inverse":
    #     normalized = (scoring_final - scoring_final.mean()) / scoring_final.std()
    #     raw_keep = torch.sigmoid(self.beta + self.scaler * normalized)
    
    
    # elif self.mask_type == "softmax":

    #     scoring_final = -scoring_final

    #     epsilon = torch.finfo(scoring_final.dtype).eps
    #     s_min, s_max = scoring_final.min(), scoring_final.max()
    #     normalized = 2 * (scoring_final - s_min) / (s_max - s_min + epsilon) - 1
    
    #     flat = normalized.view(-1)
    #     softmax_flat = torch.softmax(flat, dim=0)
    #     probs = softmax_flat.view(scoring_final.shape)


    #     raw_keep = probs * self.scaling.numel() * self.base_keep
                


    # elif self.mask_type == "softmax_inverse":
        
    #     epsilon = torch.finfo(scoring_final.dtype).eps
    #     s_min, s_max = scoring_final.min(), scoring_final.max()
    #     normalized = 2 * (scoring_final - s_min) / (s_max - s_min + epsilon) - 1
    
    #     flat = normalized.view(-1)
    #     softmax_flat = torch.softmax(flat, dim=0)
    #     probs = softmax_flat.view(scoring_final.shape)

    #     raw_keep = probs * self.scaling.numel() * self.base_keep
    # elif self.mask_type == "softmax_absolute":

    #     epsilon = torch.finfo(scoring_final.dtype).eps
    #     scoring_final = torch.abs(scoring_final)
    #     s_min, s_max = scoring_final.min(), scoring_final.max()
    #     normalized = 2 * (scoring_final - s_min) / (s_max - s_min + epsilon) - 1
    
    #     flat = normalized.view(-1)
    #     softmax_flat = torch.softmax(flat, dim=0)
    #     probs = softmax_flat.view(scoring_final.shape)

    #     raw_keep = probs * self.scaling.numel() * self.base_keep