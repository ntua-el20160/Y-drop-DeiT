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

def piecewise_linear(x, a, b):
    μ = x.mean()
    slope_lo = (μ - a) / μ
    slope_hi = (b - μ) / (1 - μ)
    return torch.where(
        x <= μ,
        a + slope_lo * x,
        b - slope_hi * (1 - x)
    )

def find_gamma(x, a, b,tar = None ,tol=1e-4, max_iter=50):
    if tar == None:
        μ = x.mean().item()
    else:
        μ = tar
    target = (μ - a) / (b - a)
    lo, hi = 1e-3, 10.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        if torch.mean(x**mid).item() > target:
            lo = mid
        else:
            hi = mid
        if abs(torch.mean(x**mid).item() - target) < tol:
            break
    return mid

def power_law_rescale(x, a, b,tar = None):
    γ = find_gamma(x=x, a=a, b=b,tar=tar)
    return a + (b - a) * x.pow(γ)

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
        self.tied_layer = tied_layer  # Store the tied layer for reference.
        self.transformer_mean = transformer_mean  # Whether to use the transformer mean for normalization.
        self.rescaling_type = rescaling_type  # Type of rescaling to apply to the scoring values.

        # Buffers will be lazily initialized based on the tied layer's output.
 
        self.register_buffer("previous", torch.empty(0))
        self.register_buffer("scaling", torch.empty(0))
        self.register_buffer("scoring", torch.empty(0))
        self.beta = torch.tensor(0.0)
        self.initialized = False

        # Aggregated statistics for updating without keeping full history:
        self.n_updates = 0  # Number of updates processed.
        self.running_scoring_mean = None  # Running (per-neuron) average of scoring.
        self.running_dropout_mean = None  # Running (per-neuron) average of keep probability.
        
        # Histograms (fixed 50 bins): cumulative counts for scoring and keep probability.
        self.scoring_hist = np.zeros(100)  
        self.keep_hist = np.zeros(100)
        self.scoring_hist_focused = np.zeros(300)  # Focused histogram for scoring.
        self.random_neurons = []  # Randomly selected neurons for histogram tracking.

        self.random_neuron_hists_scoring = [np.zeros(100) for _ in range(4)]  # List of random neuron scoring histograms.
        self.random_neuron_hists_keep = [np.zeros(100) for _ in range(4)]  # List of random neuron histograms.
        
        # For progression statistics (one scalar per update).
        self.sum_scoring = None  # Cumulative sum to compute overall average scoring.
        self.sum_keep = None     # Cumulative sum to compute overall average keep probability.
        self.progression_scoring = []  # List of overall average scoring per update.
        self.progression_keep = []     # List of overall average keep probability per update.

    
    def initialize_buffers(self, feature_shape, device):
        new_prev = torch.full(feature_shape, 1 - self.p, device=device)
        new_scaling = torch.full(feature_shape, 1 - self.p, device=device)
        new_scoring = torch.zeros(feature_shape, device=device)
        new_beta = torch.log(torch.tensor(self.base_keep / (1 - self.base_keep),
                                            dtype=new_prev.dtype,
                                            device=device))
        # Update the registered buffers.
        self.previous = new_prev
        self.scaling = new_scaling
        self.scoring = new_scoring
        self.beta = new_beta
        self.initialized = True
        num_neurons = new_prev.numel()
        flat_idxs = np.random.choice(
            num_neurons, size=min(4, num_neurons), replace=False
        )
        self.random_neurons = [
            tuple(np.unravel_index(i, new_prev.shape))
            for i in flat_idxs
        ]
        self.random_neuron_hists_scoring = [np.zeros(100) for _ in self.random_neurons ]  # List of random neuron scoring histograms.
        self.random_neuron_hists_keep = [np.zeros(100) for _ in self.random_neurons]  # List of random neuron histograms.
        



        # self.random_neurons = np.random.choice(num_neurons, size=min(4, num_neurons), replace=False).tolist()
        # self.random_neuron_hists_scoring = [np.zeros(100) for _ in self.random_neurons ]  # List of random neuron scoring histograms.
        # self.random_neuron_hists_keep = [np.zeros(100) for _ in self.random_neurons]  # List of random neuron histograms.




    def update_dropout_masks(self, scoring, stats=True,noisy = False,min_dropout = 0.0):    
        """Update the dropout masks based on the scoring tensor.
        scoring: a tensor of shape [channels] representing the scoring values.
        stats: whether to save the scoring and dropout history.
        Mask types:
        -sigmoid:sigmoid around the dropout rate shifted by the scoring.
        -sigmoid_mod: sigmoid with random noise on the final mask.
        -softmax: softmax of the negative scoring multiplied by the number of channels and chosen dropout rate.
        -softmax_renorm: softmax of the negative scoring multiplied by the number of channels and chosen dropout rate, renormalized to keep average near the set dropout rate.
        -rank: rank of the scoring values, with a ramp from 1 to 0.
        -inverse: inverse sigmoid for fine-tuning.
        -dynamic_sigmoid: dynamic sigmoid based on the min and max of the scoring values.
        """
       
        # Normalize scoring

        #ormalized = (scoring - scoring.mean()) / scoring.std()
        # epsilon = 1e-6
        # s_min, s_max = scoring.min(), scoring.max()
        # normalized = 2 * (scoring - s_min) / (s_max - s_min + epsilon) - 1
        scoring_final = scoring
        self.scoring.copy_(scoring_final)
        #Different mask types
        if self.mask_type == "sigmoid":
            # Original approach
            scoring_final = -scoring_final
            normalized = (scoring_final - scoring_final.mean()) / scoring_final.std()
            raw_keep = torch.sigmoid(self.beta + self.scaler * normalized)
            #keep_prob = torch.clamp(keep_prob, min=0.3,max=0.95)
        
        elif self.mask_type == "sigmoid_inverse":
            normalized = (scoring_final - scoring_final.mean()) / scoring_final.std()
            # Example smaller slope + random noise
            raw_keep = torch.sigmoid(self.beta + self.scaler * normalized)
            #keep_prob = torch.clamp(keep_prob, min=0.3,max=0.95)
        
        
        elif self.mask_type == "softmax":
            # Make sure scoring is not huge in magnitude.
            #print(f"Score in function {scoring_final}")
            scoring_final = -scoring_final

            epsilon = torch.finfo(scoring_final.dtype).eps
            s_min, s_max = scoring_final.min(), scoring_final.max()
            #print("Max - Min:",s_max -s_min)
            normalized = 2 * (scoring_final - s_min) / (s_max - s_min + epsilon) - 1
            #print("Normalized inside:",normalized)
        
            flat = normalized.view(-1)
            softmax_flat = torch.softmax(flat, dim=0)
            probs = softmax_flat.view(scoring_final.shape)

            #normalize for average dropout rate close to p
            #keep_prob = power_law_rescale(raw_keep, 0.3, 1.0 - min_dropout,self.base_keep)
            raw_keep = probs * self.scaling.numel() * self.base_keep
                 

            #keep_prob = raw_keep.clamp(min=0.3, max=1.0 - min_dropout)

        elif self.mask_type == "softmax_inverse":
            # Make sure scoring is not huge in magnitude.
            
            epsilon = torch.finfo(scoring_final.dtype).eps
            s_min, s_max = scoring_final.min(), scoring_final.max()
            #print("Max - Min:",s_max -s_min)
            normalized = 2 * (scoring_final - s_min) / (s_max - s_min + epsilon) - 1
            #print("Normalized inside:",normalized)
        
            flat = normalized.view(-1)
            softmax_flat = torch.softmax(flat, dim=0)
            probs = softmax_flat.view(scoring_final.shape)

            #normalize for average dropout rate close to p
            #keep_prob = power_law_rescale(raw_keep, 0.3, 1.0 - min_dropout,self.base_keep)
            raw_keep = probs * self.scaling.numel() * self.base_keep
            #keep_prob = raw_keep.clamp(min=0.3, max=1.0)
        elif self.mask_type == "softmax_absolute":
            # Make sure scoring is not huge in magnitude.
            epsilon = torch.finfo(scoring_final.dtype).eps
            s_min, s_max = scoring_final.min(), scoring_final.max()
            normalized = torch.abs(2 *self.scaler* (scoring_final - s_min) / (s_max - s_min + epsilon) - self.scaler)

            flat = normalized.view(-1)
            softmax_flat = torch.softmax(flat, dim=0)
            probs = softmax_flat.view(scoring_final.shape)

            #normalize for average dropout rate close to p
            raw_keep = probs * self.scaling.numel() * self.base_keep
            #keep_prob = raw_keep.clamp(min=0.3, max=)


        else:
            # Fallback or default
            normalized = (scoring_final - scoring_final.mean()) / scoring_final.std()
            raw_keep = torch.sigmoid(self.beta - self.scaler * normalized)
            #keep_prob = torch.clamp(keep_prob, min=0.3, max=1.0 - min_dropout)
        if noisy:
            noise = (torch.rand_like(raw_keep) - 0.5) * 2 * (1-raw_keep.abs())*0.2
            mask = (torch.rand_like(raw_keep) < 0.3).float()

            raw_keep = raw_keep + (mask*noise)

        #power_law_rescale(raw_keep, 0.3, 1.0 - min_dropout)
        raw_keep = raw_keep.clamp(0.0, 1.0)
        #print(self.rescaling_type)


        if self.rescaling_type == "linear":
            keep_prob = linear_compression(raw_keep, 0.3, 1.0 - min_dropout)
        elif self.rescaling_type == "piecewise":
            keep_prob = piecewise_linear(raw_keep, 0.3, 1.0 - min_dropout)
        elif self.rescaling_type == "power_law":
            keep_prob = power_law_rescale(raw_keep, 0.3, 1.0 - min_dropout)
        else:
            keep_prob = raw_keep.clamp(min=0.3, max=1.0 - min_dropout)

        # Step 3: Update scaling buffer and stats if needed
        if self.scaling.numel() == 0 or self.scaling.shape != keep_prob.shape:
            self.scaling = torch.full_like(keep_prob, self.base_keep)
   


            

        if stats:
            self.update_aggregated_statistics(scoring, keep_prob)
        # print("1",self.scaling.device)
        # print("2",keep_prob.device)
        # print("3",scoring_final.device)
        # Momentum-like update
        keep_prob.to(self.scaling.device)
        self.scaling = self.scaling * (1 - self.elasticity) + keep_prob * self.elasticity
        self.previous.copy_(keep_prob)




    def forward(self, input):
        
        if not self.initialized:
            if self.transformer_mean:
                feature_shape =input.shape[2:] #Exclude batch dimension and patch dimension
            else:
                feature_shape = input.shape[1:]  # Exclude the batch dimension.
            self.initialize_buffers(feature_shape, input.device)

        if not self.training:
            return input
        
        #Initialuze buffers if not done yet
        if self.base or self.previous is None:
            mask = torch.empty_like(input).bernoulli_(self.base_keep)
            #print("Neuron amount",mask.shape)
            #print("Amount of zeroes in mask: ",torch.sum(mask == 0))

            return mask * input / (self.base_keep)
        else:
            
            probs = self.previous  # shape: (x, y)

            # Expand to input shape
            expanded_probs = probs.expand_as(input)  # input shape: (b, x, y) or (b, p, x, y)
            expanded_scaling = self.scaling.expand_as(input)  # shape: (b, x, y) or (b, p, x, y)
            # Sample from Bernoulli distribution
            #mask = torch.bernoulli(expanded_probs)
            # m1 = torch.bernoulli(probs)
            # mask = m1.expand_as(input)  # Expand mask to match input shape
            mask = torch.bernoulli(expanded_probs)
            #return mask * input / (self.base_keep)  # Avoid division by zero with a small epsilon
            return mask * input / (expanded_scaling + 1e-12)  # Avoid division by zero with a small epsilon

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