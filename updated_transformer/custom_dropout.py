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

from timm.models.vision_transformer import VisionTransformer, _cfg, LayerScale
from timm.models import register_model
from timm.layers import PatchEmbed,use_fused_attn,DropPath, trunc_normal_


"""DropPath and LayerScale may need changes"""


class MyDropout(nn.Module):
    def __init__(self, p_high=0.1, p_low=0.3,elasticity = 1.0,mean_shift =0.5,p=None):
        """
        p_high: dropout probability for elements with a high score (score > split)
        p_low: dropout probability for elements with a low score (score <= split)
        elasticity: how quickly the dropout mask changes (0 = no change, 1 = immediate change)
        mean_shift: how much to shift the split point from the median
        p: standard dropout rate if needed
        """
        super(MyDropout, self).__init__()
        self.p_high = p_high
        self.p_low = p_low
        if p is None:
            self.p = (p_high+p_low)/2
        else:
            self.p = p
        self.elasticity = elasticity
        self.mean_shift = mean_shift
        self.previous  = None
        self.scaling = None
        self.base = False
    
    def count_shifts(self, mask):
        """Function to count the amount of shifts  in the mask"""
        if self.previous is None:
            return 0
        else:
            return torch.sum(mask != self.previous).item()

    def update_dropout_masks(self,scoring):
        """Function to make custom masks based on the scoring"""
        #calculate the split point
        median = torch.median(scoring)
        mean = torch.mean(scoring)
        split = median + (mean - median)*self.mean_shift
        """ test if this split is good, maybe go back to the original"""

        if self.scaling == None: #initialise scaling
            self.scaling = torch.full_like(scoring, 0.5*(self.p_high+self.p_low))

        # Assign dropout (keep) probabilities according to the split.
        # For scores above the split, keep probability is 1 - p_high; otherwise 1 - p_low.
        keep_prob = torch.where(scoring > split, 1 - self.p_high, 1 - self.p_low)

        # Print the amount of shifts (from low to high or vice versa).
        #print("Amount of shifts: ", self.count_shifts(keep_prob))
        shifts = self.count_shifts(keep_prob)
        #update scaling
        self.scaling = self.scaling*(1-self.elasticity) + keep_prob*self.elasticity
        
        self.previous = keep_prob
        return shifts

    def reset_dropout_masks(self):
        """Reset the dropout masks to None."""
        self.previous = None
        self.scaling = None
        return
    def update_parameters(self,p_high=None, p_low=None,elasticity = None,mean_shift = None,p=None):
        """Update the hyperparameters of the custom dropout layer."""
        if p_high is not None:
            self.p_high = p_high
        if p_low is not None:
            self.p_low = p_low
        if elasticity is not None:
            self.elasticity = elasticity
        if mean_shift is not None:
            self.mean_shift = mean_shift
        if p is not None:
            self.p = p
    def base_dropout(self):
        """Use the standard dropout."""
        self.base = True
        return
    def custom_dropout(self):
        """Use the custom dropout."""
        self.base = False
        return

    def forward(self, input):
        # If in evaluation mode, simply return the input unchanged.
        if not self.training:
            return input

        # If scoring is not provided, use the standard dropout.
        if self.base is True or self.previous is None:
            mask = torch.empty_like(input).bernoulli_(1 - self.p)
            return mask * input / (1 - self.p)
        else:
            # Generate noise with the same shape as the input.
            noise = torch.normal(mean=0.0, std=0.05, size=self.previous.shape, device=self.previous.device)
            # Clamp the noise to be within [-0.1, 0.1].
            noise = noise.clamp(-0.1, 0.1)
            keep_prob_noisy = self.previous + noise

            mask = torch.empty_like(input).bernoulli_(keep_prob_noisy)
            # Scale the surviving activations by dividing by the corresponding keep probability.
            return mask * input / self.scaling
