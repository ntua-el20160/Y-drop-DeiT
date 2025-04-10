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


class MyDropout(nn.Module):
    def __init__(self,elasticity = 1.0,p=0.1,tied_layer: Optional[nn.Module] = None,scaler =1.0,mask_type = "sigmoid"):
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
        self.base_keep = 1 - p
        self.tied_layer = tied_layer  # Store the tied layer for reference.

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



    def update_dropout_masks(self, scoring, stats=True):
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
        if scoring.std() > 1e-6:
            normalized = (scoring - scoring.mean()) / scoring.std()
        else:
            normalized = scoring - scoring.mean()

        #Different mask types
        if self.mask_type == "sigmoid":
            # Original approach
            keep_prob = torch.sigmoid(self.beta - self.scaler * normalized)
            keep_prob = torch.clamp(keep_prob, min=0.3,max=0.95)
        
        elif self.mask_type == "sigmoid_mod":
            # Example smaller slope + random noise
            noise = 0.01 * torch.randn_like(normalized)
            keep_prob = torch.sigmoid(self.beta - self.scaler * normalized)+ noise
            keep_prob = torch.clamp(keep_prob, min=0.3,max=0.95)
        
        elif self.mask_type == "softmax":
            # Make sure scoring is not huge in magnitude.
            probs = torch.softmax(-scoring, dim=0)

            #normalize for average dropout rate close to p
            raw_keep = probs * self.scaling.numel() * self.base_keep
            keep_prob = raw_keep.clamp(min=0.3, max=0.95)

        elif self.mask_type == "softmax_alt":
            # Make sure scoring is not huge in magnitude.
            probs = (-torch.softmax(scoring, dim=0)) +1

            #normalize for average dropout rate close to p
            raw_keep = probs * self.scaling.numel() * self.base_keep
            keep_prob = raw_keep.clamp(min=0.3, max=0.95)
        




        elif self.mask_type == "softmax_renorm":
            probs = torch.softmax(-scoring, dim=0)

            raw_keep = probs * self.scaling.numel() * self.base_keep
            keep_prob = raw_keep.clamp(min=0.0, max=1.0)
            # optionally renormalize to keep average near self.base_keep
            keep_prob = keep_prob / keep_prob.mean() * self.base_keep
            keep_prob = torch.clamp(keep_prob, min=0.3,max=0.95)

        
        elif self.mask_type == "rank":
            #flatten scores and sort them
            flat_scores = scoring.view(-1)
            sorted_indices = torch.argsort(flat_scores, descending=False)

            #create ramp from 1 to 0
            ranks = torch.arange(len(flat_scores), device=scoring.device).float()
            ramp = 1.0 - ranks / (len(flat_scores) - 1)

            # aassign ramp values to the sorted indices
            keep_prob_flat = torch.empty_like(flat_scores)
            keep_prob_flat[sorted_indices] = ramp
            keep_prob = keep_prob_flat.view_as(scoring)

            # Rescale average
            current_mean = keep_prob.mean()
            if current_mean > 1e-6:
                keep_prob = keep_prob / current_mean * self.base_keep
            keep_prob = torch.clamp(keep_prob, min=0.3, max=0.95)
        
        elif self.mask_type == "inverse":
            #inverse sigmoid for fine-tuning
            keep_prob = 1.0 - torch.sigmoid(normalized)
            keep_prob = torch.clamp(keep_prob, 0.3, max=0.95)
        
        elif self.mask_type == "dynamic_sigmoid":
            #Normalize scoring to [0, 1]
            s_min, s_max = scoring.min(), scoring.max()

            denom = (s_max - s_min).clamp(min=1e-6)
            scaled = (scoring - s_min) / denom  # [0, 1]
            # Scale to [-2, 2]
            scaled = -(4.0 * scaled - 2.0)        # map to [-2, 2]

            # Apply sigmoid to get keep probability
            keep_prob = torch.sigmoid(scaled)
            keep_prob = torch.clamp(keep_prob, 0.3, 0.95)

        else:
            # Fallback or default
            keep_prob = torch.sigmoid(self.beta - self.scaler * normalized)
            keep_prob = torch.clamp(keep_prob, min=0.3, max=0.95)

        # Step 3: Update scaling buffer and stats if needed
        if self.scaling.numel() == 0 or self.scaling.shape != keep_prob.shape:
            self.scaling = torch.full_like(keep_prob, self.base_keep)
        

        if stats:
            self.update_aggregated_statistics(scoring, keep_prob)

        # Momentum-like update
        self.scaling = self.scaling * (1 - self.elasticity) + keep_prob * self.elasticity
        self.previous.copy_(keep_prob)
        self.scoring.copy_(scoring)



    def forward(self, input):
        
        if not self.initialized:
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
            #mask = torch.empty_like(input).bernoulli_(self.previous)
            mask = torch.empty_like(input).bernoulli_(self.previous)
            #print("Neuron amount",mask.shape)
            #print("Amount of zeroes in mask: ",torch.sum(mask == 0))

            return mask * input / (self.scaling)
    def update_aggregated_statistics(self, scoring, keep_prob):
        """
        Update incremental (running) aggregated statistics with the new scoring and keep_prob values.
        This replaces storing all raw histories.
        It updates:
          - running_scoring_mean (per neuron)
          - running_dropout_mean (per neuron)
          - cumulative histograms for scoring and dropout over fixed bins.
        """
        # a) Detach and convert to CPU numpy arrays.

        scoring_det = scoring.detach().cpu().float()
        keep_prob_det = keep_prob.detach().cpu().float()
        
        
        # b) Update running means per neuron.
        if self.running_scoring_mean is None:
            self.running_scoring_mean = scoring_det.clone()
            self.running_dropout_mean = keep_prob_det.clone()
        else:
            self.running_scoring_mean = (self.running_scoring_mean * self.n_updates + scoring_det) / (self.n_updates + 1)
            self.running_dropout_mean = (self.running_dropout_mean * self.n_updates + keep_prob_det) / (self.n_updates + 1)
        
        current_sum_scoring = scoring_det.sum().item()
        
        # d) Update cumulative sums
        if self.sum_scoring is None:
            self.sum_scoring = current_sum_scoring
        else:
            self.sum_scoring += current_sum_scoring

        # Update histograms.
        bins_scoring = np.linspace(-1, 1, 101)  # 50 bins => 51 edges.
        hist_scoring, _ = np.histogram(scoring_det.numpy().flatten(), bins=bins_scoring)
        self.scoring_hist += hist_scoring
        
        bins_keep = np.linspace(0, 1, 101)
        hist_keep, _ = np.histogram(keep_prob_det.numpy().flatten(), bins=bins_keep)
        self.keep_hist += hist_keep
        
        # Increment the update counter.
        self.n_updates += 1
        
       
    def update_progression(self):
        if self.sum_scoring is not None and self.running_dropout_mean is not None:
            self.progression_keep.append(self.running_dropout_mean.mean().item())
            self.progression_scoring.append(self.sum_scoring)
    def clear_progression(self):
        """Clear the progression statistics."""
        self.n_updates = 0  # Number of updates processed.
        self.running_scoring_mean = None  # Running (per-neuron) average of scoring.
        self.running_dropout_mean = None  # Running (per-neuron) average of keep probability.
        
        # Histograms (fixed 50 bins): cumulative counts for scoring and keep probability.
        self.scoring_hist = np.zeros(100)  
        self.keep_hist = np.zeros(100)
        
        # For progression statistics (one scalar per update).
        self.sum_scoring = None  # Cumulative sum to compute overall average scoring.
        self.sum_keep = None    
    def plot_aggregated_statistics(self, epoch_label, save_dir=None):
        """
        Plot the aggregated statistics:
         • Two histograms:
              - Scoring histogram (50 bins, fixed range -5 to 5).
              - Keep probability histogram (50 bins, fixed range 0 to 1).
         • Two heatmaps:
              - Running per-neuron average scoring.
              - Running per-neuron average keep probability.
        """
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Histogram for scoring.
        bins_scoring = np.linspace(-1, 1, 101)
        bin_centers_scoring = (bins_scoring[:-1] + bins_scoring[1:]) / 2
        axs[0, 0].bar(bin_centers_scoring, self.scoring_hist, width=(bins_scoring[1]-bins_scoring[0]))
        axs[0, 0].set_title(f"{epoch_label} - Scoring Histogram")
        axs[0, 0].set_xlabel("Scoring")
        axs[0, 0].set_ylabel("Count")
        
        # Histogram for keep probability.
        bins_keep = np.linspace(0, 1, 101)
        bin_centers_keep = (bins_keep[:-1] + bins_keep[1:]) / 2
        axs[0, 1].bar(bin_centers_keep, self.keep_hist, width=(bins_keep[1]-bins_keep[0]))
        axs[0, 1].set_title(f"{epoch_label} - Keep Probability Histogram")
        axs[0, 1].set_xlabel("Keep Probability")
        axs[0, 1].set_ylabel("Count")
        
        # Heatmap for running scoring mean.
        if self.running_scoring_mean is not None:
            scoring_mean_np = self.running_scoring_mean.cpu().numpy()
            scoring_mean_2d = to_2d(scoring_mean_np)
            im0 = axs[1, 0].imshow(scoring_mean_2d, aspect='auto', cmap='viridis')
            axs[1, 0].set_title(f"{epoch_label} - Mean Scoring per Neuron")
            fig.colorbar(im0, ax=axs[1, 0])
        else:
            axs[1, 0].text(0.5, 0.5, "No Data", ha="center", va="center")
        
        # Heatmap for running keep probability mean.
        if self.running_dropout_mean is not None:
            dropout_mean_np = self.running_dropout_mean.cpu().numpy()
            dropout_mean_2d = to_2d(dropout_mean_np)
            im1 = axs[1, 1].imshow(dropout_mean_2d, aspect='auto', cmap='magma')
            axs[1, 1].set_title(f"{epoch_label} - Mean Keep Rate per Neuron")
            fig.colorbar(im1, ax=axs[1, 1])
        else:
            axs[1, 1].text(0.5, 0.5, "No Data", ha="center", va="center")
        
        plt.tight_layout()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f"{epoch_label}_aggregated_stats.png"))
        plt.close(fig)   
    

    def plot_progression_statistics(self, save_dir=None, label="progression"):
        """
        Plot the progression of overall averages over updates.
        This function creates a 2×1 plot:
         • Top subplot: progression of overall average scoring.
         • Bottom subplot: progression of overall average keep probability.
         
        The x-axis shows the update number, and the y-axis shows the corresponding progression value.
        """
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        updates = np.arange(1, len(self.progression_scoring) + 1)
        
        # Plot progression for scoring.
        axs[0].plot(updates, self.progression_scoring, marker='o', linestyle='-')
        axs[0].set_title("Overall Scoring Sum over an Epoch Progression")
        axs[0].set_xlabel("Update Number")
        axs[0].set_ylabel("Scoring Sum")
        axs[0].grid(True)
        
        # Plot progression for keep probability.
        axs[1].plot(updates, self.progression_keep, marker='o', linestyle='-')
        axs[1].set_title("Overall Average Keep Probability over an Epoch Progression")
        axs[1].set_xlabel("Update Number")
        axs[1].set_ylabel("Average Keep Probability")
        axs[1].grid(True)
        
        plt.tight_layout()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f"{label}_progression.png"))
            print(f"Progression plot saved to {os.path.join(save_dir, f'{label}_progression.png')}")
        plt.close(fig)


    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # For lazy-initialized buffers, if they are still empty, override them with the checkpoint values.
        for key in ['previous', 'scaling', 'scoring']:
            full_key = prefix + key
            if full_key in state_dict:
                checkpoint_val = state_dict[full_key]
                current_val = getattr(self, key)
                if current_val.numel() == 0:
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

    def update_hyperparameters(self,elasticity = None,p=None,tied_layer: Optional[nn.Module] = None,scaler =None,mask_type = None):
        """
        Update the hyperparameters of the custom dropout layer.
        elasticity: how quickly the dropout mask changes.
        p: dropout probability.
        tied_layer: the module whose output is tied to this dropout.
        scaler: scaling factor used in computing keep probability.
        mask_type: determines which method to use for computing the keep probability.
        """
        if elasticity is not None:
            self.elasticity = elasticity
        if p is not None:
            self.base_keep = 1 - p
            self.p = p
        if tied_layer is not None:
            self.tied_layer = tied_layer
        if mask_type is not None:
            self.mask_type = mask_type
        if scaler is not None:
            self.scaler = scaler

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
