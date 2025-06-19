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
        if self.transformer_mean:
            scoring_final = scoring.mean(dim =0)
        else:
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
            scoring_final = -scoring_final

            epsilon = 1e-6
            s_min, s_max = scoring_final.min(), scoring_final.max()
            normalized = 2 * (scoring_final - s_min) / (s_max - s_min + epsilon) - 1
        
            flat = normalized.view(-1)
            softmax_flat = torch.softmax(flat, dim=0)
            probs = softmax_flat.view(scoring_final.shape)

            #normalize for average dropout rate close to p
            #keep_prob = power_law_rescale(raw_keep, 0.3, 1.0 - min_dropout,self.base_keep)
            raw_keep = probs * self.scaling.numel() * self.base_keep
                 

            #keep_prob = raw_keep.clamp(min=0.3, max=1.0 - min_dropout)

        elif self.mask_type == "softmax_inverse":
            # Make sure scoring is not huge in magnitude.
            epsilon = 1e-6
            s_min, s_max = scoring_final.min(), scoring_final.max()
            normalized = 2 * (scoring_final - s_min) / (s_max - s_min + epsilon) - 1
        
            flat = normalized.view(-1)
            softmax_flat = torch.softmax(flat, dim=0)
            probs = softmax_flat.view(scoring_final.shape)

            #normalize for average dropout rate close to p
            raw_keep = probs * self.scaling.numel() * self.base_keep
            #keep_prob = raw_keep.clamp(min=0.3, max=1.0)
        elif self.mask_type == "softmax_absolute":
            # Make sure scoring is not huge in magnitude.
            epsilon = 1e-6
            s_min, s_max = scoring_final.min(), scoring_final.max()
            normalized = torch.abs(2 *self.scaler* (scoring - s_min) / (s_max - s_min + epsilon) - self.scaler)

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

        # Momentum-like update
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
            #mask = torch.empty_like(input).bernoulli_(self.previous)
            #print("dropout_mask_shape:",self.previous.shape)
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
        keep_prob_det = 1-keep_prob.detach().cpu().float()
        # print("Scoring shape: ",scoring_det.shape)
        # print("Keep rate shape: ",keep_prob_det.shape)        
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
        bins_scoring = np.linspace(-0.7, 0.7, 101)  # 50 bins => 51 edges.

        hist_scoring, _ = np.histogram(scoring_det.numpy().flatten(), bins=bins_scoring)
        self.scoring_hist += hist_scoring
        for i, idx_tuple in enumerate(self.random_neurons):
            # idx_tuple is e.g. (c,h,w) or (d,)
            val = scoring_det[idx_tuple].item()
            hist_scoring_neuron, _ = np.histogram([val], bins=bins_scoring)
            self.random_neuron_hists_scoring[i] += hist_scoring_neuron
        #print("Scoring hist focused before:",self.scoring_hist_focused)

        bins_scoring_focused = np.linspace(-0.05, 0.2, 301)  # Focused histogram for scoring.
        hist_scoring_focused, _ = np.histogram(scoring_det.numpy().flatten(), bins=bins_scoring_focused)
        self.scoring_hist_focused += hist_scoring_focused
        #print("Scoring hist focused after:",self.scoring_hist_focused)
        
        bins_keep = np.linspace(0.0, 0.8, 101)

        hist_keep, _ = np.histogram(keep_prob_det.numpy().flatten(), bins=bins_keep)
        for i, idx_tuple in enumerate(self.random_neurons):
            val = keep_prob_det[idx_tuple].item()
            hist_keep_neuron, _ = np.histogram([val], bins=bins_keep)
            self.random_neuron_hists_keep[i] += hist_keep_neuron
        self.keep_hist += hist_keep
        
        # Increment the update counter.
        self.n_updates += 1
        
              
    def update_progression(self,save_dir,label=''):
        if self.sum_scoring is not None and self.running_dropout_mean is not None:
            self.progression_keep.append(self.running_dropout_mean.mean().item())
            self.progression_scoring.append(self.sum_scoring)
            np.save(os.path.join(save_dir, f"{label}_progression_keep.npy"), self.progression_keep)
            np.save(os.path.join(save_dir, f"{label}_progression_scoring.npy"), self.progression_scoring)

    def clear_progression(self):
        """Clear the progression statistics."""
        self.n_updates = 0  # Number of updates processed.
        self.running_scoring_mean = None  # Running (per-neuron) average of scoring.
        self.running_dropout_mean = None  # Running (per-neuron) average of keep probability.
        
        # Histograms (fixed 50 bins): cumulative counts for scoring and keep probability.
        self.scoring_hist = np.zeros(100)  
        self.scoring_hist_focused = np.zeros(300)  # Focused histogram for scoring.
        self.keep_hist = np.zeros(100)
        self.random_neuron_hists_scoring = [np.zeros(100) for _ in self.random_neurons ]  # List of random neuron scoring histograms.
        self.random_neuron_hists_keep = [np.zeros(100) for _ in self.random_neurons]  # List of random neuron histograms.
        
        # For progression statistics (one scalar per update).
        self.sum_scoring = None  # Cumulative sum to compute overall average scoring.
        self.sum_keep = None 
    def save_statistics(self, epoch_dir,layer_label =''):
        """
        Save all numerical statistics for a given epoch to disk in a subfolder.
        Returns the path to the epoch folder.
        """
        # Create epoch-specific folder
        # epoch_dir = os.path.join(base_dir, f"epoch_{epoch_label}")
        # os.makedirs(epoch_dir, exist_ok=True)

        # 1) Histograms
        np.save(os.path.join(epoch_dir, f"{layer_label}_scoring_hist.npy"), self.scoring_hist)
        np.save(os.path.join(epoch_dir, f"{layer_label}_keep_hist.npy"), self.keep_hist)
        np.save(os.path.join(epoch_dir, f"{layer_label}_scoring_hist_focused.npy"), self.scoring_hist_focused)

        # 2) Running means (convert to numpy)
        np.save(
            os.path.join(epoch_dir, f"{layer_label}_running_scoring_mean.npy"),
            self.running_scoring_mean.cpu().numpy()
        )
        np.save(
            os.path.join(epoch_dir, f"{layer_label}_running_dropout_mean.npy"),
            self.running_dropout_mean.cpu().numpy()
        )
        for i, neuron in enumerate(self.random_neurons):
            neuron_str = "_".join(str(x) for x in neuron)

            # Build the filenames
            scoring_fname = f"{layer_label}_random_neuron_{neuron_str}_scoring_hist.npy"
            keep_fname    = f"{layer_label}_random_neuron_{neuron_str}_keep_hist.npy"

            # Save out the arrays
            np.save(os.path.join(epoch_dir, scoring_fname),
                    self.random_neuron_hists_scoring[i])
            np.save(os.path.join(epoch_dir, keep_fname),
                    self.random_neuron_hists_keep[i])
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
        bins_scoring = np.linspace(-0.7, 0.7, 101)  # 50 bins => 51 edges.

        bin_centers_scoring = (bins_scoring[:-1] + bins_scoring[1:]) / 2
        axs[0, 0].bar(bin_centers_scoring, self.scoring_hist, width=(bins_scoring[1]-bins_scoring[0]))
        axs[0, 0].set_title(f"{epoch_label} - Scoring Histogram")
        axs[0, 0].set_xlabel("Scoring")
        axs[0, 0].set_ylabel("Count")
        
        # Histogram for keep probability.
        bins_keep = np.linspace(0.0, 0.8, 101)

        bin_centers_keep = (bins_keep[:-1] + bins_keep[1:]) / 2
        axs[0, 1].bar(bin_centers_keep, self.keep_hist, width=(bins_keep[1]-bins_keep[0]))
        axs[0, 1].set_title(f"{epoch_label} - Dropout Rate Histogram")
        axs[0, 1].set_xlabel("Dropout Rate")
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
            axs[1, 1].set_title(f"{epoch_label} - Mean Dropout Rate per Neuron")
            fig.colorbar(im1, ax=axs[1, 1])
        else:
            axs[1, 1].text(0.5, 0.5, "No Data", ha="center", va="center")
        
        plt.tight_layout()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f"{epoch_label}_aggregated_stats.png"))
        plt.close(fig)

    def plot_current_stats(self, epoch,batch_idx, save_dir=None,layer_label= 0, block = False):
        """
        Plot the aggregated statistics:
         • Two histograms:
              - Scoring histogram (50 bins, fixed range -5 to 5).
              - Keep probability histogram (50 bins, fixed range 0 to 1).
         • Two heatmaps:
              - Running per-neuron average scoring.
              - Running per-neuron average keep probability.
        """
        if block:
              block_num = layer_label//4
              layer_num = layer_label%4
              epoch_label =f'Epoch {epoch} block {block_num} layer {layer_num} sample { batch_idx//350}'
              epoch_label2 =f'Epoch_{epoch}_block{block_num}_layer_{layer_num}_sample_{ batch_idx//350}'
        else:
            epoch_label =f'Epoch {epoch} layer {layer_label} sample { batch_idx//350}'
            epoch_label2 =f'Epoch_{epoch}_layer_{layer_label}_sample_{ batch_idx//350}'

        scoring_det = self.scoring.detach().cpu().float()
        keep_prob_det = 1-self.previous.detach().cpu().float()
        score_mean = scoring_det.mean().item()
        score_std = scoring_det.std().item()
        keep_mean = keep_prob_det.mean().item()
        keep_std = keep_prob_det.std().item()
        bins_scoring = np.linspace(-0.7, 0.7, 101)  # 50 bins => 51 edges.
        hist_scoring, _ = np.histogram(scoring_det.numpy().flatten(), bins=bins_scoring)
        bins_keep = np.linspace(0.0, 0.8, 101)
        hist_keep, _ = np.histogram(keep_prob_det.numpy().flatten(), bins=bins_keep)
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Histogram for scoring.
        bin_centers_scoring = (bins_scoring[:-1] + bins_scoring[1:]) / 2
        axs[0, 0].bar(bin_centers_scoring, hist_scoring, width=(bins_scoring[1]-bins_scoring[0]))
        axs[0, 0].set_title(f"{epoch_label} - Scoring Histogram")
        axs[0, 0].set_xlabel("Scoring")
        axs[0, 0].set_ylabel("Count")
        axs[0, 0].text(
            0.05, 0.95,
            f"Mean: {score_mean:.3f}\nStd: {score_std:.3f}",
            transform=axs[0, 0].transAxes,
            va="top"
        )
        
        # Histogram for keep probability.
        bin_centers_keep = (bins_keep[:-1] + bins_keep[1:]) / 2
        axs[0, 1].bar(bin_centers_keep, hist_keep, width=(bins_keep[1]-bins_keep[0]))
        axs[0, 1].set_title(f"{epoch_label} - Dropout Histogram")
        axs[0, 1].set_xlabel("Dropout Rate")
        axs[0, 1].set_ylabel("Count")
        axs[0, 1].text(
            0.05, 0.95,
            f"Mean: {keep_mean:.3f}\nStd: {keep_std:.3f}",
            transform=axs[0, 1].transAxes,
            va="top"
        )
        
        # Heatmap for running scoring mean.
        if self.scoring is not None:
            scoring_mean_np = self.scoring.cpu().numpy()
            scoring_mean_2d = to_2d(scoring_mean_np)
            im0 = axs[1, 0].imshow(scoring_mean_2d, aspect='auto', cmap='viridis')
            axs[1, 0].set_title(f"{epoch_label} - Scoring per Neuron")
            fig.colorbar(im0, ax=axs[1, 0])
        else:
            axs[1, 0].text(0.5, 0.5, "No Data", ha="center", va="center")
        
        # Heatmap for running keep probability mean.
        if self.previous is not None:
            dropout_mean_np = 1-self.previous.cpu().numpy()
            dropout_mean_2d = to_2d(dropout_mean_np)
            im1 = axs[1, 1].imshow(dropout_mean_2d, aspect='auto', cmap='magma')
            axs[1, 1].set_title(f"{epoch_label} - Dropout Rate per Neuron")
            fig.colorbar(im1, ax=axs[1, 1])
        else:
            axs[1, 1].text(0.5, 0.5, "No Data", ha="center", va="center")
        
        plt.tight_layout()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f"{epoch_label2}_current_stats.png"))
        plt.close(fig)

    def plot_random_node_histograms_scoring(self, epoch_label, save_dir=None):
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        bins_scoring = np.linspace(-0.7, 0.7, 101)  # 50 bins => 51 edges.
        for i, neuron in enumerate(self.random_neurons):
            hist_scoring = self.random_neuron_hists_scoring[i]
            bin_centers_scoring = (bins_scoring[:-1] + bins_scoring[1:]) / 2
            axs[i // 2, i % 2].bar(bin_centers_scoring, hist_scoring, width=(bins_scoring[1]-bins_scoring[0]))
            axs[i // 2, i % 2].set_title(f"{epoch_label} Neuron {neuron} - Scoring Histogram")
            axs[i // 2, i % 2].set_xlabel("Scoring")
            axs[i // 2, i % 2].set_ylabel("Count")
        plt.tight_layout()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f"{epoch_label}_random_node_scoring_histograms.png"))
        plt.close(fig)
    def plot_random_node_histograms_keep(self, epoch_label, save_dir=None):
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        bins_keep = np.linspace(0.0, 0.8, 101)
        for i, neuron in enumerate(self.random_neurons):
            hist_scoring = self.random_neuron_hists_keep[i]
            bin_centers_scoring = (bins_keep[:-1] + bins_keep[1:]) / 2
            axs[i // 2, i % 2].bar(bin_centers_scoring, hist_scoring, width=(bins_keep[1]-bins_keep[0]))
            axs[i // 2, i % 2].set_title(f"{epoch_label} Neuron {neuron} - Dropout Rate Histogram")
            axs[i // 2, i % 2].set_xlabel("Dropout Rate")
            axs[i // 2, i % 2].set_ylabel("Count")
        plt.tight_layout()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

            fig.savefig(os.path.join(save_dir, f"{epoch_label}_random_node_dropout_histograms.png"))
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
        axs[1].set_title("Overall Average Dropout Rate over an Epoch Progression")
        axs[1].set_xlabel("Update Number")
        axs[1].set_ylabel("Average Dropout Rate")
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