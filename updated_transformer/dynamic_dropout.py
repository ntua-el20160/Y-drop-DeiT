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

        # History dictionaries.
        self.stats = {"scoring_history": [], "dropout_history": []}
        self.avg_scoring = []
        self.avg_dropout = []
        self.var_scoring = []
        self.var_dropout = []

    
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
            self.append_history(scoring, keep_prob)

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
        
    def append_history(self, scoring, dropout):
        """
        Save the current raw scoring and dropout (keep probability) values
        into the history lists.
        - scoring: a tensor of shape [channels]
        - dropout: a tensor of shape [channels] (new keep probabilities)
        """
        # Convert tensors to NumPy arrays and append a copy to the history lists.
 
        self.stats["scoring_history"].append(scoring.detach().cpu().numpy().copy())
        self.stats["dropout_history"].append(dropout.detach().cpu().numpy().copy())
    def compute_and_plot_history_statistics(self, epoch_label, save_dir=None):
        """
        Compute per-channel statistics from the saved history and generate two figures:
        
        For both scoring and dropout, the figure will have 4 subplots:
        - Top-left: Heatmap of per-channel mean values.
        - Top-right: Heatmap of per-channel variance.
        - Bottom-left: Histogram of the overall (flattened) distribution.
        - Bottom-right: Heatmap of per-channel range (max - min).
        
        Parameters:
        - epoch_label: A string (e.g. "epoch_5") used in titles and filenames.
        - save_dir: Optional directory to save the plots.
        """


        # Check if history is available.
        if len(self.stats["scoring_history"]) == 0 or len(self.stats["dropout_history"]) == 0:
            print("No history recorded yet.")
            return

        # Stack the history arrays along a new axis.
        # This will give arrays of shape: [num_updates, ...channel_dims...]
        scoring_hist = np.stack(self.stats["scoring_history"], axis=0)
        dropout_hist = np.stack(self.stats["dropout_history"], axis=0)

        # Define a helper that computes per-channel statistics.
        # It will compute mean, variance, min, max, and range along axis 0 (over updates),
        # leaving the channel dimensions intact.
        def compute_stats(history):
            # Compute the mean along the update axis.
            mean_val = np.mean(history, axis=0)
            var_val = np.var(history, axis=0)
            min_val = np.min(history, axis=0)
            max_val = np.max(history, axis=0)
            range_val = max_val - min_val
            return mean_val, var_val, min_val, max_val, range_val

        # Compute statistics for scoring and dropout.
        mean_scoring, var_scoring, min_scoring, max_scoring, range_scoring = compute_stats(scoring_hist)
        mean_dropout, var_dropout, min_dropout, max_dropout, range_dropout = compute_stats(dropout_hist)

        score_mean = scoring_hist.sum()
        dropout_mean = dropout_hist.mean()
        score_var = scoring_hist.var()
        dropout_var = dropout_hist.var()

        self.avg_dropout.append(dropout_mean)
        self.avg_scoring.append(score_mean)
        self.var_dropout.append(dropout_var)
        self.var_scoring.append(score_var)
        print("Scoring Sum: ",score_mean)
        print("Scoring Variance: ",score_var)
        print("Keep Rate Mean: ",dropout_mean)
        print("Keep Rate Variance: ",dropout_var)

        # Define a helper function for plotting a 1D or 2D metric as a heatmap.
        def plot_metric_heatmap(metric, title, cmap='viridis'):
            """
            Plot the metric as a heatmap.
            If the metric is 1D, reshape it into a 1-row heatmap.
            """
            # If metric is 1D, reshape it to (1, num_channels)
            if metric.ndim == 1:
                grid = metric.reshape(1, -1)
            else:
                grid = metric  # assume already 2D or more
            plt.imshow(grid, cmap=cmap, aspect='auto')
            plt.title(title)
            plt.colorbar()
            # Optionally, add annotations if the grid is small (for clarity).
            if grid.size <= 100:
                nrows, ncols = grid.shape
                for i in range(nrows):
                    for j in range(ncols):
                        plt.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)

        # ------------------
        # Figure for Scoring Statistics
        # ------------------
        plt.figure(figsize=(12, 10))
        
        # Subplot 1: Per-channel mean as a heatmap.
        plt.subplot(2, 2, 1)
        plot_metric_heatmap(mean_scoring, f"{epoch_label} - Scoring Mean", cmap='viridis')
        
        # Subplot 2: Per-channel variance as a heatmap.
        plt.subplot(2, 2, 2)
        plot_metric_heatmap(var_scoring, f"{epoch_label} - Scoring Variance", cmap='viridis')
        
        # Subplot 3: Histogram of overall scoring distribution.
        plt.subplot(2, 2, 3)
        plt.hist(scoring_hist.flatten(), bins=50, color='skyblue', edgecolor='black')
        plt.title(f"{epoch_label} - Overall Scoring Distribution")
        plt.xlabel("Scoring Value")
        plt.ylabel("Frequency")
        
        # Subplot 4: Per-channel range (max - min) as a heatmap.
        plt.subplot(2, 2, 4)
        plot_metric_heatmap(range_scoring, f"{epoch_label} - Scoring Range", cmap='viridis')
        
        plt.tight_layout()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"{epoch_label}_scoring_stats.png"))
        #plt.show()
        
        # ------------------
        # Figure for Dropout Statistics
        # ------------------
        plt.figure(figsize=(12, 10))
        
        # Subplot 1: Per-channel mean for dropout as a heatmap.
        plt.subplot(2, 2, 1)
        plot_metric_heatmap(mean_dropout, f"{epoch_label} - Keep Rate Mean", cmap='magma')
        
        # Subplot 2: Per-channel variance for dropout as a heatmap.
        plt.subplot(2, 2, 2)
        plot_metric_heatmap(var_dropout, f"{epoch_label} - Keep Rate Variance", cmap='magma')
        
        # Subplot 3: Histogram of overall dropout distribution.
        plt.subplot(2, 2, 3)
        dropout_data = dropout_hist.flatten()  # Use dropout_hist, not scoring_hist
        dropout_upper = np.percentile(dropout_data, 100)
        plt.hist(dropout_data, bins=50, range=(dropout_data.min(), dropout_upper), color='skyblue', edgecolor='black')
        plt.title(f"{epoch_label} - Overall Dropout Distribution (Zoomed)")
        plt.xlabel("Keep Probability")
        plt.ylabel("Frequency")

        
        # Subplot 4: Per-channel range (max - min) for dropout as a heatmap.
        plt.subplot(2, 2, 4)
        plot_metric_heatmap(range_dropout, f"{epoch_label} - Keep Rate Range", cmap='magma')
        
        plt.tight_layout()
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f"{epoch_label}_dropout_stats.png"))
        #plt.show()

    def clear_update_history(self):
        """
        Clears the raw history of scoring and dropout values.
        """
        self.stats["scoring_history"] = []
        self.stats["dropout_history"] = []
        # Optionally, clear aggregated statistics if stored.
        keys_to_clear = ["scoring_per_channel_mean", "scoring_per_channel_var",
                        "scoring_per_channel_min", "scoring_per_channel_max",
                        "dropout_per_channel_mean", "dropout_per_channel_var",
                        "dropout_per_channel_min", "dropout_per_channel_max"]
        for key in keys_to_clear:
            self.stats.pop(key, None)
        print("Update history has been cleared.")
    def plot_progression_statistics(self, save_dir=None,label=None):
        """
        Plot the progression of scoring and dropout statistics over updates.
        Creates a 2x2 grid of subplots:
          - Top-left: Scoring Mean progression.
          - Top-right: Scoring Variance progression.
          - Bottom-left: Dropout Mean progression.
          - Bottom-right: Dropout Variance progression.
          
        Parameters:
        - save_dir: Optional directory where the plot will be saved.
        """
        import matplotlib.pyplot as plt
        import os
        # Create a new figure.
        plt.figure(figsize=(12, 10))

        # Create x-axis values (update numbers).
        updates = list(range(1, len(self.avg_scoring) + 1))
        
        # Subplot 1: Scoring Mean Progression.
        plt.subplot(2, 2, 1)
        plt.plot(updates, self.avg_scoring, marker='o', linestyle='-')
        plt.title("Scoring Sum Progression")
        plt.xlabel("Update")
        plt.ylabel("Scoring Sum")
        plt.grid(True)

        # Subplot 2: Scoring Variance Progression.
        plt.subplot(2, 2, 2)
        plt.plot(updates, self.var_scoring, marker='o', linestyle='-')
        plt.title("Scoring Variance Progression")
        plt.xlabel("Update")
        plt.ylabel("Scoring Variance")
        plt.grid(True)

        # Subplot 3: Dropout Mean Progression.
        plt.subplot(2, 2, 3)
        plt.plot(updates, self.avg_dropout, marker='o', linestyle='-')
        plt.title("Keep Rate Mean Progression")
        plt.xlabel("Update")
        plt.ylabel("Keep Rate Mean")
        plt.grid(True)

        # Subplot 4: Dropout Variance Progression.
        plt.subplot(2, 2, 4)
        plt.plot(updates, self.var_dropout, marker='o', linestyle='-')
        plt.title("Keep Rate Variance Progression")
        plt.xlabel("Update")
        plt.ylabel("Keep Rate Variance")
        plt.grid(True)

        plt.tight_layout()
        
        # Save the plot if a save directory is provided.
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plot_path = os.path.join(save_dir, label +"_progression_stats.png")
            plt.savefig(plot_path)
            print(f"Progression statistics plot saved to {plot_path}")
        # Optionally, you can display the plot.
        # plt.show()
        plt.close()



    def reset_dropout_masks(self):
        """Reset the dropout masks to their default values."""
        if self.initialized:
            # Instead of removing the buffers, reset them to the default (1 - p)
            self.previous.fill_(1 - self.p)
            self.scaling.fill_(1 - self.p)
        return

    def update_parameters(self,elasticity = None,p=None):
        """Update the hyperparameters of the custom dropout layer."""
        if elasticity is not None:
            self.elasticity = elasticity
        if p is not None:
            self.p = p
            self.base_keep = 1 - p
            self.beta = torch.log(torch.tensor(self.base_keep / (1 - self.base_keep), dtype=self.previous.dtype, device=self.previous.device))
        return
    def base_dropout(self):
        """Use the standard dropout."""
        self.base = True
        return
    def custom_dropout(self):
        """Use the custom dropout."""
        self.base = False
        return


