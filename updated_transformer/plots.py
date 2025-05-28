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

def plot_epoch_statistics(base_path, epoch, save_dir=None):
    """
    Load and plot all layer stats for a given epoch.
    
    Parameters
    ----------
    base_path : str
        Root path where `plots/epoch_<epoch>_data/` lives.
    epoch : int
        Epoch index (matches the suffix in `epoch_<epoch>_data`).
    save_dir : str or None
        If provided, all plots will be written here; otherwise they
        overwrite back into the epoch folder.
    """
    # locate the folder
    epoch_dir = os.path.join(base_path, f"plots/epoch_{epoch}_data")
    if not os.path.isdir(epoch_dir):
        raise FileNotFoundError(f"No such directory: {epoch_dir}")
    if save_dir is None:
        save_dir = epoch_dir

    layer_idx = 0
    while True:
        layer_label = f"layer{layer_idx}"
        # check for the key histogram to decide when to stop
        scoring_path = os.path.join(epoch_dir, f"{layer_label}_scoring_hist.npy")
        if not os.path.exists(scoring_path):
            break

        # load aggregated stats
        keep_path = os.path.join(epoch_dir, f"{layer_label}_keep_hist.npy")
        run_score_path = os.path.join(epoch_dir, f"{layer_label}_running_scoring_mean.npy")
        run_keep_path  = os.path.join(epoch_dir, f"{layer_label}_running_dropout_mean.npy")

        scoring_hist          = np.load(scoring_path)
        keep_hist             = np.load(keep_path)
        running_scoring_mean  = torch.from_numpy(np.load(run_score_path))
        running_dropout_mean  = torch.from_numpy(np.load(run_keep_path))

        # find all random‐neuron hist files for this layer
        pattern = os.path.join(epoch_dir, f"{layer_label}_random_neuron_*_scoring_hist.npy")
        scoring_files = sorted(glob.glob(pattern))
        random_neurons = []
        random_hists_scoring = []
        random_hists_keep    = []
        for sf in scoring_files:
            # extract neuron index from filename
            basename = os.path.basename(sf)
            # e.g. layer0_random_neuron_42_scoring_hist.npy
            parts = basename.split('_')
            neuron = int(parts[3])
            random_neurons.append(neuron)
            random_hists_scoring.append(np.load(sf))

            keep_file = os.path.join(
                epoch_dir,
                f"{layer_label}_random_neuron_{neuron}_keep_hist.npy"
            )
            random_hists_keep.append(np.load(keep_file))

        # call your plotting fns
        epoch_label = f"Epoch {epoch} {layer_label}"
        plot_aggregated_statistics(
            epoch_label,
            scoring_hist,
            keep_hist,
            running_scoring_mean,
            running_dropout_mean,
            save_dir=save_dir
        )
        if random_neurons:
            plot_random_node_histograms_scoring(
                random_neurons,
                random_hists_scoring,
                epoch_label,
                save_dir=save_dir
            )
            plot_random_node_histograms_keep(
                random_neurons,
                random_hists_keep,
                epoch_label,
                save_dir=save_dir
            )

        layer_idx += 1
def plot_aggregated_statistics(epoch_label,scoring_hist,keep_hist,running_scoring_mean,running_dropout_mean, save_dir=None):
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
        axs[0, 0].bar(bin_centers_scoring, scoring_hist, width=(bins_scoring[1]-bins_scoring[0]))
        axs[0, 0].set_title(f"{epoch_label} - Scoring Histogram")
        axs[0, 0].set_xlabel("Scoring")
        axs[0, 0].set_ylabel("Count")
        
        # Histogram for keep probability.
        bins_keep = np.linspace(0.0, 0.8, 101)

        bin_centers_keep = (bins_keep[:-1] + bins_keep[1:]) / 2
        axs[0, 1].bar(bin_centers_keep, keep_hist, width=(bins_keep[1]-bins_keep[0]))
        axs[0, 1].set_title(f"{epoch_label} - Dropout Rate Histogram")
        axs[0, 1].set_xlabel("Dropout Rate")
        axs[0, 1].set_ylabel("Count")
        
        # Heatmap for running scoring mean.
        if running_scoring_mean is not None:
            scoring_mean_np = running_scoring_mean.cpu().numpy()
            scoring_mean_2d = to_2d(scoring_mean_np)
            im0 = axs[1, 0].imshow(scoring_mean_2d, aspect='auto', cmap='viridis')
            axs[1, 0].set_title(f"{epoch_label} - Mean Scoring per Neuron")
            fig.colorbar(im0, ax=axs[1, 0])
        else:
            axs[1, 0].text(0.5, 0.5, "No Data", ha="center", va="center")
        
        # Heatmap for running keep probability mean.
        if running_dropout_mean is not None:
            dropout_mean_np = running_dropout_mean.cpu().numpy()
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
def plot_random_node_histograms_scoring(random_neurons,random_neuron_hists_scoring, epoch_label, save_dir=None):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    bins_scoring = np.linspace(-0.7, 0.7, 101)  # 50 bins => 51 edges.
    for i, neuron in enumerate(random_neurons):
        hist_scoring = random_neuron_hists_scoring[i]
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

def plot_random_node_histograms_keep(random_neurons,random_neuron_hists_keep, epoch_label, save_dir=None):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    bins_keep = np.linspace(0.0, 0.8, 101)
    for i, neuron in enumerate(random_neurons):
        hist_scoring = random_neuron_hists_keep[i]
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