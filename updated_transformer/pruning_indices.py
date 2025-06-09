# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import math
import torch
import torch.nn as nn
from functools import partial
from torch.jit import Final
from typing import Type, Optional, Iterable, Dict, List
import torch.nn.functional as F
import copy
from captum.attr import LayerConductance
from evaluate_gradients.MultiLayerConductance import MultiLayerConductance   
from evaluate_gradients.MultiLayerSensitivity import MultiLayerSensitivity
from timm.models.vision_transformer import VisionTransformer, _cfg, LayerScale
from timm.models import register_model
from timm.layers import PatchEmbed, use_fused_attn, DropPath, trunc_normal_

from updated_transformer.block import Block
from updated_transformer.mlp import Mlp

def select_pruning_indices(
    model: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    scoring_type: str = "Conductance",
    batches_num: int = 10,
    pruning_rate: float = 0.2,
    pruning_type: str = "Normalization",
    transformer: bool = False
) -> Dict[int, List[int]]:
    """
    Compute per-layer importance scores (already stored in model.scores['drop_i']),
    then prune a fraction `pruning_rate` of *neurons* (not individual weights) 
    using one of three strategies:
      1. "Normalization"  -> layer-wise z-score, then global threshold
      2. "Quota"          -> fix a count k_ℓ per-layer, then drop bottom k_ℓ within that layer
      3. "Hybrid"         -> convert to layer-wise percentiles, then global rank

    Returns:
        prune_indices: a dict mapping each layer‐index `i` to a list of neuron‐indices to remove.
    """   
    torch.cuda.empty_cache()
    model_clone = copy.deepcopy(model)
    model_clone.to(device)
    model_clone.eval()  
    #model.eval()  
    num_layers = len(model_clone.selected_layers)
    # Initialize conductances for each layer
    for i in range(num_layers):
        model_clone.scores[f"drop_{i}"] = None
    plot_freq = 10
    new_iter = iter(data_loader)
    for b in range(batches_num):
        try:
            batch = next(new_iter)
        except StopIteration:
            break
        if b % plot_freq == 0:
            print(f"Processing batch {b + 1}/{batches_num}...")
        x, _ = batch
        x_captum = x.detach().clone().requires_grad_().to(device, non_blocking=True)
        baseline = torch.zeros_like(x_captum)

        #forward + predict labels
        outputs = model_clone(x_captum)
        pred = outputs.argmax(dim=1)

   
         # choose between Conductance / Sensitivity
        if scoring_type == "Conductance":
            mlc = MultiLayerConductance(model_clone, model_clone.selected_layers)
            captum_attrs = mlc.attribute(
                x_captum, baselines=baseline, target=pred, n_steps=model_clone.n_steps
            )
        elif scoring_type == "Sensitivity":
            mlc = MultiLayerSensitivity(model_clone, model_clone.selected_layers)
            captum_attrs = mlc.attribute(
                x_captum, baselines=baseline, target=pred, n_steps=model_clone.n_steps
            )
        else:
            # fallback to Conductance
            mlc = MultiLayerConductance(model_clone, model_clone.selected_layers)
            captum_attrs = mlc.attribute(
                x_captum, baselines=baseline, target=pred, n_steps=model_clone.n_steps
            )

        # Average out the conductance across the batch and add it
        # captum_attrs is a list (length num_layers) of tensors shaped [batch_size, #neurons_in_that_layer].
        # We average them over the batch dimension before accumulating.
        for layer_idx, score_tensor in enumerate(captum_attrs):
            # if using Conductance, average across batch; if using Sensitivity, `score_tensor` is already [#neurons]
            if scoring_type == "Sensitivity":
                score_mean = score_tensor.clone()
            else:
                score_mean = score_tensor.mean(dim=0)
            if transformer:
                score_mean = score_mean.mean(dim =0)
            key = f"drop_{layer_idx}"
            if model_clone.scores[key] is None:
                model_clone.scores[key] = score_mean.clone().detach()
            else:
                model_clone.scores[key] += score_mean.detach()
           # 3) --- average the accumulated scores over `batches_num`
# -------------------------------------------------------------------------
    for i in range(num_layers):
        key = f"drop_{i}"
        if model_clone.scores[key] is not None:
            model_clone.scores[key] = model_clone.scores[key] / float(batches_num)
        else:
            raise RuntimeError(f"No scores were computed for layer {i}. Did data_loader run out of data?")
    for i in range(num_layers):
        key = f"drop_{i}"
        print(f"Layer {i} scores shape : {model_clone.scores[key].shape}")
        #print(f"Layer {i} scores: {model_clone.scores[key].mean().item():.4f} (std: {model_clone.scores[key].std().item():.4f})")
    # -------------------------------------------------------------------------
    # 4) --- collect all layer‐wise score‐tensors, compute total # of neurons
    # -------------------------------------------------------------------------
    torch.cuda.empty_cache()

    layer_scores: List[torch.Tensor] = []
    layer_sizes: List[int] = []
    for i in range(num_layers):
        scores_i = model_clone.scores[f"drop_{i}"]  # shape: [N_i]
        # flatten to compute mean/std and rank

        flat_scores = scores_i.view(-1)
        layer_scores.append(scores_i)
        layer_sizes.append(flat_scores.numel())
    del model_clone
    torch.cuda.empty_cache()
    total_neurons = sum(layer_sizes)
    N_remove = math.ceil(pruning_rate * total_neurons)

    prune_indices: Dict[int, List[int]] = {i: [] for i in range(num_layers)}

    if pruning_type.lower() == "normalization":
        candidates = []
        for i, scores in enumerate(layer_scores):
            mi = scores.mean()
            sig = scores.std(unbiased=False) + 1e-8  # avoid divide‐by‐zero
            z_scores = (scores - mi) / sig
            for neuron_idx in range(z_scores.size(0)):
                candidates.append((z_scores[neuron_idx].item(), i, neuron_idx))
        candidates.sort(key=lambda x: x[0])
        to_prune = candidates[:N_remove]
        for (_, layer_i, flat_j) in to_prune:
            shape = layer_scores[layer_i].shape
            idx_multi = tuple(int(x) for x in torch.unravel_index(torch.tensor(flat_j), shape))
            prune_indices[layer_i].append(idx_multi)

    elif pruning_type.lower() == "quota":
        k_list = [math.floor(pruning_rate * N_i) for N_i in layer_sizes]
        sum_k = sum(k_list)
        
        # 2.b) Adjust to hit exact global target
        if sum_k < N_remove:
            diff = N_remove - sum_k
            # compute average score per layer (lower avg → less important on average)
            avg_scores = [(layer_scores[i].view(-1).mean().item(), i) for i in range(num_layers)]
            # sort by ascending avg (least important first)
            avg_scores.sort(key=lambda x: x[0])
            idx = 0
            while diff > 0:
                layer_to_inc = avg_scores[idx % num_layers][1]
                k_list[layer_to_inc] += 1
                diff -= 1
                idx += 1

        elif sum_k > N_remove:
            diff = sum_k - N_remove
            # sort layers by descending average importance (we reduce from most "sensitive" layers)
            avg_scores = [(layer_scores[i].view(-1).mean().item(), i) for i in range(num_layers)]
            avg_scores.sort(key=lambda x: -x[0])
            idx = 0
            while diff > 0:
                layer_to_dec = avg_scores[idx % num_layers][1]
                if k_list[layer_to_dec] > 0:
                    k_list[layer_to_dec] -= 1
                    diff -= 1
                idx += 1
        for i, scores in enumerate(layer_scores):
            k_i = k_list[i]
            if k_i <= 0:
                continue

            flat_scores = scores.view(-1)
            sorted_idx = torch.argsort(flat_scores)  # ascending
            bottom_k = sorted_idx[:k_i].tolist()
            for flat_j in bottom_k:
                shape = scores.shape
                idx_multi = tuple(int(x) for x in torch.unravel_index(torch.tensor(flat_j), shape))
                prune_indices[i].append(idx_multi)
    
    elif pruning_type.lower() == "quotweighted" or pruning_type.lower() == "quotaweighted":
        # 3.a) Compute average importance per layer: s̄_ℓ = mean(flat_scores_ℓ)
        avg_importances = []
        for i, scores in enumerate(layer_scores):
            flat_scores = scores.view(-1)
            s_bar = flat_scores.mean().item()
            # Avoid division by zero; if a layer's average is zero, treat as very small epsilon
            if abs(s_bar) < 1e-12:
                s_bar = 1e-12
            avg_importances.append((s_bar, i))

        # 3.b) Compute weights w_ℓ = 1 / s̄_ℓ (higher s̄_ℓ → smaller weight → prune fewer)
        weights = []
        for s_bar, i in avg_importances:
            weights.append((1.0 / s_bar, i))

        # 3.c) Normalize weights so that sum of (weight_ℓ) = 1
        total_weight = sum(w for w, _ in weights)
        normalized = [(w / total_weight, i) for w, i in weights]

        # 3.d) Compute raw quotas: r_ℓ = normalized_weight_ℓ * N_remove
        raw_quotas = [(rw * N_remove, i) for rw, i in normalized]

        # 3.e) Round each to nearest integer: k_list[i] = round(r_ℓ)
        k_list = [0] * num_layers
        for rq, i in raw_quotas:
            k_list[i] = int(round(rq))

        # 3.f) Fix rounding error so sum(k_list) == N_remove
        sum_k = sum(k_list)
        if sum_k < N_remove:
            diff = N_remove - sum_k
            # Distribute extra slots to layers with smallest average importance
            avg_scores_sorted = sorted(avg_importances, key=lambda x: x[0])  # ascending s̄_ℓ
            idx = 0
            while diff > 0:
                layer_to_inc = avg_scores_sorted[idx % num_layers][1]
                k_list[layer_to_inc] += 1
                diff -= 1
                idx += 1
        elif sum_k > N_remove:
            diff = sum_k - N_remove
            # Remove extra slots from layers with largest average importance
            avg_scores_sorted = sorted(avg_importances, key=lambda x: -x[0])  # descending s̄_ℓ
            idx = 0
            while diff > 0:
                layer_to_dec = avg_scores_sorted[idx % num_layers][1]
                if k_list[layer_to_dec] > 0:
                    k_list[layer_to_dec] -= 1
                    diff -= 1
                idx += 1

        # 3.g) Finally, prune exactly k_list[i] neurons from layer i
        for i, scores in enumerate(layer_scores):
            k_i = k_list[i]
            if k_i <= 0:
                continue

            flat_scores = scores.view(-1)
            sorted_idx = torch.argsort(flat_scores)  # ascending
            bottom_k = sorted_idx[:k_i].tolist()
            for flat_j in bottom_k:
                shape = scores.shape
                idx_multi = tuple(int(x) for x in torch.unravel_index(torch.tensor(flat_j), shape))
                prune_indices[i].append(idx_multi)
    elif pruning_type.lower() == "hybrid":
        candidates = []  # list of (percentile, layer_idx, flat_idx)
        for i, scores in enumerate(layer_scores):
            flat_scores = scores.view(-1)
            N_i = flat_scores.numel()
            if N_i == 0:
                continue

            sorted_idx = torch.argsort(flat_scores)  # ascending
            # sorted_idx[j] is the flat index of j-th smallest score; percentile = (j+1)/N_i
            for rank_pos, flat_j in enumerate(sorted_idx):
                r = float(rank_pos + 1) / float(N_i)
                candidates.append((r, i, int(flat_j.item())))

        # sort ascending by percentile → lowest percentile = least important
        candidates.sort(key=lambda x: x[0])

        to_prune = candidates[:N_remove]
        for (_, layer_i, flat_j) in to_prune:
            shape = layer_scores[layer_i].shape
            idx_multi = tuple(int(x) for x in torch.unravel_index(torch.tensor(flat_j), shape))
            prune_indices[layer_i].append(idx_multi)

    else:
        raise ValueError(
            f"Unknown pruning_type '{pruning_type}'. Choose 'Normalization', 'Quota', or 'Hybrid'."
        )
    # for (i,) in prune_indices[0]:
    #     print(model.scores[f"drop_{0}"][i].item(), i)
    # ----------------------------------------------------------------------------
    # 8) --- return the dictionary of indices to prune
    # ----------------------------------------------------------------------------
    y= 0
    for key, tuple_list in prune_indices.items():
        x = []
        for tup in tuple_list:
            if y == 1:
                print(tup)
            if isinstance(tup, int):
                x.append(tup)
            elif isinstance(tup, tuple):
                x.append(tup[0])
        y=1
        prune_indices[key] = x

    return prune_indices