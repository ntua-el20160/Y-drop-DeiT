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
def calculate_scores(
        model: torch.nn.Module,
        batches: Iterable,
        device: torch.device,
        scoring_type: str = "Conductance",
        transformer: bool = False,
        normalization: bool = True,
        selected_layers: Optional[List[int]] = None,
        sm = True) -> Dict[int, torch.Tensor]:
    # 1) --- ensure model is in eval mode and gradients are disabled
    model.eval()
    if selected_layers is  None:
        selected_layers = model.selected_layers

    # 2) --- save original requires_grad settings
    orig_reqs = []
    for p in model.parameters():
        orig_reqs.append(p.requires_grad)
        p.requires_grad_(False)
    model.zero_grad()
    new_scores = {}

    # 3) --- select scoring type for the layers
    if scoring_type == "Conductance":
        mlc = MultiLayerConductance(model, selected_layers)
    elif scoring_type == "Conductance_alt":
        mlc = MultiLayerConductance(model.crit_for,selected_layers)
    elif scoring_type == "Sensitivity":
        mlc = MultiLayerSensitivity(model,selected_layers)
    else:
        print("Invalid scoring type. Using Conductance as default.")
        mlc = MultiLayerConductance(model, selected_layers)

    # 4) --- iterate over batches
    for x, y_batch in batches:
            # 5) --- ensure x is on the correct device and requires_grad
            x_captum = x.detach().clone().requires_grad_()
            x_captum = x_captum.to(device, non_blocking=True)
            baseline = torch.zeros_like(x_captum)
            y_batch = y_batch.to(device, non_blocking=True).long()

            # 6) --- forward pass and predict labels
            outputs = model(x_captum)
            pred = outputs.argmax(dim=1)
            # 7) --- compute captum attributes
            if scoring_type == "Conductance_alt":
                captum_out = mlc.attribute(
                    x_captum, baselines=baseline, target=None,
                    n_steps=model.n_steps,
                    internal_batch_size=None,
                    additional_forward_args=(y_batch,),
                    return_convergence_delta=False,
                    attribute_to_layer_input=False,
                    grad_kwargs={"retain_graph": False},
                )
            else:
                captum_out = mlc.attribute(
                x_captum, baselines=baseline, target=pred,
                n_steps=model.n_steps,
                internal_batch_size=None,
                return_convergence_delta=False,
                attribute_to_layer_input=False,
                grad_kwargs={"retain_graph": False},
            )
            # 8) --- process captum output
            if isinstance(captum_out, list):
                captum_attrs = [t.detach() for t in captum_out]
            elif isinstance(captum_out, tuple):
                captum_attrs = tuple(t.detach() for t in captum_out)
            else:
                captum_attrs = [captum_out.detach()]  
            # 9) --- accumulate scores
            for i, score in enumerate(captum_attrs):
                #Sensetivity no batch dimension
                if scoring_type == "Sensitivity":
                    score_mean = score
                #sum
                elif sm:
                    score_mean = score.sum(dim =0)
                else:
                    score_mean = score.mean(dim=0)
                if transformer:
                    score_mean = score_mean.sum(dim =0)

                if i not in new_scores:
                    # First time: initialize with the computed score_mean
                    new_scores[i] = score_mean.clone()
                else:
                    # Accumulate the score_mean
                    new_scores[i] += score_mean
    # 10) compute means for each layer
    num_batches = len(list(batches))
    for i in new_scores:
        new_scores[i] /= num_batches
        #print(f"Layer {i} score shape: {new_scores[i].shape}")

    means = [s.mean() for s in new_scores.values() if s is not None]
    
    # 11) --- normalize scores if required
    if normalization:
        for i in range(len(new_scores)):
            if new_scores[i] is not None:
                new_scores[i] = (new_scores[i] - new_scores[i].mean()) / new_scores[i].std()
    
    # 12) --- restore original requires_grad settings
    for p, req in zip(model.parameters(), orig_reqs):
            p.requires_grad_(req)
    # torch.cuda.empty_cache()


    model.train()
    return new_scores,means

def update_dropout_masks(
    model: torch.nn.Module,
    scores: Dict[int, torch.Tensor],
    drop_list = None,
    alt_attention_cond: bool = False,
    stats: bool = False,
    min_dropout: float = 0.0,
    noisy_dropout: bool = False
):
    if drop_list is None:
        drop_list = model.drop_list

    for i, drop_layer in enumerate(drop_list):
        score = scores[i]

        if alt_attention_cond and (i % 4 == 0):
            N = score.shape[0]  # Number of tokens
            qkv = score.reshape(N, 3, model.blocks[i // 4].attn.num_heads, model.blocks[i // 4].attn.head_dim).permute(1, 2, 0, 3)
            q, k, v = qkv.unbind(0)
            q, k = model.blocks[i // 4].attn.q_norm(q), model.blocks[i // 4].attn.k_norm(k)
            q = q * model.blocks[i // 4].attn.scale
            score = q @ k.transpose(-2, -1)

            
        drop_layer.update_dropout_masks(score, stats=stats,noisy = noisy_dropout,min_dropout=min_dropout)

def accumulated_scores_uncertainty(
        acc_scores: Dict[int, torch.Tensor] = None,
        new_scores: Dict[int, torch.Tensor] = None,
        w_avg_rate : float = 0.05,
        uncertainty: bool = False,
        acc_uncertainties: Dict[int, torch.Tensor] = None,):
    
    if acc_scores is None:
        acc_scores = { i: s.clone() for i,s in new_scores.items() }
        if uncertainty:
            acc_uncertainties = {i: torch.zeros_like(new_scores[i]) for i in new_scores.keys()}
    else:
        for i,score in acc_scores.items():
            acc_scores[i] = score*(1-w_avg_rate) + new_scores[i] * w_avg_rate
            if uncertainty:
                new_unc =  torch.abs(acc_scores[i]- new_scores[i])  ## maybe divide by something to make it a rate
                acc_uncertainties[i] = acc_uncertainties[i]*(1-w_avg_rate) + new_unc * w_avg_rate
    return acc_scores, acc_uncertainties if uncertainty else None







                     
def select_pruning_indices(
    scores: Dict[int, torch.Tensor],
    pruning_rate: float = 0.2,
    pruning_type: str = "normalization",
    prune_indices: Optional[Dict[int, List[int]]] = None,
    means: Optional[List[float]] = None
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

    num_layers = len(scores)

    # 1-initialize or normalize incoming prune_indices
    if prune_indices is None:
        prune_indices = {i: [] for i in range(num_layers)}
    else:
        # make sure every layer has a list, even if empty
        for i in range(num_layers):
            prune_indices.setdefault(i, [])    
    
    existing_flat = {
        i: set(prune_indices[i])
        for i in range(num_layers)
    }

    layer_scores: List[torch.Tensor] = []
    layer_sizes: List[int] = []
    # 2) --- collect scores for each layer flattened
    for i,score in scores.items():

        flat_scores = score.view(-1)
        layer_scores.append(flat_scores)
        layer_sizes.append(flat_scores.numel())
    
    # 3) --- calculate total number of neurons to prune
    total_neurons = sum(layer_sizes)
    N_remove = math.ceil(pruning_rate * total_neurons)

    # 4) --- Normalization: compute z-scores for each layer
    if pruning_type.lower() == "normalization":
        candidates = []
        # 4.a) Compute z-scores for each layer  
        for i, scores in enumerate(layer_scores):
            mi = scores.mean()
            sig = scores.std(unbiased=False) + 1e-8  # avoid divide‐by‐zero
            z = (scores - mi) / sig
           
            # 4.b) Collect candidates for pruning not already in prune_indices
            for idx in range(z.size(0)):
                if idx in existing_flat[i]:
                    continue
                candidates.append((z[idx].item(), i, idx))

        # 4.c) Sort candidates by z-score (ascending) → lowest z-score = least important
        candidates.sort(key=lambda x: x[0])
        
        # 4.d) Select bottom N_remove candidates
        picks = candidates[:N_remove]
        
    
        for _, layer_i, flat_j in picks:
            prune_indices[layer_i].append(flat_j)
        # 4.e) Convert flat indices to multi-dimensional indices and store in prune_indices
    
        # for (_, layer_i, flat_j) in to_prune:
        #     shape = layer_scores[layer_i].shape
        #     idx_multi = tuple(int(x) for x in torch.unravel_index(torch.tensor(flat_j), shape))
        #     prune_indices[layer_i].append(idx_multi)
    # ---5) --- Quota: compute quotas per layer
    elif pruning_type.lower() == "quota":
        # 2.a) Compute k_i = floor(pruning_rate * N_i) for each layer
        k_list = [math.floor(pruning_rate * n) for n in layer_sizes]
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
        
        for i, flat_scores  in enumerate(layer_scores):
            k_i = k_list[i]
            if k_i <= 0:
                continue

            sorted_idx = torch.argsort(flat_scores)  # ascending
            eligible = [j for j in sorted_idx if j not in existing_flat[i]]
            for flat_j in eligible[:k_i]:
                prune_indices[i].append(flat_j)

    # 6) --- Quota-weighted: compute quotas based on average importance per layer
    elif pruning_type.lower() == "quotweighted" or pruning_type.lower() == "quotaweighted":
        # 6.a) calculate means if not provided
        for i in range(4):
            print("Mean of layer {1} is {0}".format(means[i], i))
        if means is None:
            means = [scores[i].view(-1).mean().item() for i in range(num_layers)]
        print()
        # 6.b) Compute average importance per layer: s̄_ℓ = mean(flat_scores_ℓ)
        avg_importances = []
        for i, mean in enumerate(means):
            avg_importances.append((mean+torch.finfo(torch.float32).eps, i))
        
        # 6.c) Compute weights w_ℓ = 1 / s̄_ℓ (higher s̄_ℓ → smaller weight → prune fewer)
        weights = []
        for s_bar, i in avg_importances:
            weights.append((1.0 / s_bar, i))

        # 6.d) Normalize weights so that sum of (weight_ℓ) = 1
        total_weight = sum(w for w, _ in weights)
        normalized = [(w / total_weight, i) for w, i in weights]
        # for i in range(4):
        #     print("Importance of layer {1} is {0}".format(normalized[i], i))

        # 6.e) Compute raw quotas: r_ℓ = normalized_weight_ℓ * N_remove
        raw_quotas = [(rw * N_remove, i) for rw, i in normalized]

        # 6.f) Round each to nearest integer: k_list[i] = round(r_ℓ)
        k_list = [0] * num_layers
        for rq, i in raw_quotas:
            val = rq.item() if isinstance(rq, torch.Tensor) else rq
            k_list[i] = int(round(val))

        # 6.g) Fix rounding error so sum(k_list) == N_remove
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

        # 6.h) Finally, prune exactly k_list[i] neurons from layer i
        for i, scores in enumerate(layer_scores):
            if i < 4:
                print("hi")
            k_i = k_list[i]
            if k_i <= 0:
                continue

            sorted_idx = torch.argsort(scores).tolist()  # ascending
            eligible = [j for j in sorted_idx if j not in existing_flat[i]]  # already pruned indices

            for flat_j in eligible[:k_i]:
                prune_indices[i].append(flat_j)
            # if i<4:
            #     print("score of layer {1} is {0}".format(scores, i))
            #     print("Pruning layer {1} at index {0}".format(prune_indices[i], i))

            # if i <4:
            #     print("score of layer {1} is {0}".format(scores, i))

            #     print("Pruning layer {1} at index {0}".format(prune_indices[i], i))

            # for flat_j in bottom_k:
            #     shape = scores.shape
            #     idx_multi = tuple(int(x) for x in torch.unravel_index(torch.tensor(flat_j), shape))
            #     prune_indices[i].append(idx_multi)
    # 7) --- Hybrid: convert to percentiles and prune lowest N_remove
    elif pruning_type.lower() == "hybrid":
        candidates = []  # list of (percentile, layer_idx, flat_idx)
        for i, flat_scores in enumerate(layer_scores):
            N_i = flat_scores.numel()
            sorted_idx = torch.argsort(flat_scores).tolist()

              # ascending
            # sorted_idx[j] is the flat index of j-th smallest score; percentile = (j+1)/N_i
            i = 0
            for rank, flat_j in enumerate(sorted_idx):
                if flat_j in existing_flat[i]:
                    continue
                pct = (rank + 1) / N_i
                candidates.append((pct, i, flat_j))
                if i <4:
                    print("score of layer {1} is {0}".format(flat_scores, i))
                i+=1

        # sort ascending by percentile → lowest percentile = least important
        candidates.sort(key=lambda x: x[0])
        i=0

        for _, layer_i, flat_j in candidates[:N_remove]:
            prune_indices[layer_i].append(flat_j)

    else:
        raise ValueError(
            f"Unknown pruning_type '{pruning_type}'. Choose 'Normalization', 'Quota', or 'Hybrid'."
        )

    for i in range(num_layers):
        seen = set()
        uniq = []
        for j in prune_indices[i]:
            if j not in seen:
                seen.add(j)
                uniq.append(j)
        prune_indices[i] = uniq

    return prune_indices
def expand_prune_indices(
    flat_prune_indices: Dict[int, List[int]],
    scores: Dict[int, torch.Tensor]
) -> Dict[int, List[tuple]]:
    """
    Take the flat‐integer prune indices per layer, and convert each
    back into a multi-dimensional index tuple using the original score shapes.
    """
    expanded: Dict[int, List[tuple]] = {}
    for layer_i, flats in flat_prune_indices.items():
        shape = scores[layer_i].shape
        expanded[layer_i] = [
            tuple(int(x) for x in torch.unravel_index(torch.tensor(f), shape))
            for f in flats
        ]
    return expanded

    # for (i,) in prune_indices[0]:
    #     print(model.scores[f"drop_{0}"][i].item(), i)
    # ----------------------------------------------------------------------------
    # 8) --- return the dictionary of indices to prune
    # ----------------------------------------------------------------------------
    # y= 0
    # for key, tuple_list in prune_indices.items():
    #     x = []
    #     for tup in tuple_list:
    #         # if y == 1:
    #         #     print(tup)
    #         if isinstance(tup, int):
    #             x.append(tup)
    #         elif isinstance(tup, tuple):
    #             x.append(tup[0])
    #     y=1
    #     prune_indices[key] = x

    # return prune_indices