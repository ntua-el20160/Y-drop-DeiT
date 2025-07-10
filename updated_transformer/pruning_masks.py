
import torch
import torch.nn as nn
from typing import List

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

def create_linear_mask(
    layer: nn.Linear,
    indices_to_zero: List[int],
    dim: int,
    weight_mask: Optional[torch.Tensor] = None,
    bias_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Build or expand a binary mask for one nn.Linear’s weight (and bias if dim=0).

    If weight_mask/bias_mask are None, we start from an all-ones mask.
    Otherwise we zero out the new indices on top of the existing mask.

    Args:
        layer:           the nn.Linear whose shape you want to mask
        indices_to_zero: which rows (dim=0) or cols (dim=1) to zero
        dim:             0 → output-rows (+bias), 1 → input-cols
        weight_mask:     existing [out, in] mask of 1/0 or None
        bias_mask:       existing [out] mask of 1/0 or None

    Returns:
        weight_mask: shape [out_features, in_features]
        bias_mask:   shape [out_features] if dim==0 & bias exists, else None
    """
    W = layer.weight.data
    if weight_mask is None:
        weight_mask = torch.ones_like(W)
    if dim == 0:
        weight_mask[indices_to_zero, :] = 0
        if layer.bias is not None:
            if bias_mask is None:
                bias_mask = torch.ones_like(layer.bias.data)
            bias_mask[indices_to_zero] = 0
    elif dim == 1:
        weight_mask[:, indices_to_zero] = 0
    else:
        raise ValueError("dim must be 0 (rows) or 1 (cols)")
    return weight_mask, bias_mask

def apply_linear_mask(
    layer: nn.Linear,
    weight_mask: torch.Tensor,
    bias_mask: Optional[torch.Tensor] = None
) -> None:
    """
    Zero out layer.weight and layer.bias according to precomputed masks.
    """
    # layer.weight.data.mul_(weight_mask)
    # if bias_mask is not None and layer.bias is not None:
    #     layer.bias.data.mul_(bias_mask)
    # 1) store masks as buffers (so they'll move with .to(), .cpu(), etc.)

    layer.register_buffer('weight_mask', weight_mask)
    if bias_mask is not None:
        layer.register_buffer('bias_mask', bias_mask)
    
    # 2) zero out any existing weights/biases
    with torch.no_grad():
        layer.weight.mul_(layer.weight_mask)
        if bias_mask is not None and layer.bias is not None:
            layer.bias.mul_(layer.bias_mask)
    
    # 3) register backward‐hook *once* to keep gradients zero on pruned entries
    if not hasattr(layer, '_mask_hooks_registered'):
        # for the weight
        layer.weight.register_hook(lambda grad: grad * layer.weight_mask)
        # for the bias (if any)
        if bias_mask is not None and layer.bias is not None:
            layer.bias.register_hook(lambda grad: grad * layer.bias_mask)
        layer._mask_hooks_registered = True

def enforce_all_masks(model: nn.Module) -> None:
    """
    After each optimizer.step(), re-apply every mask buffer in the model
    so that momentum or weight-decay can't revive pruned weights.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear) and hasattr(module, 'weight_mask'):
            with torch.no_grad():
                module.weight.mul_(module.weight_mask)
                if hasattr(module, 'bias_mask') and module.bias is not None:
                    module.bias.mul_(module.bias_mask)

def generate_prune_masks_linear_layers(model: nn.Module,
    prune_indices: List[List[int]],
    next_layer = False
) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    if len(prune_indices) ==1:
        # single linear layer
        indices = prune_indices[0]
        wm, bm = create_linear_mask(model.selected_layers[0], indices, dim=0)
        return [(wm, bm)]
    else:
        prev = []
        masks = []
        for i, layer in enumerate(model.selected_layers):
            if next_layer:
                wm1, bm1 = create_linear_mask(layer,  prev, dim=1)
                wm2, bm2 = create_linear_mask(layer, prune_indices[i], dim=0,
                                            weight_mask=wm1, bias_mask=bm1)
                prev = prune_indices[i]
            else:
                wm2, bm2 = create_linear_mask(layer, prune_indices[i], dim=0)
            masks.append((wm2, bm2))
    return masks
def generate_prune_masks_transformer(
    model: nn.Module,
    prune_indices: List[List[int]],
    next_layer = False
) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:

    masks_per_block: List[Tuple[torch.Tensor, Optional[torch.Tensor]]] = []

    for i, block in enumerate(model.blocks):
        # print(i)
        qkv_idx, proj_idx, fc1_idx, fc2_idx = prune_indices[i*4:(i+1)*4]


        # 1) QKV input cols
        # print(qkv_idx)
        # print(proj_idx)
        wm, bm = create_linear_mask(block.attn.qkv,   qkv_idx, dim=0)
        # wm,bm =torch.ones_like(block.attn.qkv.weight.data), None
        masks_per_block.append((wm, bm))
        
        # 2) Proj output rows
        wm, bm = create_linear_mask(block.attn.proj,  proj_idx, dim=0)
        #wm,bm =torch.ones_like(block.attn.proj.weight.data), None

        masks_per_block.append((wm, bm))

        # 3) MLP fc1: in cols then out rows
        if next_layer:
            wm1, bm1 = create_linear_mask(block.mlp.fc1,  proj_idx, dim=1)
            wm2, bm2 = create_linear_mask(block.mlp.fc1,  fc1_idx, dim=0,
                                        weight_mask=wm1, bias_mask=bm1)
        else:
            wm2, bm2 = create_linear_mask(block.mlp.fc1,  fc1_idx, dim=0)
        print("FC1 mask indices:", fc1_idx)
        print("FC1 mask:", wm2)
        print(bm2)
        masks_per_block.append((wm2, bm2))

        # 4) MLP fc2: in cols then out rows
        if next_layer:
            wm1, bm1 = create_linear_mask(block.mlp.fc2,  fc1_idx, dim=1)
            wm2, bm2 = create_linear_mask(block.mlp.fc2,  fc2_idx, dim=0,
                                        weight_mask=wm1, bias_mask=bm1)
        else:
            wm2, bm2 = create_linear_mask(block.mlp.fc2,  fc2_idx, dim=0)
        masks_per_block.append((wm2, bm2))


    # classification head?
    # if hasattr(model, 'head') and isinstance(model.head, nn.Linear) and next_layer:
    #     prev_head =  (None, None)
    #     wm, bm = create_linear_mask(model.head, fc2_idx, dim=1, #check if this is the correct fc2
    #                                 weight_mask=prev_head[0],
    #                                 bias_mask=prev_head[1])
    #     masks_per_block.append([(wm, bm)])

    return masks_per_block


def apply_prune_masks_transformer(
    model: nn.Module,
    masks_per_block: List[List[Tuple[torch.Tensor, Optional[torch.Tensor]]]]
) -> None:
    """
    Enforce sparsity by applying every mask in-place.
    """
    for block, block_masks in zip(model.blocks, masks_per_block):
        if block_masks[0]!= None:
            apply_linear_mask(block.attn.qkv,   *block_masks[0])
        apply_linear_mask(block.attn.proj,  *block_masks[1])
        apply_linear_mask(block.mlp.fc1,    *block_masks[2])
        apply_linear_mask(block.mlp.fc2,    *block_masks[3])

    # head?
    if len(masks_per_block) > len(model.blocks):
        apply_linear_mask(model.head, *masks_per_block[-1][0])
def apply_prune_masks_linear_layers(
    model: nn.Module,
    masks: List[Tuple[torch.Tensor, Optional[torch.Tensor]]]
) -> None:
    """
    Enforce sparsity by applying every mask in-place.
    """
    for layer, (wm, bm) in zip(model.selected_layers, masks):
        apply_linear_mask(layer, wm, bm)