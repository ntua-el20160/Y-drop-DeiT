import torch
import torch.nn as nn
from typing import List

# def prune_linear_layer(layer: nn.Linear, indices_to_prune: List[int], dim: int) -> nn.Linear:
#     """
#     Prunes a nn.Linear layer by removing rows (dim=0) or columns (dim=1) at indices_to_prune.
#     """
#     # Original weight and bias
#     W = layer.weight.data  # [out_features, in_features]
#     b = layer.bias.data if layer.bias is not None else None

#     if dim == 0:
#         # prune output features (rows)
#         mask = torch.ones(W.size(0), dtype=torch.bool, device=W.device)
#         mask[indices_to_prune] = False
#         W_new = W[mask, :]
#         out_features_new = W_new.size(0)
#         new_layer = nn.Linear(layer.in_features, out_features_new, bias=(b is not None))
#         new_layer.weight.data.copy_(W_new)
#         if b is not None:
#             new_layer.bias.data.copy_(b[mask])

#     elif dim == 1:
#         # prune input features (columns)
#         mask = torch.ones(W.size(1), dtype=torch.bool, device=W.device)
#         mask[indices_to_prune] = False
#         W_new = W[:, mask]
#         in_features_new = W_new.size(1)
#         new_layer = nn.Linear(in_features_new, layer.out_features, bias=(b is not None))
#         new_layer.weight.data.copy_(W_new)
#         if b is not None:
#             new_layer.bias.data.copy_(b)

#     else:
#         raise ValueError("dim must be 0 (output) or 1 (input) to prune")

#     return new_layer
import torch
import torch.nn as nn
from typing import List

def prune_linear_layer(layer: nn.Linear, indices_to_zero: List[int], dim: int) -> nn.Linear:
    """
    Returns a new nn.Linear with the same in/out features as `layer`,
    but with all weights (and affected biases) at `indices_to_zero` set to zero
    along dimension `dim`.

    Args:
        layer:         the original Linear layer
        indices_to_zero:  list of row-indices (if dim=0) or col-indices (if dim=1)
        dim:           0 to zero-output-rows (and their biases),
                       1 to zero-input-columns

    Returns:
        new_layer: an nn.Linear with masked weights
    """
    W = layer.weight.data    # [out_features, in_features]
    b = layer.bias.data if layer.bias is not None else None

    # create same-shaped layer
    new_layer = nn.Linear(layer.in_features, layer.out_features, bias=(b is not None))
    new_layer.weight.data.copy_(W)
    if b is not None:
        new_layer.bias.data.copy_(b)

    if dim == 0:
        # zero entire output rows, and zero their biases
        new_layer.weight.data[indices_to_zero, :] = 0
        if b is not None:
            new_layer.bias.data[indices_to_zero] = 0

    elif dim == 1:
        # zero entire input columns
        new_layer.weight.data[:, indices_to_zero] = 0

    else:
        raise ValueError("dim must be 0 (rows) or 1 (columns) to zero")

    return new_layer


def prune_block(block: nn.Module,
                prune_idx_list: List[List[int]],
                prev_rep_pruned: List[int] = None,check =1) -> List[int]:
    """
    Prune a single transformer Block in-place.

    Args:
        block: your Block instance containing .attn and .mlp
        prune_idx_list: [qkv_idx, proj_idx, fc1_idx, fc2_idx]
        prev_rep_pruned: indices already pruned from the representation

    Returns:
        Updated list of pruned representation indices after this block.
    """
    prev_rep_pruned = prev_rep_pruned or []

    # 1) QKV: prune input dims on representation
    # qkv_idx = prune_idx_list[0]
    # rep_pruned = sorted(set(prev_rep_pruned + qkv_idx))
    # block.attn.qkv = prune_linear_layer(block.attn.qkv, rep_pruned, dim=1)
    
    # old_out = block.attn.qkv.weight.size(0)  # still 3*Cold
    # old_C = old_out // 3
    # rows_to_prune = []
    # for seg in range(3):
    #     offset = seg * old_C
    #     rows_to_prune += [offset + idx for idx in rep_pruned]
    # block.attn.qkv = prune_linear_layer(block.attn.qkv, rows_to_prune, dim=0)

    # now both in_features and out_features are consistent: 
    # in_features = Cnew, out_features = 3*Cnew
    Cnew = block.attn.qkv.in_features

    # ── Update attention internals ──
    block.attn.head_dim = Cnew // block.attn.num_heads
    block.attn.scale = block.attn.head_dim ** -0.5

    if not isinstance(block.attn.q_norm, nn.Identity):
        block.attn.q_norm = block.attn.q_norm.__class__(block.attn.head_dim)
    if not isinstance(block.attn.k_norm, nn.Identity):
        block.attn.k_norm = block.attn.k_norm.__class__(block.attn.head_dim)

    # 2) Proj: prune input (representation) then output dims
    proj_idx = prune_idx_list[1]
    #block.attn.proj = prune_linear_layer(block.attn.proj, rep_pruned, dim=1)
    block.attn.proj = prune_linear_layer(block.attn.proj, proj_idx, dim=0)
    # 3) MLP fc1: prune representation dims on input then output
    fc1_idx = prune_idx_list[2]
    block.mlp.fc1 = prune_linear_layer(block.mlp.fc1, proj_idx, dim=1)
    block.mlp.fc1 = prune_linear_layer(block.mlp.fc1, fc1_idx, dim=0)
    hidden_pruned = fc1_idx
    Cnew2 = block.mlp.fc1.in_features  # should be same as Cnew

    
    # 4) MLP fc2: prune hidden dims on input then representation dims on output
    fc2_idx = prune_idx_list[3]
    block.mlp.fc2 = prune_linear_layer(block.mlp.fc2, hidden_pruned, dim=1)
    block.mlp.fc2 = prune_linear_layer(block.mlp.fc2, fc2_idx, dim=0)

    # Update LayerNorms to match new channel count
    block.norm1 = nn.LayerNorm(Cnew)
    block.norm2 = nn.LayerNorm(Cnew2)

    return fc2_idx


def prune_model(model: nn.Module, prune_indices):
    """
    Prune all blocks of the Vision Transformer in-place using a flat index list.

    Args:
        model: instance of MyVisionTransformer with .blocks iterable
        prune_idx_flat: flat list of length num_blocks*4, grouping every 4 lists as
                         [qkv_idx, proj_idx, fc1_idx, fc2_idx] per block.
    """
    num_blocks = len(prune_indices) // 4
    prev_rep_pruned: List[int] = []
    check = 0
    for i, block in enumerate(model.blocks):
        # slice out the 4 sub-lists for this block
        block_idxs = [prune_indices[i * 4 + j] for j in range(4)]
        prev_rep_pruned = prune_block(block, block_idxs, prev_rep_pruned,check)
        check+=1

    # Adjust classification head if present
    if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        model.head = prune_linear_layer(model.head, prev_rep_pruned, dim=1)
