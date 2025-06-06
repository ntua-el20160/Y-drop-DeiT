import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from timm.models.vision_transformer import  LayerScale

def prune_vit_blocks(
    model: nn.Module,
    prune_map: Dict[int, List[int]]
) -> None:
    """
    Prune each Block in model.blocks according to a flat prune_map:
      • prune_map[b*4 + 0]: head indices to drop from block.attn.qkv
      • prune_map[b*4 + 1]: row indices to drop from block.attn.proj
      • prune_map[b*4 + 2]: row indices to drop from block.mlp.fc1
      • prune_map[b*4 + 3]: row indices to drop from block.mlp.fc2

    After pruning block b, we rebuild all LayerNorm and LayerScale modules
    in that block to use the new embedding dimension. We also rebuild block
    b+1’s norm1 and qkv (and LayerScale) to accept the new dimension. Finally,
    we rebuild model.norm, model.fc_norm, and model.head to match the last
    block’s output dimension.

    This modifies `model` in-place; it does not return anything.
    """

    model.eval()

    # ------------------------------------------------------------
    # Helper to locate (parent_module, attribute_name) for a given child submodule
    # so we can do: setattr(parent_module, attribute_name, new_module)
    # ------------------------------------------------------------
    def _find_parent_and_attr(root: nn.Module, child: nn.Module) -> Tuple[nn.Module, str]:
        for qualified_name, module in root.named_modules():
            if module is child:
                if "." not in qualified_name:
                    return root, qualified_name
                parent_name, attr_name = qualified_name.rsplit(".", 1)
                parent = root
                for part in parent_name.split("."):
                    parent = getattr(parent, part)
                return parent, attr_name
        raise ValueError(f"Module {child} not found in model as a submodule.")

    # ------------------------------------------------------------
    # Helper: prune a Linear's outputs (rows). Returns (new_linear, keep_mask).
    # ------------------------------------------------------------
    def _prune_linear_outputs(orig: nn.Linear, prune_indices: List[int]) -> Tuple[nn.Linear, torch.Tensor]:
        orig_out, orig_in = orig.out_features, orig.in_features
        device = orig.weight.device

        # 1) Build a boolean mask of length=orig_out (True=keep, False=prune)
        keep_mask = torch.ones(orig_out, dtype=torch.bool, device=device)
        keep_mask[prune_indices] = False
        new_out = int(keep_mask.sum().item())

        # 2) Create new Linear(in_features=orig_in, out_features=new_out)
        new_lin = nn.Linear(in_features=orig_in, out_features=new_out,
                            bias=(orig.bias is not None)).to(device)

        # 3) Copy surviving rows of weight & bias
        with torch.no_grad():
            new_lin.weight.copy_(orig.weight[keep_mask, :])
            if orig.bias is not None:
                new_lin.bias.copy_(orig.bias[keep_mask])

        return new_lin, keep_mask

    # ------------------------------------------------------------
    # Helper: prune a Linear's inputs (columns) given a boolean mask over in_features.
    # Returns new nn.Linear.
    # ------------------------------------------------------------
    def _prune_linear_inputs(orig: nn.Linear, keep_mask: torch.Tensor) -> nn.Linear:
        orig_out, orig_in = orig.out_features, orig.in_features
        device = orig.weight.device

        if keep_mask.numel() != orig_in:
            raise ValueError(f"Input-mask length {keep_mask.numel()} != orig in_features {orig_in}")

        new_in = int(keep_mask.sum().item())
        new_lin = nn.Linear(in_features=new_in, out_features=orig_out,
                            bias=(orig.bias is not None)).to(device)

        with torch.no_grad():
            kept_cols = keep_mask.nonzero(as_tuple=False).squeeze(1)
            new_lin.weight.copy_(orig.weight[:, kept_cols])
            if orig.bias is not None:
                new_lin.bias.copy_(orig.bias)

        return new_lin

    # ------------------------------------------------------------
    # Helper: prune QKV (nn.Linear(dim → 3*dim)) by dropping entire heads.
    # Returns (new_qkv, keep_heads_mask, new_num_heads).
    # ------------------------------------------------------------
    def _prune_qkv(
        orig_qkv: nn.Linear,
        heads_to_prune: List[int],
        num_heads: int,
        head_dim: int
    ) -> Tuple[nn.Linear, torch.Tensor, int]:
        device = orig_qkv.weight.device
        orig_dim = num_heads * head_dim

        # 1) Build a per-head boolean mask
        keep_heads_mask = torch.ones(num_heads, dtype=torch.bool, device=device)
        keep_heads_mask[heads_to_prune] = False
        new_num_heads = int(keep_heads_mask.sum().item())
        new_dim = new_num_heads * head_dim

        # 2) Expand that to a per-dimension mask of length=orig_dim
        head_mask_flat = keep_heads_mask.repeat_interleave(head_dim)  # [orig_dim]

        # 3) Split weight & bias into three blocks (Q, K, V)
        W = orig_qkv.weight.data     # shape [3*orig_dim, orig_dim]
        B = orig_qkv.bias.data if orig_qkv.bias is not None else None

        new_weights = []
        new_biases = []

        for block_idx in range(3):
            rs = block_idx * orig_dim
            re = (block_idx + 1) * orig_dim
            W_block = W[rs:re, :]                  # [orig_dim, orig_dim]
            W_block = W_block[head_mask_flat, :]   # keep rows → [new_dim, orig_dim]
            W_block = W_block[:, head_mask_flat]   # keep cols → [new_dim, new_dim]
            new_weights.append(W_block)

            if B is not None:
                b_block = B[rs:re]                # [orig_dim]
                b_block = b_block[head_mask_flat] # [new_dim]
                new_biases.append(b_block)

        W_new = torch.cat(new_weights, dim=0)     # [3*new_dim, new_dim]
        B_new = torch.cat(new_biases, dim=0) if B is not None else None

        # 4) Create new nn.Linear(in_features=new_dim, out_features=3*new_dim)
        new_qkv = nn.Linear(
            in_features=new_dim,
            out_features=3 * new_dim,
            bias=(orig_qkv.bias is not None)
        ).to(device)

        with torch.no_grad():
            new_qkv.weight.copy_(W_new)
            if B_new is not None:
                new_qkv.bias.copy_(B_new)

        return new_qkv, keep_heads_mask, new_num_heads

    # ------------------------------------------------------------
    # 1) Gather the blocks and sanity-check prune_map length
    # ------------------------------------------------------------
    blocks = list(model.blocks)  # assume model.blocks is an nn.Sequential
    num_blocks = len(blocks)

    expected_len = num_blocks * 4
    if any(i not in prune_map for i in range(expected_len)):
        raise ValueError(f"prune_map must have keys 0..{expected_len-1}, but got {list(prune_map.keys())}")

    # ------------------------------------------------------------
    # 2) Iterate over each block index b = 0 .. num_blocks-1
    #    For block b, use keys b*4+0..b*4+3 of prune_map
    # ------------------------------------------------------------
    for b in range(num_blocks):
        base = b * 4
        block = blocks[b]

        # --- (A) Prune QKV heads via prune_map[base + 0]
        heads_to_prune = prune_map[base + 0]
        attn_module = block.attn    # the Attention submodule

        orig_qkv = attn_module.qkv
        old_num_heads = attn_module.num_heads
        head_dim = attn_module.head_dim

        # Prune QKV → new_qkv, keep_heads_mask, new_num_heads
        new_qkv, keep_heads_mask, new_num_heads = _prune_qkv(
            orig_qkv,
            heads_to_prune,
            old_num_heads,
            head_dim
        )

        # Replace block.attn.qkv in-place
        parent_qkv, attr_qkv = _find_parent_and_attr(model, orig_qkv)
        setattr(parent_qkv, attr_qkv, new_qkv)
        attn_module.qkv = new_qkv

        # Update the Attention module’s head count, embed_dim, and scale
        attn_module.num_heads = new_num_heads
        new_dim = new_num_heads * head_dim
        attn_module.embed_dim = new_dim
        attn_module.scale = 1.0 / (head_dim ** 0.5)

        # --- (B) Rebuild block.norm1 (LayerNorm) to new_dim (applied before attention)
        old_norm1 = block.norm1
        parent_n1, attr_n1 = _find_parent_and_attr(model, old_norm1)
        new_norm1 = nn.LayerNorm(new_dim).to(old_norm1.weight.device)
        setattr(parent_n1, attr_n1, new_norm1)
        block.norm1 = new_norm1

        # --- (C) Rebuild block.ls1 (LayerScale) to new_dim (applied right after attention)
        # If `init_values` was provided at block creation, they set ls1 = LayerScale(dim, init_values).
        # Here we check if the old ls1 was a LayerScale or Identity.
        old_ls1 = block.ls1
        parent_ls1, attr_ls1 = _find_parent_and_attr(model, old_ls1)
        if isinstance(old_ls1, nn.Identity):
            new_ls1 = nn.Identity()
        else:
            # Capture init_values from old_ls1 if it was a LayerScale:
            init_value = getattr(old_ls1, 'init_values', None)
            new_ls1 = LayerScale(new_dim, init_values=init_value).to(old_ls1.gamma.device)
        setattr(parent_ls1, attr_ls1, new_ls1)
        block.ls1 = new_ls1

        # --- (D) Prune inputs of block.attn.proj using keep_heads_mask
        orig_proj = attn_module.proj  # nn.Linear(old_dim → old_dim)
        old_proj_dim = old_num_heads * head_dim  # = orig_dim

        mask_in_proj = keep_heads_mask.repeat_interleave(head_dim)  # length = old_proj_dim
        pruned_proj_in = _prune_linear_inputs(orig_proj, mask_in_proj)

        parent_proj, attr_proj = _find_parent_and_attr(model, orig_proj)
        setattr(parent_proj, attr_proj, pruned_proj_in)
        attn_module.proj = pruned_proj_in

        # After pruning inputs, proj’s in_features = new_dim, out_features = old_proj_dim
        # But we still need to prune proj’s outputs next.

        # --- (E) Prune outputs of block.attn.proj via prune_map[base + 1]
        proj_prune_idxs = prune_map[base + 1]
        pruned_proj_out, keep_proj_out = _prune_linear_outputs(pruned_proj_in, proj_prune_idxs)

        parent_proj_out, attr_proj_out = _find_parent_and_attr(model, pruned_proj_in)
        setattr(parent_proj_out, attr_proj_out, pruned_proj_out)
        attn_module.proj = pruned_proj_out

        # Now block’s “post-attention” embedding dimension = final_proj_out_dim
        final_proj_out_dim = pruned_proj_out.out_features

        # --- (F) Rebuild block.norm2 (LayerNorm) to final_proj_out_dim (applied before MLP)
        old_norm2 = block.norm2
        parent_n2, attr_n2 = _find_parent_and_attr(model, old_norm2)
        new_norm2 = nn.LayerNorm(final_proj_out_dim).to(old_norm2.weight.device)
        setattr(parent_n2, attr_n2, new_norm2)
        block.norm2 = new_norm2

        # --- (G) Rebuild block.ls2 (LayerScale) to final_proj_out_dim (applied after MLP)
        old_ls2 = block.ls2
        parent_ls2, attr_ls2 = _find_parent_and_attr(model, old_ls2)
        if isinstance(old_ls2, nn.Identity):
            new_ls2 = nn.Identity()
        else:
            init_value2 = getattr(old_ls2, 'init_values', None)
            new_ls2 = LayerScale(final_proj_out_dim, init_values=init_value2).to(old_ls2.gamma.device)
        setattr(parent_ls2, attr_ls2, new_ls2)
        block.ls2 = new_ls2

        # --- (H) Prune inputs of block.mlp.fc1 using keep_proj_out mask
        orig_fc1 = block.mlp.fc1  # nn.Linear(old_proj_out_dim → hidden_dim)
        pruned_fc1_in = _prune_linear_inputs(orig_fc1, keep_proj_out)

        parent_fc1, attr_fc1 = _find_parent_and_attr(model, orig_fc1)
        setattr(parent_fc1, attr_fc1, pruned_fc1_in)
        block.mlp.fc1 = pruned_fc1_in

        # --- (I) Prune outputs of block.mlp.fc1 via prune_map[base + 2]
        fc1_prune_idxs = prune_map[base + 2]
        pruned_fc1_out, keep_fc1_out = _prune_linear_outputs(pruned_fc1_in, fc1_prune_idxs)

        parent_fc1_out, attr_fc1_out = _find_parent_and_attr(model, pruned_fc1_in)
        setattr(parent_fc1_out, attr_fc1_out, pruned_fc1_out)
        block.mlp.fc1 = pruned_fc1_out

        final_fc1_out_dim = pruned_fc1_out.out_features

        # --- (J) Prune inputs of block.mlp.fc2 using keep_fc1_out mask
        orig_fc2 = block.mlp.fc2  # nn.Linear(old_hidden_dim → old_proj_out_dim)
        pruned_fc2_in = _prune_linear_inputs(orig_fc2, keep_fc1_out)

        parent_fc2, attr_fc2 = _find_parent_and_attr(model, orig_fc2)
        setattr(parent_fc2, attr_fc2, pruned_fc2_in)
        block.mlp.fc2 = pruned_fc2_in

        # --- (K) Prune outputs of block.mlp.fc2 via prune_map[base + 3]
        fc2_prune_idxs = prune_map[base + 3]
        pruned_fc2_out, keep_fc2_out = _prune_linear_outputs(pruned_fc2_in, fc2_prune_idxs)

        parent_fc2_out, attr_fc2_out = _find_parent_and_attr(model, pruned_fc2_in)
        setattr(parent_fc2_out, attr_fc2_out, pruned_fc2_out)
        block.mlp.fc2 = pruned_fc2_out

        final_block_out_dim = pruned_fc2_out.out_features

        # ------------------------------------------------------------
        # 3) If there is a next block, rebuild its norm1, qkv, and ls1 to match final_block_out_dim
        # ------------------------------------------------------------
        if b + 1 < num_blocks:
            next_block = blocks[b + 1]

            # (A) Rebuild next_block.norm1 to LayerNorm(final_block_out_dim)
            old_next_norm1 = next_block.norm1
            parent_n1_nxt, attr_n1_nxt = _find_parent_and_attr(model, old_next_norm1)
            new_next_norm1 = nn.LayerNorm(final_block_out_dim).to(old_next_norm1.weight.device)
            setattr(parent_n1_nxt, attr_n1_nxt, new_next_norm1)
            next_block.norm1 = new_next_norm1

            # (B) Rebuild next_block.ls1 (LayerScale) to final_block_out_dim
            old_next_ls1 = next_block.ls1
            parent_ls1_nxt, attr_ls1_nxt = _find_parent_and_attr(model, old_next_ls1)
            if isinstance(old_next_ls1, nn.Identity):
                new_next_ls1 = nn.Identity()
            else:
                init_val1 = getattr(old_next_ls1, 'init_values', None)
                new_next_ls1 = LayerScale(final_block_out_dim, init_values=init_val1).to(old_next_ls1.gamma.device)
            setattr(parent_ls1_nxt, attr_ls1_nxt, new_next_ls1)
            next_block.ls1 = new_next_ls1

            # (C) Rebuild next_block.attn.qkv to take input_dim = final_block_out_dim
            orig_next_qkv = next_block.attn.qkv
            old_next_num_heads = next_block.attn.num_heads
            head_dim_next = next_block.attn.head_dim
            orig_bias = orig_next_qkv.bias is not None
            old_next_dim = old_next_num_heads * head_dim_next

            if final_block_out_dim > old_next_dim:
                raise RuntimeError(
                    f"Next block expects input={old_next_dim}, but previous block output={final_block_out_dim}."
                )

            # Extract weight & bias
            Wn = orig_next_qkv.weight.data   # [3*old_next_dim, old_next_dim]
            Bn = orig_next_qkv.bias.data if orig_bias else None

            new_weights = []
            new_biases = []

            for chunk_idx in range(3):
                rs = chunk_idx * old_next_dim
                re = (chunk_idx + 1) * old_next_dim
                W_chunk = Wn[rs:re, :]                             # [old_dim, old_dim]
                W_chunk = W_chunk[:final_block_out_dim, :final_block_out_dim]  # [new_dim, new_dim]
                new_weights.append(W_chunk)
                if orig_bias:
                    b_chunk = Bn[rs:re]                           # [old_dim]
                    b_chunk = b_chunk[:final_block_out_dim]       # [new_dim]
                    new_biases.append(b_chunk)

            Wn_new = torch.cat(new_weights, dim=0)               # [3*new_dim, new_dim]
            Bn_new = torch.cat(new_biases, dim=0) if orig_bias else None

            new_next_qkv = nn.Linear(
                in_features=final_block_out_dim,
                out_features=3 * final_block_out_dim,
                bias=orig_bias
            ).to(Wn_new.device)

            with torch.no_grad():
                new_next_qkv.weight.copy_(Wn_new)
                if Bn_new is not None:
                    new_next_qkv.bias.copy_(Bn_new)

            parent_qkv_nxt, attr_qkv_nxt = _find_parent_and_attr(model, orig_next_qkv)
            setattr(parent_qkv_nxt, attr_qkv_nxt, new_next_qkv)
            next_block.attn.qkv = new_next_qkv

            # (D) Rebuild next_block.attn.proj inputs (in_features = final_block_out_dim)
            old_next_proj = next_block.attn.proj
            col_mask = torch.zeros(old_next_dim, dtype=torch.bool, device=Wn_new.device)
            col_mask[:final_block_out_dim] = True  # keep first final_block_out_dim columns
            pruned_next_proj = _prune_linear_inputs(old_next_proj, col_mask)
            parent_proj_nxt, attr_proj_nxt = _find_parent_and_attr(model, old_next_proj)
            setattr(parent_proj_nxt, attr_proj_nxt, pruned_next_proj)
            next_block.attn.proj = pruned_next_proj

    # ------------------------------------------------------------
    # 4) After all blocks are pruned, update model.norm, model.fc_norm, and model.head
    # ------------------------------------------------------------
    final_dim = blocks[-1].mlp.fc2.out_features

    # 4.A) Rebuild model.norm (LayerNorm)
    old_norm = model.norm
    parent_norm, attr_norm = _find_parent_and_attr(model, old_norm)
    new_norm = nn.LayerNorm(final_dim).to(old_norm.weight.device)
    setattr(parent_norm, attr_norm, new_norm)
    model.norm = new_norm

    # 4.B) Rebuild model.fc_norm
    old_fc_norm = model.fc_norm
    parent_fc_norm, attr_fc_norm = _find_parent_and_attr(model, old_fc_norm)
    new_fc_norm = nn.LayerNorm(final_dim).to(old_fc_norm.weight.device)
    setattr(parent_fc_norm, attr_fc_norm, new_fc_norm)
    model.fc_norm = new_fc_norm

    # 4.C) model.head_drop remains unchanged (dropout is dimension‐agnostic)

    # 4.D) Rebuild model.head (Linear(final_dim → num_classes))
    old_head = model.head
    parent_head, attr_head = _find_parent_and_attr(model, old_head)
    num_classes = old_head.out_features
    new_head = nn.Linear(final_dim, num_classes).to(old_head.weight.device)
    setattr(parent_head, attr_head, new_head)
    model.head = new_head

    # Done. All blocks (and final head) are now pruned in-place.
