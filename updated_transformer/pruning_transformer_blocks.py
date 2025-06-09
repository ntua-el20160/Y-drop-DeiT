import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from timm.models.vision_transformer import LayerScale

def prune_vit_blocks(
    model: nn.Module,
    prune_map: Dict[int, List[int]]
) -> None:
    """
    In-place pruning of a VisionTransformer-like model:
      • prune_map[b*4 + 0]: dimension indices to drop from block.attn.qkv (per-head dims)
      • prune_map[b*4 + 1]: row indices to drop from block.attn.proj
      • prune_map[b*4 + 2]: row indices to drop from block.mlp.fc1
      • prune_map[b*4 + 3]: row indices to drop from block.mlp.fc2

    Assumes each block's qkv embed_dim remains divisible by num_heads after pruning.
    Raises ValueError on out-of-bounds or incompatible dims.
    """
    model.eval()

    # ----- Helpers -----------------------------------------------------
    def _find_parent_and_name(root: nn.Module, child: nn.Module) -> Tuple[nn.Module, str]:
        for parent in root.modules():
            for name, mod in parent.named_children():
                if mod is child:
                    return parent, name
        raise ValueError(f"Module {child} not found in model")

    def _prune_linear_outputs(orig: nn.Linear, prune_idxs: List[int]) -> Tuple[nn.Linear, torch.Tensor]:
        out_f, in_f = orig.out_features, orig.in_features
        if any(i < 0 or i >= out_f for i in prune_idxs):
            raise ValueError(f"Output prune idx out of bounds: {prune_idxs} vs out_features={out_f}")
        device = orig.weight.device
        mask = torch.ones(out_f, dtype=torch.bool, device=device)
        mask[prune_idxs] = False
        new_out = mask.sum().item()
        new_lin = nn.Linear(in_f, new_out, bias=orig.bias is not None).to(device)
        with torch.no_grad():
            new_lin.weight.copy_(orig.weight[mask])
            if orig.bias is not None:
                new_lin.bias.copy_(orig.bias[mask])
        return new_lin, mask

    def _prune_linear_inputs(orig: nn.Linear, keep_mask: torch.Tensor) -> nn.Linear:
        in_f = orig.in_features
        if keep_mask.numel() != in_f:
            raise ValueError(f"Input mask length {keep_mask.numel()} != in_features {in_f}")
        idx = keep_mask.nonzero(as_tuple=False).squeeze(1)
        new_in = idx.numel()
        new_lin = nn.Linear(new_in, orig.out_features, bias=orig.bias is not None).to(orig.weight.device)
        with torch.no_grad():
            new_lin.weight.copy_(orig.weight[:, idx])
            if orig.bias is not None:
                new_lin.bias.copy_(orig.bias)
        return new_lin

    def _prune_qkv_dims(orig: nn.Linear, drop_dims: List[int], num_heads: int) -> Tuple[nn.Linear, torch.Tensor]:
        """
        Prune specific embedding-dimension indices from QKV projection.
        drop_dims: indices in [0 .. embed_dim) to remove from each head's embedding dims.
        """
        out3, in_f = orig.out_features, orig.in_features
        # out3 = 3 * embed_dim
        if out3 % 3 != 0 or out3 // 3 != in_f:
            raise ValueError("Unexpected qkv dimensions")
        embed_dim = in_f
        device = orig.weight.device
        # validate
        if any(d < 0 or d >= embed_dim for d in drop_dims):
            raise ValueError(f"Drop dim out of bounds: {drop_dims} vs embed_dim={embed_dim}")
        # mask dims to keep
        keep = torch.ones(embed_dim, dtype=torch.bool, device=device)
        keep[drop_dims] = False
        new_embed = keep.sum().item()
        if new_embed % num_heads != 0:
            raise ValueError(f"Pruned embed_dim={new_embed} not divisible by num_heads={num_heads}")
        # gather weights
        W = orig.weight.data  # [3*embed_dim, embed_dim]
        B = orig.bias.data if orig.bias is not None else None
        new_blocks = []
        new_bias = []
        for i in range(3):
            start, end = i*embed_dim, (i+1)*embed_dim
            block = W[start:end]
            pruned = block[keep][:, keep]
            new_blocks.append(pruned)
            if B is not None:
                new_bias.append(B[start:end][keep])
        Wn = torch.cat(new_blocks, dim=0)
        Bn = torch.cat(new_bias, dim=0) if B is not None else None
        # build new linear
        new_qkv = nn.Linear(new_embed, 3*new_embed, bias=Bn is not None).to(device)
        with torch.no_grad():
            new_qkv.weight.copy_(Wn)
            if Bn is not None:
                new_qkv.bias.copy_(Bn)
        return new_qkv, keep

    # ----- Main pruning ------------------------------------------------
    blocks = list(model.blocks)
    n_blocks = len(blocks)
    missing = [i for i in range(4*n_blocks) if i not in prune_map]
    if missing:
        raise ValueError(f"prune_map missing keys: {missing}")

    for b, block in enumerate(blocks):
        base = 4 * b
        attn = block.attn
        # (1) prune QKV dimensions
        drop_dims = prune_map[base]
        new_qkv, dim_keep = _prune_qkv_dims(attn.qkv, drop_dims, attn.head_dim)
        p, nm = _find_parent_and_name(model, attn.qkv)
        setattr(p, nm, new_qkv)
        attn.qkv = new_qkv
        # update dims
        new_embed = dim_keep.sum().item()
        attn.head_dim = new_embed // attn.num_heads
        attn.embed_dim = new_embed
        attn.scale = attn.head_dim ** -0.5

        # rebuild norm1 & ls1 to new_embed
        for old, cls in [(block.norm1, nn.LayerNorm), (block.ls1, LayerScale)]:
            p, nm = _find_parent_and_name(model, old)
            if isinstance(old, nn.Identity):
                new_mod = nn.Identity()
            else:
                cfg = {} if cls is nn.LayerNorm else {"init_values": getattr(old, 'init_values', None)}
                new_mod = cls(new_embed, **cfg).to(old.weight.device)
            setattr(p, nm, new_mod)
            setattr(block, nm, new_mod)

        # (2) prune proj inputs then outputs
        proj = attn.proj
        pr_in = _prune_linear_inputs(proj, dim_keep)
        p, nm = _find_parent_and_name(model, proj)
        setattr(p, nm, pr_in); attn.proj = pr_in
        pr_out, out_keep = _prune_linear_outputs(pr_in, prune_map[base+1])
        p, nm = _find_parent_and_name(model, pr_in)
        setattr(p, nm, pr_out); attn.proj = pr_out
        post_dim = pr_out.out_features

        # rebuild norm2 & ls2
        for old, cls in [(block.norm2, nn.LayerNorm), (block.ls2, LayerScale)]:
            p, nm = _find_parent_and_name(model, old)
            if isinstance(old, nn.Identity):
                new_mod = nn.Identity()
            else:
                cfg = {} if cls is nn.LayerNorm else {"init_values": getattr(old, 'init_values', None)}
                new_mod = cls(post_dim, **cfg).to(old.weight.device)
            setattr(p, nm, new_mod)
            setattr(block, nm, new_mod)

        # (3) prune MLP
        fc1 = block.mlp.fc1
        in1 = _prune_linear_inputs(fc1, out_keep)
        p, nm = _find_parent_and_name(model, fc1)
        setattr(p, nm, in1); block.mlp.fc1 = in1
        out1, keep1 = _prune_linear_outputs(in1, prune_map[base+2])
        p, nm = _find_parent_and_name(model, in1)
        setattr(p, nm, out1); block.mlp.fc1 = out1

        fc2 = block.mlp.fc2
        in2 = _prune_linear_inputs(fc2, keep1)
        p, nm = _find_parent_and_name(model, fc2)
        setattr(p, nm, in2); block.mlp.fc2 = in2
        out2, _ = _prune_linear_outputs(in2, prune_map[base+3])
        p, nm = _find_parent_and_name(model, in2)
        setattr(p, nm, out2); block.mlp.fc2 = out2
        block_out = out2.out_features

        # (4) adjust next block if exists
        if b + 1 < n_blocks:
            nxt = blocks[b+1]
            p, nm = _find_parent_and_name(model, nxt.norm1)
            new_norm1 = nn.LayerNorm(block_out).to(nxt.norm1.weight.device)
            setattr(p, nm, new_norm1); nxt.norm1 = new_norm1
            # adjust next qkv input dims
            oq = nxt.attn.qkv
            mask2 = dim_keep
            q_in = _prune_linear_inputs(oq, mask2)
            p, nm = _find_parent_and_name(model, oq)
            setattr(p, nm, q_in); nxt.attn.qkv = q_in

    # rebuild final norms and head
    final_dim = blocks[-1].mlp.fc2.out_features
    for attr in ['norm','fc_norm']:
        if hasattr(model, attr):
            m = getattr(model, attr)
            p, nm = _find_parent_and_name(model, m)
            new_m = nn.LayerNorm(final_dim).to(m.weight.device)
            setattr(p, nm, new_m); setattr(model, attr, new_m)
    head = model.head
    p, nm = _find_parent_and_name(model, head)
    new_head = nn.Linear(final_dim, head.out_features).to(head.weight.device)
    setattr(p, nm, new_head); model.head = new_head
