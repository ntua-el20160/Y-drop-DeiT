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
import math

from timm.models.vision_transformer import VisionTransformer, _cfg, LayerScale
from timm.models import register_model
from timm.layers import PatchEmbed,use_fused_attn,DropPath, trunc_normal_
from updated_transformer.custom_dropout import MyDropout

"""DropPath and LayerScale may need changes"""
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False,train =True) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=train)
    return attn_weight @ value,attn_weight


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            attn_drop_custom: float = None,
            proj_drop: float = 0.,
            proj_drop_custom: float = None,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        #self.fused_attn = use_fused_attn()
        self.fused_attn = False

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        
        if attn_drop_custom is not None:
            self.attn_drop = MyDropout(p_high=attn_drop_custom[1], p_low=attn_drop_custom[0])
        else:
            self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        if proj_drop_custom is not None:
            self.proj_drop = MyDropout(p_high=proj_drop_custom[1], p_low=proj_drop_custom[0])
        else:
            self.proj_drop = nn.Dropout(proj_drop)

        self._saved = {}  # e.g. keys: 'attn', 'proj'
        self._hooks = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x,_ = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            if 'attn' in self._saved:
                self._saved['attn'] = attn  # or accumulate if needed
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        if 'proj' in self._saved:
            self._saved['proj'] = x
        x = self.proj_drop(x)
        return x
    def save_output_gradients(self, x: torch.Tensor, n_steps: int):
        """
        Run forward pass with dropout disabled so that we capture the outputs
        at the layers of interest for each interpolation step.
        """
        # Prepare to save outputs:
        self._saved = {'attn': None, 'proj': None}
        # Temporarily disable dropout in this module.
        orig_attn_drop, orig_proj_drop = self.attn_drop, self.proj_drop
        self.attn_drop = nn.Identity()
        self.proj_drop = nn.Identity()
        
        # Register forward hooks to retain gradients.
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    output.retain_grad()
                self._saved[name] = output
            return hook
        h1 = self.register_forward_hook(hook_fn('attn'))
        h2 = self.proj.register_forward_hook(hook_fn('proj'))
        self._hooks = [h1, h2]
        
        # Run forward pass on x (which is your interpolation batch, shape [(n_steps+1)*B, ...])
        y = self.forward(x)
        
        # Restore dropout modules.
        self.attn_drop = orig_attn_drop
        self.proj_drop = orig_proj_drop
        # Return the forward output if needed.
        return y
    
    def update_dropout_masks(self, n_steps: int):
        """
        Compute conductance from the saved outputs and gradients, update
        the dropout masks in self.attn_drop and self.proj_drop, then clear saved data.
        """
        # Here, we assume that self._saved['attn'] and self._saved['proj']
        # have shape [(n_steps+1)*B, ...]. We need to reshape, compute differences,
        # and then integrate gradients.
        # For brevity, we illustrate the computation for one saved tensor.
        conductance = {}
        for key in ['attn', 'proj']:
            act = self._saved.get(key)
            if act is None or act.grad is None:
                continue
            B = act.shape[0] // (n_steps + 1)  # assume self._n_steps was stored
            new_shape = (n_steps + 1, B) + act.shape[1:]
            acts = act.view(new_shape)
            grads = act.grad.view(new_shape)
            diffs = acts[1:] - acts[:-1]
            grad_seg = grads[:-1]
            integrated = (diffs * grad_seg).sum(dim=0) /n_steps  # shape: [B, ...]
            avg_conductance = integrated.mean(dim=0)                    # shape: same as one sample output
            conductance[key] = avg_conductance
        # Now, use these conductance scores to update dropout masks.
        if 'attn' in conductance:
            self.attn_drop.update_dropout_masks(conductance['attn'])
        if 'proj' in conductance:
            self.proj_drop.update_dropout_masks(conductance['proj'])
        # Clear saved data and remove hooks.
        self._saved.clear()
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def base_dropout(self):
        self.proj_drop.base_dropout()
        self.attn_drop.base_dropout()
        return
    def custom_dropout(self):
        self.proj_drop.custom_dropout()
        self.attn_drop.custom_dropout()
        return