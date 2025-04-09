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
from updated_transformer.dynamic_dropout import MyDropout

"""DropPath and LayerScale may need changes"""
# def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
#         is_causal=False, scale=None, enable_gqa=False,train =True) -> torch.Tensor:
#     L, S = query.size(-2), key.size(-2)
#     scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
#     attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
#     if is_causal:
#         assert attn_mask is None
#         temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
#         attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
#         attn_bias.to(query.dtype)

#     if attn_mask is not None:
#         if attn_mask.dtype == torch.bool:
#             attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
#         else:
#             attn_bias = attn_mask + attn_bias

#     if enable_gqa:
#         key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
#         value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

#     attn_weight = query @ key.transpose(-2, -1) * scale_factor
#     attn_weight += attn_bias
#     attn_weight = torch.softmax(attn_weight, dim=-1)
#     attn_weight = torch.dropout(attn_weight, dropout_p, train=train)
#     return attn_weight @ value,attn_weight


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
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            ydrop: bool = False,
            mask_type: Optional[str] = 'sigmoid',
            elasticity: Optional[float] = 0.01,
            scaler: Optional[float] = 1.0,
    ) -> None:
        super().__init__()

        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = False

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attention_identity_layer = nn.Identity()
        print(f"[Attention] ydrop: {ydrop}, attn_drop: {attn_drop}, proj_drop: {proj_drop}, "
              f"mask_type: {mask_type}, elasticity: {elasticity}, scaler: {scaler}")

        
        if ydrop is True:
            self.attn_drop = MyDropout(elasticity=elasticity, p=attn_drop, tied_layer=self.attention_identity_layer, mask_type=mask_type, scaler=scaler)
        else:
            self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        if ydrop is True:
            self.proj_drop = MyDropout(elasticity=elasticity, p=proj_drop, tied_layer=self.proj, mask_type=mask_type, scaler=scaler)
        else:
            self.proj_drop = nn.Dropout(proj_drop)


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
            attn =self.attention_identity_layer(attn)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        x = self.proj_drop(x)
        return x
  
    def use_normal_dropout(self):
        if self.ydrop:
            self.proj_drop.use_normal_dropout()
            self.attn_drop.use_normal_dropout()
        return
    def use_ydrop(self):
        if self.ydrop:
            self.proj_drop.use_ydrop()
            self.attn_drop.use_ydrop()
        return