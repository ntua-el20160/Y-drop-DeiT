# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from typing import Type, Optional

from timm.models.vision_transformer import VisionTransformer, _cfg, LayerScale
from timm.models import register_model
from timm.layers import PatchEmbed, use_fused_attn, DropPath, trunc_normal_
from updated_transformer.dynamic_dropout import MyDropout


class SelfAttentionModule(nn.Module):
    """
    Full multi-head self-attention (raw scores, softmax, dropout, out-v).
    Returns values of shape [B, heads, N, head_dim].
    Stores raw attn scores in self.attn_scores.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Project input to q,k,v
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # Optional per-head normalization
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        # Dropout on attention weights
        self.attn_drop = nn.Dropout(attn_drop)
        # Hold raw scores
        self.attn_scores: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # 1) Linear -> [B,N,3,heads,head_dim] -> permute -> [3,B,heads,N,head_dim]
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # each [B,heads,N,head_dim]
        # 2) optional norm + scale
        q = self.q_norm(q) * self.scale
        k = self.k_norm(k)
        # 3) raw attention scores
        scores = q @ k.transpose(-2, -1)           # [B,heads,N,N]
        attn = scores.softmax(dim=-1)
        # store before dropout
        self.attn_scores = attn.detach()
        # 4) dropout
        attn = self.attn_drop(attn)
        # 5) weighted sum -> values
        out = attn @ v                             # [B,heads,N,head_dim]
        return out


class Attention(nn.Module):
    """
    Wrapper that projects, runs SelfAttentionModule, merges heads, and applies final proj+drop.
    Supports Y-Drop on both attn weights and output projection.
    """
    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        ydrop: bool = False,
        mask_type: Optional[str] = 'sigmoid',
        elasticity: Optional[float] = 0.01,
        scaler: Optional[float] = 1.0,
        smooth_scoring: Optional[bool] = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = False
        self.ydrop = ydrop

        # Core attention
        self.attn_module = SelfAttentionModule(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            norm_layer=norm_layer,
            attn_drop=0.0,  # drop via MyDropout if ydrop
        )
        # Replace drop on attn weights
        if ydrop:
            self.attn_module.attn_drop = MyDropout(
                elasticity=elasticity,
                p=attn_drop,
                tied_layer=self.attn_module,
                mask_type=mask_type,
                scaler=scaler,
                smoothed=smooth_scoring,
            )
        else:
            self.attn_module.attn_drop = nn.Dropout(attn_drop)

        # Projection
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        if ydrop:
            self.proj_drop = MyDropout(
                elasticity=elasticity,
                p=proj_drop,
                tied_layer=self.proj,
                mask_type=mask_type,
                scaler=scaler,
                smoothed=smooth_scoring,
            )
        else:
            self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        B, N, C = x.shape
        if self.fused_attn:
            # fallback to fused
            qkv = (
                self.attn_module.qkv(x)
                .reshape(B, N, 3, self.num_heads, self.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv.unbind(0)
            out, _ = F.scaled_dot_product_attention(
                q * self.scale, k, v,
                dropout_p=(self.attn_module.attn_drop.p if self.training else 0.0)
            )
            out = out  # [B, heads, N, head_dim]
        else:
            out = self.attn_module(x)  # [B, heads, N, head_dim]
        # merge heads
        x = out.transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        # final proj + drop
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def use_normal_dropout(self):
        if self.ydrop:
            self.attn_module.attn_drop.use_normal_dropout()
            self.proj_drop.use_normal_dropout()

    def use_ydrop(self):
        if self.ydrop:
            self.attn_module.attn_drop.use_ydrop()
            self.proj_drop.use_ydrop()
