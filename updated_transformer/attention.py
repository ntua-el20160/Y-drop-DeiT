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
            transformer_mean: bool = False,

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
        self.pruning_identity_layer = nn.Identity()
        print(self.qkv)
        print(f"[Attention] ydrop: {ydrop}, attn_drop: {attn_drop}, proj_drop: {proj_drop}, "
              f"mask_type: {mask_type}, elasticity: {elasticity}, scaler: {scaler}")

        
        if ydrop is True:
            self.attn_drop = MyDropout(elasticity=elasticity, p=attn_drop, tied_layer=self.attention_identity_layer, mask_type=mask_type, scaler=scaler,transformer_mean= False)
        else:
            self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        print(self.proj)
        if ydrop is True:
            self.proj_drop = MyDropout(elasticity=elasticity, p=proj_drop, tied_layer=self.proj, mask_type=mask_type, scaler=scaler,transformer_mean=transformer_mean)
        else:
            self.proj_drop = nn.Dropout(proj_drop)
            


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        print(q.shape, k.shape, v.shape)
        if self.fused_attn:
            x,_ = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn =self.attention_identity_layer(attn)
            print(attn.shape)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        print(x.shape)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.pruning_identity_layer(x)
        x = self.proj(x)
        #print(x.shape)

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