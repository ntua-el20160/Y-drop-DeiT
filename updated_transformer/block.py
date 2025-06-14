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

from timm.models.vision_transformer import VisionTransformer, _cfg, LayerScale
from timm.models import register_model
from timm.layers import PatchEmbed,use_fused_attn,DropPath, trunc_normal_

from updated_transformer.mlp import Mlp
from updated_transformer.attention import Attention


"""DropPath and LayerScale may need changes"""


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            mlp_layer: Type[nn.Module] = Mlp,
            ydrop: bool = False,
            mask_type: Optional[str] = 'sigmoid',
            elasticity: Optional[float] = 0.01,
            scaler: Optional[float] = 1.0,
            transformer_mean: bool = False,
            rescaling_type: Optional[str] = None,


    ) -> None:
        super().__init__()
        print(f"[Block] ydrop: {ydrop}, attn_drop: {attn_drop}, proj_drop: {proj_drop}, "
              f"mask_type: {mask_type}, elasticity: {elasticity}, scaler: {scaler}")
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            ydrop=ydrop,
            mask_type=mask_type,
            elasticity=elasticity,
            scaler=scaler,
            transformer_mean=transformer_mean,
            rescaling_type=rescaling_type,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            bias=proj_bias,
            drop=proj_drop,
            ydrop=ydrop,
            mask_type=mask_type,
            elasticity=elasticity,
            scaler=scaler,
            transformer_mean=transformer_mean,
            rescaling_type=rescaling_type,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.selected_layers = [self.attn.attention_identity_layer,self.attn.proj,self.mlp.fc1,self.mlp.fc2]
        self.drop_list = [self.attn.attn_drop,self.attn.proj_drop,self.mlp.drop1,self.mlp.drop2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print("hi")
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
        

    def use_normal_dropout(self):
        self.attn.use_normal_dropout()
        self.mlp.use_normal_dropout()
        return
    def use_ydrop(self):
        self.attn.use_ydrop()
        self.mlp.use_ydrop() 
        return
    
    

