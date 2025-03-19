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
            proj_drop_custom: float = [0.1,0.3],
            attn_drop_custom: float = [0.1,0.3],
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            mlp_layer: Type[nn.Module] = Mlp,

    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_drop_custom=attn_drop_custom,
            proj_drop_custom=proj_drop_custom,
            norm_layer=norm_layer,
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
            drop_custom=proj_drop_custom,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
        
    def save_output_gradients(self, x: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        Run the block with dropout disabled to capture outputs and gradients
        for the layers of interest in Attention and Mlp.
        """
        # Pass through the attention branch:
        x_norm1 = self.norm1(x)
        attn_out = self.attn.save_output_gradients(x_norm1, n_steps)
        #attn_out = self.attn(x_norm1)
        attn_out = self.ls1(attn_out)
        attn_out = self.drop_path1(attn_out)
        x1 = x + attn_out

        # Pass through the MLP branch:
        x_norm2 = self.norm2(x1)
        mlp_out = self.mlp.save_output_gradients(x_norm2, n_steps)
        #mlp_out = self.mlp(x_norm2)
        mlp_out = self.ls2(mlp_out)
        mlp_out = self.drop_path2(mlp_out)
        x2 = x1 + mlp_out
        return x2

    def calculate_conductance(self,n_steps: int,n_batches: int = None) -> None:
        """
        Delegate dropout mask update to submodules based on computed conductance.
        """
        #print('next batch')
        self.attn.calculate_conductance(n_steps,n_batches)
        self.mlp.calculate_conductance(n_steps,n_batches)
    def update_dropout_masks(self):
        shifts_attn =self.attn.update_dropout_masks()
        shifts_mlp =self.mlp.update_dropout_masks()
        return shifts_attn+shifts_mlp
        
    def base_dropout(self):
        self.attn.base_dropout()
        self.mlp.base_dropout()
        return
    def custom_dropout(self):
        self.attn.custom_dropout()
        self.mlp.custom_dropout() 
        return
    
    def update_hyperparameters(self,p_high=None, p_low=None,elasticity = None,mean_shift = None,p=None,layer= None,module =None):
        if module == "attn":
            self.attn.update_hyperparameters(p_high,p_low,elasticity,mean_shift,p,layer)
        elif module =="mlp":
            self.mlp.update_hyperparameters(p_high,p_low,elasticity,mean_shift,p,layer)
        else:
            self.attn.update_hyperparameters(p_high,p_low,elasticity,mean_shift,p,layer)
            self.mlp.update_hyperparameters(p_high,p_low,elasticity,mean_shift,p,layer)
    
    

