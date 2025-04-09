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
from updated_transformer.dynamic_dropout import MyDropout
from timm.layers import to_2tuple

from timm.models.vision_transformer import VisionTransformer, _cfg, LayerScale
from timm.models import register_model
from timm.layers import PatchEmbed,use_fused_attn,DropPath, trunc_normal_


"""DropPath and LayerScale may need changes"""


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks

    NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
            ydrop: bool = False,
            mask_type: Optional[str] = 'sigmoid',
            elasticity: Optional[float] = 0.01,
            scaler: Optional[float] = 1.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        print(f"[Mlp] ydrop: {ydrop}, drop: {drop}, mask_type: {mask_type}, "
              f"elasticity: {elasticity}, scaler: {scaler}")
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()

        if ydrop is True:
            self.drop1 = MyDropout(elasticity=elasticity, p=drop_probs[0], tied_layer=self.act, mask_type=mask_type, scaler=scaler)
        else:
            self.drop1 = nn.Dropout(drop_probs[0])
        
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        

        if ydrop is True:
            self.drop2 = MyDropout(elasticity=elasticity, p=drop_probs[1], tied_layer=self.fc2, mask_type=mask_type, scaler=scaler)
        else:
            self.drop2 = nn.Dropout(drop_probs[1])
            

    def forward(self, x):
        x = self.fc1(x)
        x_act = self.act(x)
        # Save the output after activation if hooks are set.
        x_drop1 = self.drop1(x_act)
        x_norm = self.norm(x_drop1)
        x_fc2 = self.fc2(x_norm)
        # Save the output after fc2.
        x_drop2 = self.drop2(x_fc2)
        return x_drop2
    
    def use_normal_dropout(self):
        self.drop1.use_normal_dropout()
        self.drop2.use_normal_dropout()
    def use_ydrop(self):
        self.drop1.use_ydrop()
        self.drop2.use_ydrop()