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
from updated_transformer.custom_dropout import MyDropout
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
            drop_custom = None,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        
        if drop_custom is not None:
            self.drop1 = MyDropout(p_high=drop_custom[1], p_low=drop_custom[0])
        else:
            self.drop1 = nn.Dropout(drop_probs[0])

        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])

        if drop_custom is not None:
            self.drop2 = MyDropout(p_high=drop_custom[1], p_low=drop_custom[0])
        else:
            self.drop2 = nn.Dropout(drop_probs[1])
            
        self._saved = {}   # to hold outputs for fc1_act and fc2.
        self._hooks = []

    def forward(self, x):
        x = self.fc1(x)
        x_act = self.act(x)
        # Save the output after activation if hooks are set.
        if 'act' in self._saved:
            self._saved['act'] = x_act
        x_drop1 = self.drop1(x_act)
        x_norm = self.norm(x_drop1)
        x_fc2 = self.fc2(x_norm)
        # Save the output after fc2.
        if 'fc2' in self._saved:
            self._saved['fc2'] = x_fc2
        x_drop2 = self.drop2(x_fc2)
        return x_drop2
    def save_output_gradients(self, x: torch.Tensor, n_steps: int):
        """
        Run forward pass with dropout disabled to capture outputs at:
          - after activation (fc1+act) and
          - after fc2.
        """
        self._saved = {'act': None, 'fc2': None}
        orig_drop1, orig_drop2 = self.drop1, self.drop2
        self.drop1 = nn.Identity()
        self.drop2 = nn.Identity()
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    output.retain_grad()
                self._saved[name] = output
            return hook
        h1 = self.register_forward_hook(hook_fn('act'))  # register on self (after fc1+act) via forward override
        h2 = self.fc2.register_forward_hook(hook_fn('fc2'))
        self._hooks = [h1, h2]
        y= self.forward(x)
        self.drop1 = orig_drop1
        self.drop2 = orig_drop2
        self._n_steps = n_steps
        return y
    
    def update_dropout_masks(self,n_steps: int):
        """
        Compute integrated gradients for the saved outputs and update the dropout masks.
        """
        conductance = {}
        for key in ['act', 'fc2']:
            act = self._saved.get(key)
            if act is None or act.grad is None:
                continue
            B = act.shape[0] // (n_steps + 1)
            new_shape = (n_steps + 1, B) + act.shape[1:]
            acts = act.view(new_shape)
            grads = act.grad.view(new_shape)
            diffs = acts[1:] - acts[:-1]
            grad_seg = grads[:-1]
            integrated = (diffs * grad_seg).sum(dim=0) / n_steps  # shape: [B, ...]
            avg_conductance = integrated.mean(dim=0)                    # shape: same as one sample output
            conductance[key] = avg_conductance
        if 'act' in conductance:
            self.drop1.update_dropout_masks(conductance['act'])
        if 'fc2' in conductance:
            self.drop2.update_dropout_masks(conductance['fc2'])
        self._saved.clear()
        for h in self._hooks:
            h.remove()
        self._hooks = []
        
    def base_dropout(self):
        self.drop1.base_dropout()
        self.drop2.base_dropout()
    def custom_dropout(self):
        self.drop1.custom_dropout()
        self.drop2.custom_dropout()
