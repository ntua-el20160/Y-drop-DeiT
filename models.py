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

from updated_transformer.block import Block
from updated_transformer.mlp import Mlp
"""DropPath and LayerScale may need changes"""


from timm.models.vision_transformer import VisionTransformer
import torch
import torch.nn as nn

class MyVisionTransformer(VisionTransformer):
    """
    A subclass of timm's VisionTransformer that uses custom transformer blocks
    (with integrated conductance computation) for updating dropout masks.
    """
    def __init__(self, *args, n_steps: int = 5, **kwargs):
        super(MyVisionTransformer, self).__init__(*args, **kwargs)
        self.n_steps = n_steps  # number of interpolation steps for conductance
        # Ensure that self.blocks is built using your custom block_fn that implements
        # update_dropout_masks, e.g., MyTransformerBlock.
        
    def split_images(self, x: torch.Tensor, n_steps: int = None) -> torch.Tensor:
        """
        Create interpolation paths from a baseline (zeros) to x.
        Returns a tensor of shape [B, n_steps+1, C, H, W] where B is the batch size.
        """
        if n_steps is None:
            n_steps = self.n_steps
        baseline = torch.zeros_like(x)  # shape: [B, C, H, W]
        alphas = torch.linspace(0, 1, steps=n_steps + 1, device=x.device).view(1, n_steps + 1, 1, 1, 1)
        x_exp = x.unsqueeze(1)          # shape: [B, 1, C, H, W]
        baseline_exp = baseline.unsqueeze(1)  # shape: [B, 1, C, H, W]
        interpolated = baseline_exp + alphas * (x_exp - baseline_exp)
        return interpolated  # shape: [B, n_steps+1, C, H, W]
    
    def calcualate_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute conductance scores through the transformer blocks.
        x: input image batch.

        This function creates an interpolation path from a baseline to x,
        then sequentially passes it through each transformer block. Each block
        is expected to (a) save outputs and gradients via save_output_gradients, and
        (b) update its dropout masks using update_dropout_masks().
        """
        # Create interpolation paths from baseline to input.
        interp = self.split_images(x, n_steps=self.n_steps)
        # Flatten the interpolation dimension into the batch dimension.
        B, steps, C, H, W = interp.shape
        interp_flat = interp.view(-1, C, H, W)

        # 3. Process through the initial embedding layers.
        x_emb = self.patch_embed(interp_flat)  # shape: [(n_steps+1)*B, N, D]
        x_emb = self._pos_embed(x_emb)
        x_emb = self.patch_drop(x_emb)
        x_emb = self.norm_pre(x_emb)

        # For each block, first run the forward pass to save outputs and gradients.
        for block in self.blocks:
            x_emb =block.save_output_gradients(x_emb, n_steps=self.n_steps)
        
        x_feat = self.norm(x_emb)
        x_pool = self.pool(x_feat)            # pool() remains as in the base class.
        x_fc = self.fc_norm(x_pool)
        x_out = self.head_drop(x_fc)
        out = self.head(x_out)
        
        # 6. Compute scalar loss and backpropagate to compute gradients.
        loss = out.sum()
        loss.backward()

        # Then, update dropout masks based on the saved data.
        for block in self.blocks:
            block.update_dropout_masks(n_steps=self.n_steps)
        
        # Optionally, you can run a normal forward pass on x now.
        # Here, we return the original input (or you can run self.forward(x)).
        return out
    def base_dropout(self):
        for block in self.blocks:
            block.base_dropout()
        return
    def custom_dropout(self):
        for block in self.blocks:
            block.custom_dropout()
        return




@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    # model = VisionTransformer(
    #     patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
    #     norm_layer=partial(nn.LayerNorm, eps=1e-6),mlp_layer=Mlp, block_fn=Block, **kwargs)
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('pretrained_cfg_overlay', None)
    kwargs.pop('cache_dir', None)
    model = MyVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),mlp_layer=Mlp, block_fn=Block, n_steps =5, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    # model = VisionTransformer(
    #     patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
    #     norm_layer=partial(nn.LayerNorm, eps=1e-6),mlp_layer=Mlp, block_fn=Block, **kwargs)
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('pretrained_cfg_overlay', None)
    kwargs.pop('cache_dir', None)
    model = MyVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),mlp_layer=Mlp, block_fn=Block, n_steps =5, **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    # model = VisionTransformer(
    #     patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
    #     norm_layer=partial(nn.LayerNorm, eps=1e-6), mlp_layer=Mlp, block_fn=Block, **kwargs)
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('pretrained_cfg_overlay', None)
    kwargs.pop('cache_dir', None)
    model = MyVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), mlp_layer=Mlp, block_fn=Block, n_steps =5, **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

