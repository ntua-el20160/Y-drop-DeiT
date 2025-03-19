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
    
    def find_knn_and_distance(interpolated: torch.Tensor, dataset: torch.Tensor, k: int):
        """
        For each image in the batch and each interpolation step, find the k nearest neighbors
        in the dataset and calculate the Euclidean distance (straight path distance) to them.
        
        Parameters:
            interpolated (torch.Tensor): Tensor of shape [B, n_steps+1, C, H, W] from split_images.
            dataset (torch.Tensor): Tensor of all images in the dataset of shape [N, C, H, W].
            k (int): Number of nearest neighbors to retrieve.
        
        Returns:
            knn_indices (torch.Tensor): Indices of the k nearest neighbors for each interpolation step.
                                        Shape: [B, n_steps+1, k]
            knn_distances (torch.Tensor): The corresponding Euclidean distances.
                                        Shape: [B, n_steps+1, k]
        """
        B, n_steps_plus_one, C, H, W = interpolated.shape
        N = dataset.shape[0]
        
        # Flatten the spatial dimensions (and channels) so each image becomes a vector.
        # For the interpolation images: [B, n_steps+1, C*H*W]
        interpolated_flat = interpolated.view(B, n_steps_plus_one, -1)
        # For the dataset: [N, C*H*W]
        dataset_flat = dataset.view(N, -1)
        
        # Compute pairwise distances between each interpolation image and every image in the dataset.
        # The result will have shape [B, n_steps+1, N]
        distances = torch.cdist(interpolated_flat, dataset_flat, p=2)
        
        # For each interpolation image, find the indices and distances of the k nearest neighbors.
        knn_distances, knn_indices = torch.topk(distances, k=k, largest=False)
        
        return knn_indices, knn_distances
        
    # def calculate_scores(self, batches: torch.Tensor,device: torch.device,n_batches:int =1,) -> torch.Tensor:
    #     """
    #     Compute conductance scores through the transformer blocks.
    #     x: input image batch.

    #     This function creates an interpolation path from a baseline to x,
    #     then sequentially passes it through each transformer block. Each block
    #     is expected to (a) save outputs and gradients via save_output_gradients, and
    #     (b) update its dropout masks using update_dropout_masks().
    #     """
    #     for batch in batches:
    #         x, _ = batch  # assuming batch is (samples, targets)
                
    #         x = x.to(device, non_blocking=True)
    #         #print(x.shape)
    #         interp = self.split_images(x, n_steps=self.n_steps)
    #         # Flatten the interpolation dimension into the batch dimension.
    #         B, steps, C, H, W = interp.shape
    #         interp_flat = interp.view(-1, C, H, W)

    #         # 3. Process through the initial embedding layers.
    #         x_emb = self.patch_embed(interp_flat)  # shape: [(n_steps+1)*B, N, D]
    #         x_emb = self._pos_embed(x_emb)
    #         x_emb = self.patch_drop(x_emb)
    #         x_emb = self.norm_pre(x_emb)

    #         # For each block, first run the forward pass to save outputs and gradients.
    #         for block in self.blocks:
    #             x_emb =block.save_output_gradients(x_emb, n_steps=self.n_steps)
            
    #         x_feat = self.norm(x_emb)
    #         x_pool = self.pool(x_feat)            # pool() remains as in the base class.
    #         x_fc = self.fc_norm(x_pool)
    #         x_out = self.head_drop(x_fc)
    #         out = self.head(x_out)
            
    #         # 6. Compute scalar loss and backpropagate to compute gradients.
    #         loss = out.sum()
    #         loss.backward()

    #         # Then, update dropout masks based on the saved data.
    #         for block in self.blocks:
    #             block.calculate_conductance(n_steps=self.n_steps,n_batches=n_batches)
        
    #     for block in self.blocks:
    #         block.update_dropout_masks()
        
    #     # Optionally, you can run a normal forward pass on x now.
    #     # Here, we return the original input (or you can run self.forward(x)).
    #     return out
    def calculate_scores(self, batches: torch.Tensor, device: torch.device, n_batches: int = 1) -> torch.Tensor:
        """
        Compute conductance scores through the transformer blocks.
        x: input image batch.

        This function creates an interpolation path from a baseline to x,
        then sequentially passes it through each transformer block. Each block
        is expected to (a) save outputs and gradients via save_output_gradients, and
        (b) update its dropout masks using update_dropout_masks().
        """
        for batch in batches:
            x, _ = batch  # assuming batch is (samples, targets)
            x = x.to(device, non_blocking=True)
            B = x.shape[0]
            
            # Process in mini-batches if B is greater than 32.
            if B > 32:
                
                for i in range(0, B, 32):
                    mini_x = x[i:i+32]
                    interp = self.split_images(mini_x, n_steps=self.n_steps)
                    mini_B, steps, C, H, W = interp.shape
                    interp_flat = interp.view(-1, C, H, W)

                    # Process through the initial embedding layers.
                    x_emb = self.patch_embed(interp_flat)  # shape: [(n_steps+1)*mini_B, N, D]
                    x_emb = self._pos_embed(x_emb)
                    x_emb = self.patch_drop(x_emb)
                    x_emb = self.norm_pre(x_emb)

                    # For each block, run the forward pass to save outputs and gradients.
                    for block in self.blocks:
                        x_emb = block.save_output_gradients(x_emb, n_steps=self.n_steps)

                    x_feat = self.norm(x_emb)
                    x_pool = self.pool(x_feat)
                    x_fc = self.fc_norm(x_pool)
                    x_out = self.head_drop(x_fc)
                    out = self.head(x_out)

                    # Compute scalar loss and backpropagate.
                    loss = out.sum()
                    loss.backward()

                    # Update conductance for each block.
                    for block in self.blocks:
                        block.calculate_conductance(n_steps=self.n_steps, n_batches=n_batches)
            else:
                # Process normally when B is less than or equal to 32.
                interp = self.split_images(x, n_steps=self.n_steps)
                B, steps, C, H, W = interp.shape
                interp_flat = interp.view(-1, C, H, W)
                
                x_emb = self.patch_embed(interp_flat)
                x_emb = self._pos_embed(x_emb)
                x_emb = self.patch_drop(x_emb)
                x_emb = self.norm_pre(x_emb)
                
                for block in self.blocks:
                    x_emb = block.save_output_gradients(x_emb, n_steps=self.n_steps)
                
                x_feat = self.norm(x_emb)
                x_pool = self.pool(x_feat)
                x_fc = self.fc_norm(x_pool)
                x_out = self.head_drop(x_fc)
                out = self.head(x_out)
                
                loss = out.sum()
                loss.backward()
                
                for block in self.blocks:
                    block.calculate_conductance(n_steps=self.n_steps, n_batches=n_batches)

    # After processing all batches, update the dropout masks.
        for block in self.blocks:
            block.update_dropout_masks()

        return out

    def base_dropout(self):
        for block in self.blocks:
            block.base_dropout()
        return
    def custom_dropout(self):
        for block in self.blocks:
            block.custom_dropout()
        return
    def update_hyperparameters(self,p_high=None, p_low=None,elasticity = None,mean_shift = None,p=None,layer= None,module =None):
        for block in self.blocks:
            block.update_hyperparameters(p_high,p_low,elasticity,mean_shift,p,layer,module)
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

