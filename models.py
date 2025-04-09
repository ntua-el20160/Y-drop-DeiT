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
import copy
from typing import Iterable
from captum.attr import LayerConductance
from evaluate_gradients.MultiLayerConductance import MultiLayerConductance   

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
        self.drop_list = []
        self.selected_layers = []
        self.scores = {}
        for block in self.blocks:
            for module in block.drop_list:
                self.drop_list.append(module)
            for layer in block.selected_layers:
                self.selected_layers.append(layer)

    def use_normal_dropout(self):
        for drop in self.drop_list:
            drop.use_normal_dropout()

    def use_ydrop(self):
        for drop in self.drop_list:
            drop.use_ydrop()

    def plot_aggregated_statistics(self, epoch_label, save_dir=None):
        for i,_ in enumerate(self.drop_list):
            block_num = i//4
            layer_num = i%4 
            self.drop_list[i].plot_aggregated_statistics(epoch_label+f" Block {block_num} layer{layer_num}", save_dir)

    def update_progression(self):
        for i,_ in enumerate(self.drop_list):
            self.drop_list[i].update_progression()
 
    def plot_progression_statistics(self, save_dir=None,label =''):
        for i,_ in enumerate(self.drop_list):
            block_num = i//4
            layer_num = i%4
            self.drop_list[i].plot_progression_statistics(save_dir,label =label + f" Block_{block_num}_layer_{layer_num}")

    def clear_progression(self):
        for drop in self.drop_list:
            drop.clear_progression()
    def calculate_scores(self, batches: Iterable, device: torch.device,stats = True) -> None:
        # Create a detached copy of the model for IG computation.
        model_clone = copy.deepcopy(self)
        model_clone.to(device)
        model_clone.eval()  
        
        # Initialize conductances for each layer
        for i, _ in enumerate(model_clone.selected_layers):
            model_clone.scores[f'drop_{i}'] = None

        for batch in batches:
            x, _ = batch  # Batch is (samples, targets)
            x_captum = x.detach().clone().requires_grad_()
            x_captum = x_captum.to(device, non_blocking=True)
            baseline = torch.zeros_like(x_captum)

            # Get model predictions
            outputs = model_clone(x_captum)
            pred = outputs.argmax(dim=1)

            #calculate conductunce for batch
            mlc = MultiLayerConductance(model_clone, model_clone.selected_layers)
            captum_attrs = mlc.attribute(x_captum, baselines=baseline, target=pred, n_steps=model_clone.n_steps)

            # Average out the conductance across the batch and add it
            for i, score in enumerate(captum_attrs):
                score_mean = score.mean(dim=0)
                if model_clone.scores[f'drop_{i}'] is None:
                    # First time: initialize with the computed score_mean
                    model_clone.scores[f'drop_{i}'] = score_mean.clone()
                else:
                    # Accumulate the score_mean
                    model_clone.scores[f'drop_{i}'] += score_mean

        # Update the dropout masks based on the accumulated conductances
        for i, drop_layer in enumerate(model_clone.drop_list):
            drop_layer.update_dropout_masks(model_clone.scores[f'drop_{i}'], stats=stats)



        #load the update on the model from the copy
        for i,_ in enumerate(model_clone.drop_list):
            self.drop_list[i].load_state_dict(model_clone.drop_list[i].state_dict())
            self.drop_list[i].scaling = model_clone.drop_list[i].scaling.detach().clone()
            self.drop_list[i].previous = model_clone.drop_list[i].previous.detach().clone()
            self.drop_list[i].stats = model_clone.drop_list[i].stats.copy()
            self.drop_list[i].avg_scoring = model_clone.drop_list[i].avg_scoring
            self.drop_list[i].avg_dropout = model_clone.drop_list[i].avg_dropout
            self.drop_list[i].var_scoring = model_clone.drop_list[i].var_scoring
            self.drop_list[i].var_dropout = model_clone.drop_list[i].var_dropout
        print("Memory allocated:", torch.cuda.memory_allocated(device))
        print("Max memory reserved:", torch.cuda.max_memory_reserved(device))    
        del model_clone
        torch.cuda.empty_cache()

        self.train()





@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    # Remove extra keys that timm might not want
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('pretrained_cfg_overlay', None)
    kwargs.pop('cache_dir', None)

    # Extract relevant dropout / custom-dropout params
    drop = kwargs.pop('drop_rate', 0.0)
    ydrop = kwargs.pop('ydrop', True)
    mask_type = kwargs.pop('mask_type', 'sigmoid')
    elasticity = kwargs.pop('elasticity', 0.01)
    scaler = kwargs.pop('scaler', 1.0)
    n_steps = kwargs.pop('n_steps', 5)

    print("[Registered Model - Tiny] drop_rate:", drop)
    print("[Registered Model - Tiny] ydrop:", ydrop)
    print("[Registered Model - Tiny] mask_type:", mask_type)
    print("[Registered Model - Tiny] elasticity:", elasticity)
    print("[Registered Model - Tiny] scaler:", scaler)
    print("[Registered Model - Tiny] n_steps:", n_steps)

    from functools import partial
    from updated_transformer.block import Block
    from updated_transformer.mlp import Mlp

    # Create partial constructors for your custom Block & Mlp
    block_partial = partial(
        Block,
        ydrop=ydrop,
        mask_type=mask_type,
        elasticity=elasticity,
        scaler=scaler,
        attn_drop=drop,
        proj_drop=drop,
    )
    mlp_partial = partial(
        Mlp,
        ydrop=ydrop,
        mask_type=mask_type,
        elasticity=elasticity,
        scaler=scaler,
        drop=drop,  # Use the same drop or separate if desired
    )

    # Build MyVisionTransformer using your partials
    model = MyVisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        block_fn=block_partial,
        mlp_layer=mlp_partial,
        n_steps=n_steps,
        proj_drop_rate=drop,
        attn_drop_rate=drop,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
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
    # Remove extra keys that timm might not want
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('pretrained_cfg_overlay', None)
    kwargs.pop('cache_dir', None)

    # Extract relevant dropout / custom-dropout params
    drop = kwargs.pop('drop_rate', 0.0)
    ydrop = kwargs.pop('ydrop', True)
    mask_type = kwargs.pop('mask_type', 'sigmoid')
    elasticity = kwargs.pop('elasticity', 0.01)
    scaler = kwargs.pop('scaler', 1.0)
    n_steps = kwargs.pop('n_steps', 5)

    print("[Registered Model - Small] drop_rate:", drop)
    print("[Registered Model - Small] ydrop:", ydrop)
    print("[Registered Model - Small] mask_type:", mask_type)
    print("[Registered Model - Small] elasticity:", elasticity)
    print("[Registered Model - Small] scaler:", scaler)
    print("[Registered Model - Small] n_steps:", n_steps)

    from functools import partial
    from updated_transformer.block import Block
    from updated_transformer.mlp import Mlp

    # Create partial constructors for your custom Block & Mlp
    block_partial = partial(
        Block,
        ydrop=ydrop,
        mask_type=mask_type,
        elasticity=elasticity,
        scaler=scaler,
        attn_drop=drop,
        proj_drop=drop,
    )
    mlp_partial = partial(
        Mlp,
        ydrop=ydrop,
        mask_type=mask_type,
        elasticity=elasticity,
        scaler=scaler,
        drop=drop,
    )

    # Build MyVisionTransformer using your partials
    model = MyVisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        block_fn=block_partial,
        mlp_layer=mlp_partial,
        n_steps=n_steps,
        proj_drop_rate=drop,
        attn_drop_rate=drop,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
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
    # Remove extra keys that timm might not want
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('pretrained_cfg_overlay', None)
    kwargs.pop('cache_dir', None)

    drop = kwargs.pop('drop_rate', 0.0)
    ydrop = kwargs.pop('ydrop', True)
    mask_type = kwargs.pop('mask_type', 'sigmoid')
    elasticity = kwargs.pop('elasticity', 0.01)
    scaler = kwargs.pop('scaler', 1.0)
    n_steps = kwargs.pop('n_steps', 5)

    print("[Registered Model] drop_rate:", drop)
    print("[Registered Model] ydrop:", ydrop)
    print("[Registered Model] mask_type:", mask_type)
    print("[Registered Model] elasticity:", elasticity)
    print("[Registered Model] scaler:", scaler)
    print("[Registered Model] n_steps:", n_steps)

    from functools import partial
    from updated_transformer.block import Block
    from updated_transformer.mlp import Mlp

    # Make partial constructors
    block_partial = partial(
        Block,
        ydrop=ydrop,
        mask_type=mask_type,
        elasticity=elasticity,
        scaler=scaler,
        attn_drop=drop,
        proj_drop=drop,
    )
    mlp_partial = partial(
        Mlp,
        ydrop=ydrop,
        mask_type=mask_type,
        elasticity=elasticity,
        scaler=scaler,
        drop=drop,  # use the same drop for MLP as well, or pass differently
    )

    model = MyVisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        block_fn=block_partial,
        mlp_layer=mlp_partial,
        n_steps=n_steps,
        proj_drop_rate=drop,
        attn_drop_rate=drop,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

