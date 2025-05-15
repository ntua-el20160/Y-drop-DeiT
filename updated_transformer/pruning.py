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
from evaluate_gradients.MultiLayerSensitivity import MultiLayerSensitivity
from timm.models.vision_transformer import VisionTransformer, _cfg, LayerScale
from timm.models import register_model
from timm.layers import PatchEmbed,use_fused_attn,DropPath, trunc_normal_

from updated_transformer.block import Block
from updated_transformer.mlp import Mlp

def pruuning(model: torch.nn.Module,data_loader: Iterable, device: torch.device,scoring_type = "Conductance",batches_num:int =10) -> None:
        # Create a detached copy of the model for IG computation.
        # model_clone = copy.deepcopy(model)
        # model_clone.to(device)
        model.eval()  
        
        # Initialize conductances for each layer
        for i, _ in enumerate(model.selected_layers):
            model.scores[f'drop_{i}'] = None
        new_iter = iter(data_loader)
        for i in range(batches_num):
            batch = next(new_iter)  # Get a batch of data
            x, _ = batch  # Batch is (samples, targets)
            x_captum = x.detach().clone().requires_grad_()
            x_captum = x_captum.to(device, non_blocking=True)
            baseline = torch.zeros_like(x_captum)

            # Get model predictions
            outputs = model(x_captum)
            pred = outputs.argmax(dim=1)

            #calculate conductunce for batch
            if scoring_type == "Conductance":
                mlc = MultiLayerConductance(model, model.selected_layers)
                captum_attrs = mlc.attribute(x_captum, baselines=baseline, target=pred, n_steps=model_clone.n_steps)
            elif scoring_type == "Sensitivity":
                mlc = MultiLayerSensitivity(model, model.selected_layers)
                captum_attrs = mlc.attribute(x_captum, baselines=baseline, target=pred, n_steps=model_clone.n_steps)
            else:
                print("Invalid scoring type. Using Conductance as default.")
                mlc = MultiLayerConductance(model, model.selected_layers)
                captum_attrs = mlc.attribute(x_captum, baselines=baseline, target=pred, n_steps=model_clone.n_steps)

            # Average out the conductance across the batch and add it
            for i, score in enumerate(captum_attrs):
                #print(captum_attrs)
                score_mean = score if scoring_type == "Sensitivity" else score.mean(dim=0)

                if model.scores[f'drop_{i}'] is None:
                    # First time: initialize with the computed score_mean
                    model.scores[f'drop_{i}'] = score_mean.clone()
                else:
                    # Accumulate the score_mean
                    model.scores[f'drop_{i}'] += score_mean

        # Update the dropout masks based on the accumulated conductances
        for i, drop_layer in enumerate(model_clone.drop_list):
            drop_layer.update_dropout_masks(model_clone.scores[f'drop_{i}'], stats=stats,update_freq=update_freq)



        #load the update on the model from the copy
        for i,_ in enumerate(model_clone.drop_list):
            self.drop_list[i].load_state_dict(model_clone.drop_list[i].state_dict())
            self.drop_list[i].scaling = model_clone.drop_list[i].scaling.detach().clone()
            self.drop_list[i].previous = model_clone.drop_list[i].previous.detach().clone()
            self.drop_list[i].running_scoring_mean = model_clone.drop_list[i].running_scoring_mean
            self.drop_list[i].running_dropout_mean = model_clone.drop_list[i].running_dropout_mean
            self.drop_list[i].keep_hist = model_clone.drop_list[i].keep_hist
            self.drop_list[i].scoring_hist = model_clone.drop_list[i].scoring_hist
            self.drop_list[i].progression_scoring = model_clone.drop_list[i].progression_scoring
            self.drop_list[i].progression_keep = model_clone.drop_list[i].progression_keep
            self.drop_list[i].sum_scoring = model_clone.drop_list[i].sum_scoring
            self.drop_list[i].sum_keep = model_clone.drop_list[i].sum_keep
        del model_clone
        torch.cuda.empty_cache()

        self.train()
        return outputs