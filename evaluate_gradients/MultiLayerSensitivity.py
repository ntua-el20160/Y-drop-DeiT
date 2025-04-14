from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal
import torch
from torch import Tensor
from torch.nn import Module
from captum.attr._utils.attribution import GradientAttribution, LayerAttribution
from captum.log import log_usage
from captum._utils.common import _format_output

class MultiLayerSensitivity(LayerAttribution, GradientAttribution):
    r"""
    Computes sensitivity with respect to parameters for the given layers.
    For each parameter (weight, bias, etc.), sensitivity is defined as:
        sensitivity = |theta * grad(theta)|
    The final output is aggregated so that for each layer the sensitivity
    is returned as a tensor whose shape matches the layer's output dimension.
      - For a linear layer, if weight is of shape [out_features, in_features],
        we sum the weight sensitivity over the input dimension to yield [out_features].
        If a bias is present (shape [out_features]), it is added to the aggregated weight sensitivity.
    The output is returned as a list, one element per layer.
    """

    def __init__(
        self,
        forward_func: Callable[..., Tensor],
        layer: Union[Module, List[Module]],
        device_ids: Optional[List[int]] = None,
    ) -> None:
        if not isinstance(layer, list):
            self.layer = [layer]
        else:
            self.layer = layer
        # Initialize base attribution classes.
        LayerAttribution.__init__(self, forward_func, self.layer, device_ids)
        GradientAttribution.__init__(self, forward_func)

    @log_usage()
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        target: Optional[Union[int, Tuple[int, ...]]] = None,
        additional_forward_args: Optional[Any] = None,
        **kwargs,  # Ignore n_steps, baselines etc.
    ) -> Union[List[Tensor]]:
        # Ensure inputs is a tuple.
        if isinstance(inputs, Tensor):
            inputs = (inputs,)
        
        # Forward pass: for sensitivity we do not need baselines,
        # so simply call forward_func with the inputs.
        outputs = self.forward_func(*inputs)
        
        # If a target is provided, select the corresponding outputs.
        # (Assumes outputs is at least 2D, e.g., [N, ...])
        if target is not None:
            outputs = outputs[torch.arange(outputs.shape[0]), target]
        
        # Collect all parameters from all layers.
        params = []
        for layer in self.layer:
            for _, param in layer.named_parameters():
                params.append(param)
        
        # Compute gradients with respect to these parameters.
        # We assume that outputs.sum() is scalar.
        grads = torch.autograd.grad(outputs.sum(), params, retain_graph=True, create_graph=False)
        
        # Process each layer to compute an aggregated sensitivity per layer.
        # For a linear layer with weight shape [out_features, in_features],
        # we will sum the weight sensitivity over the in_features to yield [out_features].
        sens_list: List[Tensor] = []
        i = 0
        for layer in self.layer:
            # Initialize placeholders for weight and bias sensitivity.
            layer_weight_agg: Optional[Tensor] = None
            layer_bias_agg: Optional[Tensor] = None
            for name, param in layer.named_parameters():
                grad_param = grads[i]
                # Compute elementwise sensitivity: |param * grad|
                sens = torch.abs(param * grad_param)
                i += 1
                if "weight" in name:
                    # For weights, sum over the input dimension.
                    # (Assumes weight shape is [out_features, in_features])
                    if sens.dim() > 1:
                        layer_weight_agg = sens.sum(dim=1)
                    else:
                        layer_weight_agg = sens
                elif "bias" in name:
                    # For biases, use the sensitivity directly.
                    layer_bias_agg = sens
                else:
                    # If other parameters exist, you can define a default aggregation.
                    pass
            # Combine weight and bias sensitivity.
            if layer_weight_agg is None:
                combined = layer_bias_agg
            elif layer_bias_agg is None:
                combined = layer_weight_agg
            else:
                combined = layer_weight_agg + layer_bias_agg
            sens_list.append(combined)
        
        # Format output similar to conductance (if a single module was provided).
        if isinstance(self.layer, Module):
            return _format_output(False, sens_list)
        else:
            return sens_list

    @property
    def multiplies_by_inputs(self) -> bool:
        # Sensitivity here is computed with respect to parameters.
        return False
