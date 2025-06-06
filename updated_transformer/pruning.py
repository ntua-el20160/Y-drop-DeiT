import torch
import torch.nn as nn
from typing import Dict, List, Tuple

def prune_selected_layers(
    model: nn.Module,
    prune_map: Dict[int, List[int]]
) -> None:
    """
    Given a model with `model.selected_layers = [layer0, layer1, ...]`
    and a `prune_map` mapping each selected-layer index i to a list of
    output-neuron indices to remove from that layer, this function
    (in-place) rebuilds each selected nn.Linear so that its outputs
    are pruned. It then also adjusts the *inputs* of the next Linear
    in the chain (whether selected or not) so that dimensions match.
    
    Assumptions:
    1. `model.selected_layers` is a Python list of nn.Linear modules,
       in the order they appear in the forward pass (so output of
       selected_layers[i] feeds directly into selected_layers[i+1], or
       into a final classifier).
    2. The final classifier (e.g. `model.fc3` in CNN6_S1) is a
       nn.Linear whose `in_features` == the original `out_features`
       of the last selected layer. We will prune its input columns.
    
    This function replaces each module in-place, so no return value.
    """

    # ------------------------------------------------------------------------
    # Helper to find the (qualified) name of a module inside `model`
    # and return (parent_module, attribute_name) so we can call setattr.
    # ------------------------------------------------------------------------
    def _find_parent_and_attr(root: nn.Module, child: nn.Module) -> Tuple[nn.Module, str]:
        """
        Traverse root.named_modules() to locate where `child` appears as a direct
        attribute. Returns (parent_module, attribute_name) so that
            setattr(parent_module, attribute_name, new_module)
        will replace it.
        """
        #root: the whole model, child: a submodule (nn.Linear)
        #parent_module: the parent of child(holds child as one of it's attributes), attribute_name: the name of child in parent_module
        
        for qualified_name, module in root.named_modules():#searches all modules in the model, they have the form like (strin:"conv1", the actual module: nn.Conv2d)
            if module is child:
                # If qualified_name == "", then child == root itself. We do not
                # handle replacing root; assume selected_layers are submodules.
                if "." not in qualified_name:
                    return root, qualified_name  # rare: child is assigned as root.attr ""
                # Otherwise split by last dot
                parent_name, attr_name = qualified_name.rsplit(".", 1)
                parent = root
                for part in parent_name.split("."):
                    parent = getattr(parent, part)
                return parent, attr_name
        raise ValueError(f"Module {child} not found in model as a submodule.")

    # ------------------------------------------------------------------------
    # Helper to prune a single nn.Linear's outputs by index list.
    # ------------------------------------------------------------------------
    def _prune_linear_outputs(orig_linear: nn.Linear, prune_indices: List[int]) -> Tuple[nn.Linear, torch.Tensor]:
        """
        Build a new nn.Linear whose `out_features = orig_out - len(prune_indices)`,
        copying over the surviving rows of weight & bias. Also return a boolean
        `keep_mask` of length orig_out, True if that output index is kept.
        """
        orig_out, orig_in = orig_linear.out_features, orig_linear.in_features
        # Build a mask of length=orig_out
        keep_mask = torch.ones(orig_out, dtype=torch.bool)
        keep_mask[prune_indices] = False
        new_out = keep_mask.sum().item()

        # Create the new Linear
        new_linear = nn.Linear(in_features=orig_in, out_features=new_out, bias=(orig_linear.bias is not None))

        with torch.no_grad():
            # Copy surviving rows of weight: orig_linear.weight has shape [orig_out, orig_in]
            new_linear.weight.copy_(orig_linear.weight[keep_mask, :])
            # Copy surviving bias entries if present
            if orig_linear.bias is not None:
                new_linear.bias.copy_(orig_linear.bias[keep_mask])

        return new_linear, keep_mask

    # ------------------------------------------------------------------------
    # Helper to prune a single nn.Linear's inputs by a boolean mask
    # (i.e. remove certain columns). Returns a new nn.Linear.
    # ------------------------------------------------------------------------
    def _prune_linear_inputs(orig_linear: nn.Linear, keep_mask: torch.Tensor) -> nn.Linear:
        """
        Given an existing nn.Linear with weight shape [out_features, in_features],
        and a boolean keep_mask of length in_features, create a new nn.Linear
        with in_features = keep_mask.sum(), copying only the columns where keep_mask=True.
        Bias is unchanged.
        """
        orig_out, orig_in = orig_linear.out_features, orig_linear.in_features
        if keep_mask.numel() != orig_in:
            raise ValueError(f"Input-mask length {keep_mask.numel()} != orig in_features {orig_in}")

        new_in = keep_mask.sum().item()
        new_linear = nn.Linear(in_features=new_in, out_features=orig_out, bias=(orig_linear.bias is not None))

        with torch.no_grad():
            # Copy only columns where keep_mask is True
            kept_cols = keep_mask.nonzero(as_tuple=False).squeeze(1)  # shape [new_in]
            # orig_linear.weight is [orig_out, orig_in]
            new_linear.weight.copy_(orig_linear.weight[:, kept_cols])
            if orig_linear.bias is not None:
                new_linear.bias.copy_(orig_linear.bias)

        return new_linear

    # --------------------------------------------------------------
    # Step 1: Locate each selected layer in `model` by name, so we can replace it.
    # We'll build a list of (layer_module, parent_module, attr_name).
    # --------------------------------------------------------------
    selected = model.selected_layers  # list of nn.Module (Linear) references
    layer_replacements = []  # will hold tuples [(orig_layer, parent, attr), ...]
    for layer in selected:
        parent_mod, attr_name = _find_parent_and_attr(model, layer)
        layer_replacements.append((layer, parent_mod, attr_name))

    # --------------------------------------------------------------
    # Step 2: Prune in a chain. For each selected layer i:
    #   (A) Prune its outputs using prune_map[i] → obtain new_layer_i, keep_mask_i
    #   (B) Replace the module in the model graph
    #   (C) Prune the *inputs* of the next Linear:
    #         if next selected layer exists, prune its inputs by keep_mask_i
    #         else prune all Linear modules whose in_features match original out_i.
    # We do this in the order of selected_layers.
    # --------------------------------------------------------------
    # We keep track of the original out_features of each layer to find downstream match.
    original_out_features = [layer.out_features for layer in selected]

    for idx, (orig_layer, parent_mod, attr_name) in enumerate(layer_replacements):
        # 2.A) Prune orig_layer's outputs
        to_prune = prune_map.get(idx, [])
        new_layer, keep_mask = _prune_linear_outputs(orig_layer, to_prune)

        # 2.B) Replace orig_layer with new_layer in the model
        setattr(parent_mod, attr_name, new_layer)

        # 2.C) Now prune inputs of the next layer in the chain
        # If there is a next selected layer:
        if idx + 1 < len(layer_replacements):
            next_layer, next_parent, next_attr = layer_replacements[idx + 1]
            # But note: next_layer is still the *old* nn.Linear instance; we must replace it too.
            # First, prune its inputs by the keep_mask we just produced.
            pruned_next = _prune_linear_inputs(next_layer, keep_mask)
            # Now replace next_layer with this pruned‐input version *temporarily*; we will prune its outputs later,
            # so update `layer_replacements[idx+1]` to point to `pruned_next` so that the output‐pruning step
            # uses the updated module.
            setattr(next_parent, next_attr, pruned_next)
            # Update our tuple so that the next iteration sees pruned_next as the layer to prune outputs of.
            layer_replacements[idx + 1] = (pruned_next, next_parent, next_attr)

        else:
            # No next selected layer: we prune any final classifier(s) that consume orig_layer's outputs.
            # We look for any nn.Linear in model whose in_features == original_out_features[idx].
            # For each such linear, we prune its inputs by keep_mask.
            consumed_out = original_out_features[idx]
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and module.in_features == consumed_out:
                    # Locate parent so we can replace it
                    parent_mod2, attr_name2 = _find_parent_and_attr(model, module)
                    pruned_final = _prune_linear_inputs(module, keep_mask)
                    setattr(parent_mod2, attr_name2, pruned_final)

    # --------------------------------------------------------------
    # All selected layers (and downstream classifiers) have now been replaced
    # in the original model in-place. No return value.
    # --------------------------------------------------------------
