# ydrop_utils.py
# ----------------------------------------------
# Utilities to replace selected Dropout modules
# with your MyDropout and to collect paired LNs.
# ----------------------------------------------

from typing import List, Tuple, Sequence, Type
import torch.nn as nn

# Import your custom dropout
# Adjust the path if needed.
from updated_transformer.dynamic_dropout import MyDropout


def collect_encoder_dropouts_and_selected_layers(
    model: nn.Module,
    *,
    encoder_prefix: str = "bert.encoder",
    dropout_types: Sequence[Type[nn.Module]] = (MyDropout,),
) -> Tuple[List[nn.Module], List[nn.Module]]:
    """
    Find replaced dropouts in the encoder and align each with its LayerNorm:

      keep:
        - ...attention.output.dropout   -> ...attention.output.LayerNorm
        - ...output.dropout (FFN)       -> ...output.LayerNorm
      skip:
        - ...attention.self.dropout

    Returns:
      (drop_list, selected_layers)  aligned 1:1
    """
    drop_list: List[nn.Module] = []
    selected_layers: List[nn.Module] = []

    name_to_module = dict(model.named_modules())

    def _append(drop_name: str, drop_mod: nn.Module, sel_name: str):
        ln = name_to_module.get(sel_name)
        if ln is None:
            raise KeyError(f"Could not resolve LayerNorm for {drop_name} (looked for {sel_name})")
        drop_list.append(drop_mod)
        selected_layers.append(ln)

    for name, module in model.named_modules():
        if not name.startswith(encoder_prefix):
            continue
        if not isinstance(module, tuple(dropout_types)):
            continue

        if name.endswith("attention.output.dropout"):
            base = name.rsplit(".dropout", 1)[0]  # ...attention.output
            _append(name, module, base + ".LayerNorm")
            continue

        if name.endswith("output.dropout") and ".attention." not in name:
            base = name.rsplit(".dropout", 1)[0]  # ...layer.X.output
            _append(name, module, base + ".LayerNorm")
            continue

        # skip self-attention dropout explicitly
        if name.endswith("attention.self.dropout"):
            continue

    assert len(drop_list) == len(selected_layers)
    return drop_list, selected_layers


def replace_dropout(
    module: nn.Module,
    *,
    drop_rate: float = 0.1,
    elasticity: float = 0.01,
    mask_type: str = "sigmoid",
    transformer_mean: bool = True,
):
    """
    Replace nn.Dropout with MyDropout everywhere EXCEPT:
      - ...attention.self.dropout (left untouched)

    Other nn.Dropout (including embeddings/head) will be replaced.
    Change the recursion if you want to limit to encoder only.
    """
    def _rec(mod: nn.Module, prefix: str = ""):
        for name, child in mod.named_children():
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Dropout):
                if full.endswith("attention.self.dropout"):
                    # keep original Dropout at self-attention site
                    continue
                # replace with your MyDropout
                setattr(mod, name, MyDropout(
                    p=drop_rate,
                    elasticity=elasticity,
                    mask_type=mask_type,
                    transformer_mean=transformer_mean,
                ))
            else:
                _rec(child, full)

    _rec(module)
