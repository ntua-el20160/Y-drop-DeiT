import math
import torch
import torch.nn as nn
from functools import partial
from torch.jit import Final
from typing import Type, Optional, Iterable, Dict, List
import torch.nn.functional as F
import copy
from captum.attr import LayerConductance
from evaluate_gradients.MultiLayerConductance import MultiLayerConductance   
from evaluate_gradients.MultiLayerSensitivity import MultiLayerSensitivity
def _is_main_process():
    # rank 0 if DDP, else True
    try:
        import torch.distributed as dist
        return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    except Exception:
        return True
    
def p(*args, **kwargs):
    if _is_main_process():
        print(*args, **kwargs)

def tstats(t: torch.Tensor, name: str, extra: str = ""):
    if t is None:
        p(f"[{name}] = None {extra}")
        return
    with torch.no_grad():
        nan = torch.isnan(t).any().item() if t.is_floating_point() else 0
        inf = torch.isinf(t).any().item() if t.is_floating_point() else 0
        shape = tuple(t.shape)
        dev = str(t.device)
        dt  = str(t.dtype)
        if t.numel() > 0 and t.is_floating_point():
            mn  = t.min().item()
            mx  = t.max().item()
            mean= t.float().mean().item()
            std = t.float().std().item()
            p(f"[{name}] shape={shape} dtype={dt} device={dev} min={mn:.4g} max={mx:.4g} mean={mean:.4g} std={std:.4g} nan={int(nan)} inf={int(inf)} {extra}")
        else:
            p(f"[{name}] shape={shape} dtype={dt} device={dev} nan={int(nan)} inf={int(inf)} {extra}")

def boolfrac(b: torch.Tensor) -> float:
    if b is None or b.numel() == 0: return float("nan")
    return b.float().mean().item()

def hist1d(t: torch.Tensor, bins=10):
    if t is None or t.numel() == 0: return "[]"
    t = t.detach().float()
    lo, hi = float(t.min()), float(t.max())
    if lo == hi: return f"[all {lo:.4g}]"
    h = torch.histc(t, bins=bins, min=lo, max=hi)
    edges = torch.linspace(lo, hi, bins+1)
    parts = [f"{edges[i].item():.2g}–{edges[i+1].item():.2g}:{int(h[i].item())}" for i in range(bins)]
    return "[" + ", ".join(parts) + "]"
def calculate_scores(
        model: torch.nn.Module,
        batches: Iterable,
        device: torch.device,
        scoring_type: str = "Conductance",
        mode: bool = "mean",
        normalization: bool = False,
        selected_layers: Optional[List[int]] = None,
        sm = False,
        baseline = None,
        n_steps = 4) -> Dict[int, torch.Tensor]:
    # 1) --- ensure model is in eval mode and gradients are disabled
    model.eval()
    if selected_layers is  None:
        selected_layers = model.selected_layers

    # 2) --- save original requires_grad settings
    orig_reqs = []
    for param in model.parameters():
        orig_reqs.append(param.requires_grad)
        param.requires_grad_(False)
    model.zero_grad()

    # [NEW] we will lazily (re)build the captum object if we need to switch to alt for MLM
    if scoring_type == "Conductance":
        mlc = MultiLayerConductance(model.pred_scalar_forward, selected_layers)
    else:
        # For MLM, standard Conductance on "target class" is awkward; we attribute to LOSS
        mlc = MultiLayerConductance(model.crit_for, selected_layers)
    # 4) --- iterate over batches
    batch_count = 0
    new_scores = {}

    for x, y_batch in batches:
        batch_count += 1
      

        # [NEW] --------- Detect BERT-MLM batch vs. old vision batch ----------
        # We support two NLP shapes:
        #   (A) x is a dict with "input_ids" (+ masks), y_batch is labels [B,T]
        #   (B) x is a tuple: (x_unmasked, x_masked, labels, attention_mask, token_type_ids)
        # for key, value in x.items():
        #         print(f"  {key}: {type(value)}")
        if isinstance(x, dict) and "input_ids" in x:
            input_ids     = x["input_ids"]
            attention_mask = x.get("attention_mask", None)
            token_type_ids = x.get("token_type_ids", None)
            labels         = y_batch
        elif isinstance(x, (tuple, list)) and len(x) >= 3:
            # tuple layout: (x_unmasked, x_masked, labels, attention_mask, token_type_ids)
            # Use the masked ids as the actual input to the model forward.
            # If you prefer unmasked, swap x_masked with x_unmasked here.
            x_unmasked, x_masked, labels = x[0], x[1], x[2]
            attention_mask = x[3] if len(x) > 3 else None
            token_type_ids = x[4] if len(x) > 4 else None
            input_ids = x_masked

        else:
            raise ValueError(
                "calculate_scores_nlp expects (inputs_dict, labels) or "
                "(x_unmasked, x_masked, labels, attention_mask, token_type_ids)."
            )
        # if batch_count <= 2:  # only first couple batches to keep logs light
        #     tstats(input_ids, "input_ids")
        #     if attention_mask is not None:
        #         tstats(attention_mask, "attention_mask", extra=f" frac_true={boolfrac(attention_mask):.3f}")
        #     if token_type_ids is not None:
        #         tstats(token_type_ids, "token_type_ids")
        #     if labels is not None:
        #         mask_pos = (labels != -100)
        #         p(f"masked_frac (labels!=-100): {boolfrac(mask_pos):.3f}")
        #         lab_vals = labels[mask_pos]
        #         if lab_vals.numel() > 0:
        #             p(f"labels(hist on masked): {hist1d(lab_vals.float(), bins=10)}")
        # # ---------------------------------------------------------------------

        # 5) --- ensure input ids (or images) on device and requires_grad
        # x_captum = input_ids.detach().clone().requires_grad_()
        # x_captum = x_captum.to(device, non_blocking=True)
        embed_layer = model.get_input_embeddings() 
        x_captum = embed_layer(input_ids.to(device, non_blocking=True)).detach().clone().requires_grad_()

        # [NEW] move aux tensors if present (MLM case)
        attention_mask = attention_mask.to(device, non_blocking=True) if attention_mask is not None else None
        token_type_ids = token_type_ids.to(device, non_blocking=True) if token_type_ids is not None else None
        labels = labels.to(device, non_blocking=True) if labels is not None else None

        def _token_id_from(cfg, tok_attr):
            # try config
            tid = getattr(cfg, tok_attr, None)
            if tid is not None:
                return tid
            # fallback to tokenizer if attached
            tok = getattr(model, "tokenizer", None)
            if tok is None:
                return None
            # try tokenizer.<mask_token_id> / <pad_token_id>
            tid = getattr(tok, tok_attr, None)
            if tid is not None:
                return tid
            # try tokenizer.<mask_token> / <pad_token> then convert to id
            name_attr = tok_attr.replace("_id", "")  # e.g. "mask_token"
            name = getattr(tok, name_attr, None)
            if name is not None:
                try:
                    return tok.convert_tokens_to_ids(name)
                except Exception:
                    return None
            return None

        
        emb_dtype = embed_layer.weight.dtype
        B, T, H = x_captum.shape
        # baseline: if none provided, use zeros_like(x) as you already did
        # local_baseline = baseline
        # if local_baseline is None:
        if baseline in ("zeroes", "zeros", "zero"):
            local_baseline = torch.zeros_like(x_captum)
        elif baseline in ("random", "noise", "random_noise"):
                # Gaussian noise in embedding space; scale to embedding std for stability
                std = float(embed_layer.weight.detach().float().std().clamp_min(1e-6))
                local_baseline = torch.randn_like(x_captum) * std
        elif baseline == "mask":
            # [MASK] token embedding as baseline (word embedding space)
            mask_id = _token_id_from(model.config, "mask_token_id")
            if mask_id is None:
                # raise ValueError("Baseline 'mask' requested but model.config.mask_token_id is None.")
                mask_id = 103 # common default for BERT-based models
            mask_vec = embed_layer(torch.tensor([mask_id], device=device)).to(dtype=emb_dtype)  # [1, H]
            local_baseline = mask_vec.view(1, 1, H).expand(B, T, H).detach()
        elif baseline in ("pad", "padding"):
            pad_id = _token_id_from(model.config, "pad_token_id")
            if pad_id is None:
                raise ValueError("Baseline 'pad' requested but model.config.pad_token_id is None.")
            pad_vec = embed_layer(torch.tensor([pad_id], device=device)).to(dtype=emb_dtype)    # [1, H]
            local_baseline = pad_vec.view(1, 1, H).expand(B, T, H).detach()
        else:
            raise ValueError(f"Unknown baseline kind: {baseline}")
        
        #local_baseline.requires_grad_(False)

        # labels / targets to device
        y_batch = labels.to(device, non_blocking=True) if labels is not None else None

        # 6) --- forward pass and "prediction"
        # [NEW] call the model with kwargs for NLP; else old positional call
        with torch.no_grad():
            # outputs = model(input_ids=x_captum,
            #                 attention_mask=attention_mask,
            #                 token_type_ids=token_type_ids)
            outputs = model(inputs_embeds=x_captum,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)


        # [NEW] figure out output shape; if it's [B, T, V], we are in MLM
        logits = getattr(outputs, "logits", outputs)
        # if batch_count <= 2:
        #     tstats(logits, "logits")
        #     if logits.ndim == 3:  # [B,T,V]
        #         if attention_mask is not None and labels is not None:
        #             p(f"seq_len: logits.T={logits.shape[1]} labels.T={labels.shape[1]} attn.T={attention_mask.shape[1]}")

        #is_mlm = logits.ndim == 3  # [B, T, V]
        pred = logits.argmax(dim=-1)
        # if batch_count == 1:
        #     p(f"scoring_type={scoring_type}, n_steps={n_steps}, "
        #     f"baseline={'given' if baseline is not None else 'zeros_like(input)'} "
        #     f"selected_layers={selected_layers}")
        if scoring_type == "Conductance_alt":
            captum_out = mlc.attribute(
                x_captum,
                baselines=local_baseline,
                target=None,                       # loss-based path ignores target
                n_steps=n_steps,
                internal_batch_size=None,
                additional_forward_args=(attention_mask, labels, token_type_ids),
                return_convergence_delta=False,
                attribute_to_layer_input=False,
                grad_kwargs={"retain_graph": False},
            )
        else:
            captum_out = mlc.attribute(
                x_captum,
                baselines=local_baseline,
                target=None,
                n_steps=n_steps,
                internal_batch_size=None,
                additional_forward_args=(attention_mask, labels, token_type_ids),
                return_convergence_delta=False,
                attribute_to_layer_input=False,
                grad_kwargs={"retain_graph": False},
            )
        # old classification path: [B, C]
        # if not is_mlm:
        #     pred = logits.argmax(dim=1)
        # else:
        #     pred = None  # not used for loss-based conductance
        # if batch_count <= 2:
        #     if isinstance(captum_out, (list, tuple)):
        #         p("captum_out parts: " + ", ".join([str(tuple(t.shape)) for t in captum_out]))
        #     else:
        #         p(f"captum_out shape: {tuple(captum_out.shape)}")
        # 8) --- process captum output
        if isinstance(captum_out, list):
            captum_attrs = [t.detach() for t in captum_out]
        elif isinstance(captum_out, tuple):
            captum_attrs = tuple(t.detach() for t in captum_out)
        else:
            captum_attrs = [captum_out.detach()]

        # 9) --- accumulate scores
        for i, score in enumerate(captum_attrs):

            if sm:
                score_mean = score.sum(dim=0)
            else:
                score_mean = score.mean(dim=0)
            # if batch_count == 1 and i == 0:
            #     tstats(score, "captum_score_raw[layer0]")
            #     tstats(score_mean, f"score_mean(mode={mode})[layer0]")

            if mode == "cls":
                score_mean = score_mean[0]
            elif mode == "mean":
                score_mean = score_mean.mean(dim=0)
            elif mode == "sum":
                score_mean = score_mean.sum(dim=0)
            elif mode == "topk":
                topk_tokens = 10
                token_scores = score_mean.sum(dim=1)
                idx = token_scores.topk(topk_tokens).indices
                score_mean =  score_mean[idx].mean(dim=0)
            elif mode == 'nothing':
                pass

            if i not in new_scores:
                new_scores[i] = score_mean.clone()
            else:
                new_scores[i] += score_mean

    # 10) compute means for each layer
    # p(f"calculate_scores: processed {batch_count} sub-batches")

    for i in new_scores:
        new_scores[i] /= batch_count

    means = [s.mean() for s in new_scores.values() if s is not None]

    # 11) --- normalize scores if required
    if normalization:
        for i in range(len(new_scores)):
            if new_scores[i] is not None:
                new_scores[i] = (new_scores[i] - new_scores[i].mean()) / (new_scores[i].std() + 1e-6)  # [NEW] safer denom

    # 12) --- restore original requires_grad settings
    for param, req in zip(model.parameters(), orig_reqs):
        param.requires_grad_(req)

    model.train()
    return new_scores, means
