import torch
import torch.nn.functional as F
from typing import Iterable, Optional, Dict, List, Tuple
from evaluate_gradients.MultiLayerConductance import MultiLayerConductance

# ---------- Forward wrappers ----------
def _forward_mlm_with_embeds_factory_pooled(model):
    """
    (inputs_embeds, attention_mask, token_type_ids, mlm_mask) -> [B, V]
    Pools token logits over masked positions so Captum targets are [B].
    """
    def forward_logits_pooled(inputs_embeds, attention_mask=None, token_type_ids=None, mlm_mask=None):
        out = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        logits = out.logits  # [B, L, V]
        if mlm_mask is None:
            return logits.mean(dim=1)  # [B, V] fallback
        mlm_mask = mlm_mask.to(dtype=logits.dtype)
        denom = mlm_mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1)  # [B,1,1]
        pooled = (logits * mlm_mask.unsqueeze(-1)).sum(dim=1) / denom.squeeze(-1)  # [B, V]
        return pooled
    return forward_logits_pooled

def _forward_mlm_loss_with_embeds_factory(model):
    """
    (inputs_embeds, attention_mask, token_type_ids, labels) -> [B] per-example MLM loss
    Good for Conductance_alt (attribute to loss).
    """
    def forward_loss(inputs_embeds, attention_mask=None, token_type_ids=None, labels=None):
        out = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        logits = out.logits  # [B, L, V]
        if labels is None:
            return logits.mean(dim=(1, 2))  # [B] safe fallback
        B, L, V = logits.shape
        loss_flat = F.cross_entropy(
            logits.reshape(B * L, V),
            labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(B, L)  # [B, L]
        counts = (labels != -100).sum(dim=1).clamp_min(1)  # [B]
        loss_per_example = loss_flat.sum(dim=1) / counts   # [B]
        return loss_per_example
    return forward_loss

def _majority_non_ignore_per_row(labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    B = labels.size(0)
    out = torch.zeros(B, dtype=torch.long, device=labels.device)
    for b in range(B):
        row = labels[b]
        vals = row[row != ignore_index]
        out[b] = vals.mode().values if vals.numel() > 0 else 0
    return out

# ---------- Main scorer ----------
def calculate_mlm_scores(
    model: torch.nn.Module,
    tokenizer,
    batches: Iterable,                       # dicts: input_ids, attention_mask?, token_type_ids?, labels?
    device: torch.device,
    *,
    scoring_type: str = "Conductance",       # "Conductance" | "Conductance_alt" | "Conductance_pooled"
    baseline: str = "mask",                  # "mask" | "pad"
    target_mode: str = "pred",               # "pred" | "label" (ignored by Conductance_alt)
    selected_layers: Optional[List[torch.nn.Module]] = None,
    n_steps: int = 5,
    normalization: bool = False,
    sm: bool = False,                         # batch reduce: sum (True) or mean (False)
    mode: Optional[str] = "masked",              # None | "cls" | "mean" | "sum" | "topk" | "masked"
) -> Tuple[Dict[int, torch.Tensor], List[torch.Tensor]]:

    model.eval()
    if selected_layers is None:
        if hasattr(model, "selected_layers"):
            selected_layers = model.selected_layers
        else:
            raise ValueError("Please pass `selected_layers` (list of nn.Module) for the layers to score.")

    # freeze params
    orig_reqs = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    model.zero_grad(set_to_none=True)

    # 1) ✅ Always use pooled logits for "Conductance" / "Conductance_pooled"
    if scoring_type == "Conductance_alt":
        forward_func = _forward_mlm_loss_with_embeds_factory(model)
    else:
        forward_func = _forward_mlm_with_embeds_factory_pooled(model)

    mlc = MultiLayerConductance(forward_func, selected_layers)

    emb_layer = model.get_input_embeddings()
    # baseline embeddings
    if baseline == "mask":
        if tokenizer.mask_token_id is None:
            raise ValueError("Tokenizer has no [MASK] token; choose baseline='pad'.")
        baseline_id = tokenizer.mask_token_id
    elif baseline == "pad":
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer has no [PAD] token; set tokenizer.pad_token_id first.")
        baseline_id = tokenizer.pad_token_id
    else:
        raise ValueError("baseline must be 'mask' or 'pad'")

    new_scores: Dict[int, torch.Tensor] = {}
    batch_count = 0

    for batch in batches:
        input_ids      = batch["input_ids"].to(device, non_blocking=True)          # [B, L]
        attention_mask = batch.get("attention_mask")
        attention_mask = attention_mask.to(device, non_blocking=True) if attention_mask is not None else None
        token_type_ids = batch.get("token_type_ids")
        token_type_ids = token_type_ids.to(device, non_blocking=True) if token_type_ids is not None else None
        labels         = batch.get("labels")
        labels         = labels.to(device, non_blocking=True) if labels is not None else None

        # embeddings to attribute
        inputs_embeds = emb_layer(input_ids).detach().clone().requires_grad_()      # [B, L, H]
        base_ids = torch.full_like(input_ids, baseline_id, device=device)
        baseline_embeds = emb_layer(base_ids)

        # masked positions
        if labels is not None:
            mlm_mask = (labels != -100).to(inputs_embeds.dtype)                     # [B, L]
        else:
            if tokenizer.mask_token_id is None:
                raise ValueError("Need labels or mask_token_id to derive mlm_mask.")
            mlm_mask = (input_ids == tokenizer.mask_token_id).to(inputs_embeds.dtype)

        # 2) ✅ Build valid targets (always from pooled [B,V])
        if scoring_type != "Conductance_alt":
            with torch.no_grad():
                pooled = _forward_mlm_with_embeds_factory_pooled(model)(
                    inputs_embeds, attention_mask, token_type_ids, mlm_mask
                )  # [B, V]
            if target_mode == "label" and labels is not None and (mlm_mask.sum() > 0):
                target = _majority_non_ignore_per_row(labels, ignore_index=-100)    # [B]
            else:
                target = pooled.argmax(dim=-1)                                      # [B]
        else:
            target = None  # loss-forward returns [B] scalars

        # attribution
        if scoring_type == "Conductance_alt":
            captum_out = mlc.attribute(
                inputs=(inputs_embeds,),
                baselines=(baseline_embeds,),
                target=None,
                n_steps=n_steps,
                internal_batch_size=None,
                additional_forward_args=(attention_mask, token_type_ids, labels),
                return_convergence_delta=False,
                attribute_to_layer_input=False,
                grad_kwargs={"retain_graph": False},
            )
        else:
            captum_out = mlc.attribute(
                inputs=(inputs_embeds,),
                baselines=(baseline_embeds,),
                target=target,
                n_steps=n_steps,
                internal_batch_size=None,
                additional_forward_args=(attention_mask, token_type_ids, mlm_mask),
                return_convergence_delta=False,
                attribute_to_layer_input=False,
                grad_kwargs={"retain_graph": False},
            )

        # unify list
        if isinstance(captum_out, (list, tuple)):
            captum_attrs = [t.detach() for t in captum_out]
        else:
            captum_attrs = [captum_out.detach()]

        # accumulate per-layer
        for i, score in enumerate(captum_attrs):
            print(f"Layer {i} score shape before: {tuple(score.shape)}")
            # score ~ [B, L, H]
            if mode == "masked" and score.dim() >= 2:
                # 3) ✅ Don’t mutate mlm_mask; expand a local copy
                mask_exp = mlm_mask
                while mask_exp.dim() < score.dim():
                    mask_exp = mask_exp.unsqueeze(-1)
                denom = mlm_mask.sum(dim=1, keepdim=True).clamp_min(1)          # [B,1]
                masked_sum = (score * mask_exp).sum(dim=1)                      # [B, H]
                score_reduced_batch = masked_sum / denom                        # [B, H]
                score_reduced = score_reduced_batch.sum(dim=0) if sm else score_reduced_batch.mean(dim=0)  # [H]
                
            elif i % 4 != 1 and i % 4 != 0:  # attention scores or attn probs
                # reduce batch first
                score_reduced = score.sum(dim=0) if sm else score.mean(dim=0)   # [L, H]

                # token reductions
                if mode == "cls":
                    score_reduced = score_reduced[0]                 # [H]
                elif mode == "mean":
                    score_reduced = score_reduced.mean(dim=0)        # [H]
                elif mode == "sum":
                    score_reduced = score_reduced.sum(dim=0)         # [H]
                elif mode == "topk":
                    topk_tokens = min(10, score_reduced.size(0))
                    token_scores = score_reduced.abs().sum(dim=1)    # [L]
                    idx = token_scores.topk(topk_tokens).indices
                    score_reduced = score_reduced.index_select(0, idx).mean(dim=0)  # [H]
            else:
                score_reduced = score.sum(dim=0) if sm else score.mean(dim=0)
            print(f"Layer {i} score shape after: {tuple(score_reduced.shape)}")
            if i not in new_scores:
                new_scores[i] = score_reduced.clone()
            else:
                new_scores[i] += score_reduced

        batch_count += 1

    if batch_count == 0:
        raise RuntimeError("No batches were processed.")
    for i in new_scores:
        new_scores[i] /= batch_count

    means = [s.mean() for s in new_scores.values() if s is not None]

    if normalization:
        for i in range(len(new_scores)):
            if new_scores[i] is not None:
                x = new_scores[i]
                new_scores[i] = (x - x.mean()) / (x.std(unbiased=False) + 1e-8)

    # restore grads
    for p, req in zip(model.parameters(), orig_reqs):
        p.requires_grad_(req)
    model.train()
    print("Finished calculating scores")
    return new_scores, means
