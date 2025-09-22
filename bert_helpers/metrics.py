# metrics.py
# ------------------------------------------------
# Metric helpers for MLM: top-k accuracy & PPPL.
# - mlm_compute_metrics(): used by Trainer
# - pseudo_perplexity():   used by callback
# ------------------------------------------------

from typing import Dict, Iterable, List, Tuple
import math
import numpy as np
import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel


def mlm_compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute masked-token accuracy@1 and accuracy@5 from logits and labels.
    - eval_pred.predictions: [B, T, V]
    - eval_pred.label_ids:   [B, T] with -100 for non-masked positions
    """
    logits, labels = eval_pred
    # Convert to numpy
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = logits  # [B, T, V]
    labels = labels  # [B, T]

    mask = labels != -100
    masked_total = int(mask.sum())
    if masked_total == 0:
        return {"masked_acc@1": 0.0, "masked_acc@5": 0.0, "masked_tokens": 0}

    # Top-1
    top1 = preds.argmax(axis=-1)  # [B, T]
    correct1 = (top1[mask] == labels[mask]).sum().item()
    acc1 = correct1 / masked_total

    # Top-5 membership (fast argpartition)
    k = 5
    # indices of 5 largest per position
    topk_idx = np.argpartition(preds, -k, axis=-1)[..., -k:]  # [B, T, 5]
    # gather labels for masked positions and compare
    masked_labels = labels[mask]                              # [M]
    # flatten topk at masked positions
    topk_masked = topk_idx[mask]                              # [M, 5]
    # Check membership
    correct5 = (topk_masked == masked_labels[:, None]).any(axis=-1).sum().item()
    acc5 = correct5 / masked_total

    return {
        "masked_acc@1": acc1,
        "masked_acc@5": acc5,
        "masked_tokens": float(masked_total),
        # eval_loss will be logged by Trainer as eval_loss
    }


@torch.no_grad()
def pseudo_perplexity_for_texts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: Iterable[str],
    *,
    max_texts: int = 64,
    max_len: int = 128,
    chunk_size: int = 64,
    device: torch.device,
) -> float:
    """
    Compute PPPL via pseudo-log-likelihood by masking each token once.
    - Truncates to `max_len`
    - Evaluates at most `max_texts`
    - Processes masked positions in chunks to control VRAM.

    Returns the average PPPL across texts.
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    # Special tokens to skip when scoring
    special_ids = set([tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id])

    # restrict number of texts
    if isinstance(texts, list):
        subset = texts[:max_texts]
    else:
        subset = []
        for i, t in enumerate(texts):
            if i >= max_texts:
                break
            subset.append(t)

    for text in subset:
        enc = tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_len, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)         # [1, T]
        attn = enc["attention_mask"].to(device)         # [1, T]
        T = input_ids.size(1)

        # positions to score (non-special and with attention=1)
        pos = [i for i in range(T)
               if attn[0, i].item() == 1 and int(input_ids[0, i]) not in special_ids]
        if not pos:
            continue

        # Build masked copies in chunks
        for start in range(0, len(pos), chunk_size):
            chunk = pos[start:start+chunk_size]
            # Make a batch of copies
            batch_ids = input_ids.repeat(len(chunk), 1)          # [C, T]
            labels = torch.full_like(batch_ids, -100)            # only supervise at the masked spot

            for row, j in enumerate(chunk):
                labels[row, j] = batch_ids[row, j]               # remember the original
                batch_ids[row, j] = tokenizer.mask_token_id      # mask the token

            outputs = model(input_ids=batch_ids, attention_mask=attn.repeat(len(chunk), 1))
            # logits: [C, T, V]
            logits = outputs.logits
            log_probs = torch.log_softmax(logits, dim=-1)

            rows = torch.arange(len(chunk), device=device)
            # gather -log p(original_token | masked)
            nll = -log_probs[rows, torch.tensor(chunk, device=device), labels[rows, torch.tensor(chunk, device=device)]]
            nll = nll.sum().item()

            total_nll += nll
            total_tokens += len(chunk)

    if total_tokens == 0:
        return float("nan")

    # PPPL = exp( total_nll / total_tokens )
    return math.exp(total_nll / total_tokens)
