#!/usr/bin/env python
"""
Finetune MLM-pretrained BERT(-like) checkpoints on GLUE with optional Y-drop.

Features
- Loads encoder weights from a local MLM-pretrained checkpoint (BertForMaskedLM or similar)
  and instantiates a SequenceClassification head with the right num_labels.
- Optional Y-drop: replaces dropout layers in the encoder with MyDropout and passes knobs to HookedTrainer if present.
- Handles all GLUE tasks, including MNLI (matched/mismatched) and STS-B (regression).
- Uses evaluate/glue metrics. Saves metrics and model.

Example
-------
# SST-2 (accuracy)
python finetune_glue_y_drop.py \
  --model_path ./runs/bert-mini-mlm/checkpoint-5000 \
  --task_name sst2 \
  --output_dir ./runs/ft-sst2-mini \
  --ydrop --elasticity 0.01 --hidden_dropout_prob 0.1

# MNLI (reports matched + mismatched)
python finetune_glue_y_drop.py \
  --model_path ./runs/bert-base-mlm \
  --task_name mnli \
  --output_dir ./runs/ft-mnli-base \
  --num_train_epochs 3 --per_device_train_batch_size 32

# STS-B (regression: Pearson/Spearman)
python finetune_glue_y_drop.py \
  --model_path ./runs/bert-mini-mlm \
  --task_name stsb \
  --output_dir ./runs/ft-stsb-mini
"""
from __future__ import annotations
import argparse
import os
import math
from typing import Tuple, Optional, Dict, Any

import torch
from datasets import load_dataset
import evaluate

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)

# Optional imports (your custom modules)
_HOOKED_TRAINER_AVAILABLE = False
try:
    from bert_helpers.final_trainer import HookedTrainer as MaybeHookedTrainer
    _HOOKED_TRAINER_AVAILABLE = True
except Exception:
    MaybeHookedTrainer = Trainer

_MYDROPOUT_AVAILABLE = False
try:
    from updated_transformer.dynamic_dropout import MyDropout
    _MYDROPOUT_AVAILABLE = True
except Exception:
    MyDropout = None  # type: ignore


GLUE_SENTS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

GLUE_NUM_LABELS = {
    "cola": 2,
    "sst2": 2,
    "mrpc": 2,
    "qqp": 2,
    "stsb": 1,   # regression
    "mnli": 3,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
}

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Finetune GLUE with optional Y-drop")
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to your MLM-pretrained checkpoint dir (local).")
    p.add_argument("--tokenizer_dir", type=str, default=None,
                   help="Optional local tokenizer dir. Defaults to model_path if omitted.")
    p.add_argument("--task_name", type=str, required=True, choices=list(GLUE_SENTS.keys()))
    p.add_argument("--output_dir", type=str, required=True)

    # Tokenization
    p.add_argument("--max_seq_length", type=int, default=256)

    # Train/eval
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=32)
    p.add_argument("--per_device_eval_batch_size", type=int, default=128)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--eval_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    p.add_argument("--save_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    p.add_argument("--eval_steps", type=int, default=None)
    p.add_argument("--save_steps", type=int, default=None)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--metric_for_best_model", type=str, default=None,
                   help="If set, enables load_best_model_at_end. E.g. 'accuracy', 'matthews_correlation', 'pearson'.")
    p.add_argument("--greater_is_better", action="store_true", help="Interpret metric_for_best_model as larger-is-better.")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report_to", type=str, default="none")

    # Y-drop controls
    p.add_argument("--ydrop", action="store_true", help="Enable Y-drop.")
    p.add_argument("--hidden_dropout_prob", type=float, default=0.1, help="Dropout probability for encoder modules")
    p.add_argument("--elasticity", type=float, default=0.01)
    p.add_argument("--mask_type", type=str, default="rank_loss_inverse")
    p.add_argument("--rescaling_type", type=str, default="linear")
    p.add_argument("--after_norm", action="store_true", help="Select LayerNorms instead of Dense for any hook logic.")
    p.add_argument("--after_relu", action="store_true")
    p.add_argument("--annealing_factor", type=float, default=0.1)
    p.add_argument("--n_steps", type=int, default=4)
    p.add_argument("--update_batches", type=int, default=1)
    p.add_argument("--update_samples", type=int, default=4)
    p.add_argument("--update_freq", type=int, default=1)
    p.add_argument("--scoring_type", type=str, default="Conductance_alt")
    p.add_argument("--mode", type=str, default="mean")
    p.add_argument("--baseline", type=str, default=None)
    return p


def _maybe_apply_ydrop_to_bert(model, p_dropout: float, elasticity: float,
                               mask_type: str, rescaling_type: str,
                               collect_after_norm: bool) -> Dict[str, Any]:
    """
    Replace encoder dropouts with MyDropout for BERT-like models.
    Returns a dict with selected_layers and drop_list (useful if your HookedTrainer expects them).
    """
    info = {"selected_layers": [], "drop_list": []}
    if not _MYDROPOUT_AVAILABLE:
        print("[WARN] Y-drop requested but MyDropout not importable. Skipping replacement.")
        return info

    # Find base model holding the encoder
    base = None
    if hasattr(model, "bert"):
        base = model.bert
    else:
        print("[WARN] Could not find .bert or .roberta on model; skipping Y-drop.")
        return info

    encoder = getattr(base, "encoder", None)
    if encoder is None or not hasattr(encoder, "layer"):
        print("[WARN] No encoder.layer found; skipping Y-drop.")
        return info

    print("[Y-drop] Replacing dropout modules in the encoder...")
    for layer in encoder.layer:
        # Replace the two standard BERT dropouts:
        layer.attention.output.dropout = MyDropout(
            elasticity=elasticity, p=p_dropout, tied_layer=None, mask_type=mask_type,
            scaler=1.0, transformer_mean=True, rescaling_type=rescaling_type
        )
        layer.output.dropout = MyDropout(
            elasticity=elasticity, p=p_dropout, tied_layer=None, mask_type=mask_type,
            scaler=1.0, transformer_mean=True, rescaling_type=rescaling_type
        )
        # Optionally collect modules for scoring hooks
        if collect_after_norm:
            info["selected_layers"].append(layer.attention.output.LayerNorm)
            info["selected_layers"].append(layer.output.LayerNorm)
        else:
            info["selected_layers"].append(layer.attention.output.dense)
            info["selected_layers"].append(layer.output.dense)

        info["drop_list"].append(layer.attention.output.dropout)
        info["drop_list"].append(layer.output.dropout)

    # put dropouts in normal mode by default
    for d in info["drop_list"]:
        d.use_normal_dropout()

    print(f"[Y-drop] Selected modules: {len(info['selected_layers'])}, Dropout modules: {len(info['drop_list'])}")
    return info


def get_task_keys(task_name: str) -> Tuple[str, Optional[str]]:
    return GLUE_SENTS[task_name]


def build_compute_metrics(task_name: str):
    glue_metric = evaluate.load("glue", task_name)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if task_name == "stsb":
            # regression: model outputs shape [B,1]; convert to float
            preds = preds.reshape(-1)
        else:
            preds = preds.argmax(axis=-1)
        result = glue_metric.compute(predictions=preds, references=labels)
        # For convenience, add composite for STS-B and the common ACC/F1 pair tasks
        if task_name in {"mrpc", "qqp"} and "f1" in result and "accuracy" in result:
            result["acc_f1_mean"] = (result["accuracy"] + result["f1"]) / 2.0
        if task_name == "stsb" and "pearson" in result and "spearmanr" in result:
            result["pearson_spearman_mean"] = (result["pearson"] + result["spearmanr"]) / 2.0
        return result

    return compute_metrics


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)

    task = args.task_name.lower()
    sentence1_key, sentence2_key = get_task_keys(task)
    num_labels = GLUE_NUM_LABELS[task]
    is_regression = (task == "stsb")

    # Config
    config = AutoConfig.from_pretrained(args.model_path)
    config.num_labels = num_labels
    if is_regression:
        config.problem_type = "regression"

    # Tokenizer
    tok_src = args.tokenizer_dir if args.tokenizer_dir else args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    if tokenizer.pad_token is None:
        # safer training
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token

    # Model: load from MLM-pretrained checkpoint but as SequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        config=config,
        ignore_mismatched_sizes=True,  # classifier head will be freshly initialized
    )

    # Optional Y-drop
    ydrop_info = {"selected_layers": [], "drop_list": []}
    if args.ydrop:
        ydrop_info = _maybe_apply_ydrop_to_bert(
            model,
            p_dropout=args.hidden_dropout_prob,
            elasticity=args.elasticity,
            mask_type=args.mask_type,
            rescaling_type=args.rescaling_type,
            collect_after_norm=args.after_norm,
        )
        # If your HookedTrainer expects these fields on the model, attach them:
        if len(ydrop_info["selected_layers"]) > 0:
            model.selected_layers = ydrop_info["selected_layers"]  # type: ignore[attr-defined]
        if len(ydrop_info["drop_list"]) > 0:
            model.drop_list = ydrop_info["drop_list"]  # type: ignore[attr-defined]

    # Data
    raw = load_dataset("glue", task)
    def preprocess(batch):
        if sentence2_key is None:
            return tokenizer(batch[sentence1_key], max_length=args.max_seq_length, truncation=True)
        return tokenizer(batch[sentence1_key], batch[sentence2_key], max_length=args.max_seq_length, truncation=True)

    encoded = raw.map(preprocess, batched=True, remove_columns=raw["train"].column_names)

    # For MNLI, evaluate matched/mismatched separately
    eval_names_splits = [("validation", "validation")]
    if task == "mnli":
        eval_names_splits = [("validation_matched", "validation_matched"),
                             ("validation_mismatched", "validation_mismatched")]

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Metrics
    compute_metrics = build_compute_metrics(task)

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        evaluation_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=None if args.report_to == "none" else [args.report_to],
        load_best_model_at_end=bool(args.metric_for_best_model),
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better if args.metric_for_best_model else None,
    )

    # Pick trainer class
    TrainerCls = MaybeHookedTrainer if _HOOKED_TRAINER_AVAILABLE else Trainer

    trainer = TrainerCls(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded[eval_names_splits[0][1]],  # primary eval split
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # If HookedTrainer + Y-drop knobs are expected, attach them
    if _HOOKED_TRAINER_AVAILABLE and args.ydrop:
        trainer.ydrop = True
        trainer.n_steps = args.n_steps
        trainer.update_samples = args.update_samples
        trainer.update_freq = args.update_freq
        trainer.mask_type = args.mask_type
        trainer.scoring_type = args.scoring_type
        trainer.after_norm = args.after_norm
        trainer.mode = args.mode
        trainer.baseline = args.baseline
        trainer.annealing_factor = args.annealing_factor
        trainer.update_batches = args.update_batches

    # Train
    train_result = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    # Evaluate (primary)
    print("\nPrimary validation evaluation:")
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # Evaluate MNLI mismatched if present
    if task == "mnli" and "validation_mismatched" in encoded:
        print("\nMNLI mismatched evaluation:")
        mm_metrics = trainer.evaluate(eval_dataset=encoded["validation_mismatched"])
        # Prefix keys to keep both
        mm_metrics = {f"mm_{k}": v for k, v in mm_metrics.items()}
        trainer.log_metrics("eval_mm", mm_metrics)
        trainer.save_metrics("eval_mm", mm_metrics)

    # (Optional) pretty print a useful scalar like perplexity—NOT applicable here (classification/regression),
    # but you can log loss values that trainer already provides.
    print("\nDone.")

if __name__ == "__main__":
    main()
