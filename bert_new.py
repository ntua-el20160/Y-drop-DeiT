#!/usr/bin/env python
"""
Train BERT (base or smaller variants) for Masked Language Modeling on a local text dataset.

Highlights
- Supports popular BERT checkpoints out-of-the-box (bert-base-uncased + prajjwal1/* tiny/mini/small/medium).
- Simple CLI to choose model, dataset path, and key hyperparameters.
- Defaults mirror *original BERT pretraining* where sensible for small-scale runs:
  - MLM probability: 0.15
  - Max sequence length: 512 (note: original BERT did curriculum 128 then 512; we keep 512 by default)
  - Optimizer: AdamW with lr=1e-4, weight_decay=0.01, warmup_ratio=0.01
  - Batch sizes set for single-GPU practicality (override as needed)
- Computes dataset stats (files, bytes, characters, tokens) before training.
- Evaluates validation perplexity.

Usage examples
--------------
# Train bert-base-uncased on your wikipedia-20percent directory
python train_bert_mlm.py \
  --dataset_path ./datasets/wikipedia-20percent \
  --model_name bert-base-uncased \
  --output_dir ./runs/bert-base-wiki20p

# Train a smaller model (good for quick experiments)
python train_bert_mlm.py \
  --dataset_path ./datasets/tinystories \
  --model_name prajjwal1/bert-small \
  --per_device_train_batch_size 32 \
  --num_train_epochs 3

Notes on the dataset folder
---------------------------
- The script will recursively read **all .txt files** under --dataset_path.
- If your dataset is a single text file, that's fine too.
- A small validation split is created from the training texts using --validation_split.
"""

from __future__ import annotations
from datasets import load_from_disk, Dataset, concatenate_datasets
import argparse
import os
import sys
import math
import glob
from pathlib import Path
from typing import List, Dict
from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast
import datasets
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
from bert_helpers.final_trainer import HookedTrainer
import torch
import torch.nn.functional as F
from types import MethodType
from updated_transformer.dynamic_dropout import MyDropout
def mlm_crit_for(self,
                 inputs_embeds: torch.Tensor,
                 attention_mask: torch.Tensor | None = None,
                 labels: torch.Tensor | None = None,
                 token_type_ids: torch.Tensor | None = None,
                 **forward_kwargs) -> torch.Tensor:
    """
    Per-sample MLM loss for BERT-like masked LM heads.
    Returns a [B] tensor. Keeps gradients.

    Args:
      input_ids: [B, T]
      attention_mask: [B, T] (optional)
      labels: [B, T] with -100 on non-masked tokens (required)
      **forward_kwargs: anything else your model.forward accepts (e.g., token_type_ids)

    Notes:
      - We *don’t* pass labels into forward(), so the model won't compute/average loss internally.
    """
    if labels is None:
        raise ValueError("mlm_crit_for requires `labels` shaped [B, T] with -100 on non-masked positions.")

    outputs = self(inputs_embeds=inputs_embeds, attention_mask=attention_mask, token_type_ids=token_type_ids, **forward_kwargs)
    logits = outputs.logits  # [B, T, V]
    # print("logits shape in mlm crit_for:", logits.shape)

    B, T, V = logits.shape
    # Per-token CE with ignore_index so non-masked tokens contribute 0 after masking below
    per_token_loss = F.cross_entropy(
        logits.view(-1, V),           # [B*T, V]
        labels.view(-1),              # [B*T]
        ignore_index=-100,
        reduction="none",
    ).view(B, T)                      # [B, T]

    valid = (labels != -100)          # [B, T] masked positions
    # Sum over masked tokens, then normalize by how many masked tokens each sample had
    denom = valid.sum(dim=1).clamp_min(1)          # [B]
    per_sample = (per_token_loss * valid).sum(dim=1) / denom

    # If a sample had 0 masked tokens (shouldn't happen with proper collator), its loss is 0.4
    # print("loss per sample in mlm crit_for shape and values:", per_sample.shape, per_sample)
    return per_sample  # [B]
def pred_scalar_forward(model, ids, attn=None, lbls=None, tok_types=None):
                # No torch.no_grad(): we want gradients through the model.
                out = model(input_ids=ids,
                            attention_mask=attn,
                            token_type_ids=tok_types)
                logit = out.logits  # [B,T,V]
                B, T, V = logit.shape

                # masked positions: labels != -100 if labels provided, else use attn padding to avoid pads
                if lbls is not None:
                    mask_pos = (lbls != -100)                 # [B,T]
                else:
                    mask_pos = attn.bool() if attn is not None else torch.ones_like(ids, dtype=torch.bool)

                # choose class indices per token
                if (lbls is not None):
                    cls_ids = torch.clamp(lbls, min=0)        # [-100 -> 0] won't be used where mask_pos=False
                else:
                    # predicted class per token
                    cls_ids = logit.argmax(dim=-1)            # [B,T]

                # gather logits at (token, class)
                # idx for gather
                idx_b = torch.arange(B, device=ids.device).unsqueeze(1).expand(B, T)  # [B,T]
                gathered = logit[idx_b, torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T), cls_ids]  # [B,T]

                # zero out non-masked positions, then mean over masked tokens per sample
                gathered = gathered * mask_pos  # [B,T]
                denom = mask_pos.sum(dim=1).clamp_min(1)      # [B]
                score = gathered.sum(dim=1) / denom           # [B] scalar per sample
                return score

def find_text_files(root: str | Path) -> List[str]:
    root = Path(root)
    # Match common text file patterns
    patterns = ["**/*.txt", "**/*.text", "**/*.md"]
    files: List[str] = []
    for p in patterns:
        files.extend([str(f) for f in root.glob(p)])
    # If no matches but a file was passed, still allow it
    if not files and root.is_file():
        files = [str(root)]
    if not files:
        raise FileNotFoundError(
            f"No text files found under: {root}. Expected .txt/.text/.md files."
        )
    return sorted(files)


def load_text_dataset(dataset_path: str, validation_split: float = 0.01) -> DatasetDict:
    files = find_text_files(dataset_path)
    data_files = {"train": files}
    raw = load_dataset("text", data_files=data_files)
    # Create a deterministic train/validation split
    if validation_split and 0.0 < validation_split < 1.0:
        raw = raw["train"].train_test_split(test_size=validation_split, seed=42)
        ds = DatasetDict(train=raw["train"], validation=raw["test"])  # type: ignore
    else:
        ds = DatasetDict(train=raw["train"])  # type: ignore
    return ds



def compute_dataset_stats(tokenizer, ds, max_samples: int = 50_000):
    # pick a training split robustly
    if isinstance(ds, DatasetDict):
        split_name = "train" if "train" in ds else next(iter(ds.keys()))
        train_split = ds[split_name]

    n = min(len(train_split), max_samples)
    sample = train_split.shuffle(seed=42).select(range(n))

    if "text" in sample.column_names:
        print("Text column found")
        texts = sample["text"]
        texts = [x if isinstance(x, str) else str(x) for x in texts]
        total_chars = sum(len(t) for t in texts)
        enc = tokenizer(texts, add_special_tokens=False)
        total_tokens = sum(len(ids) for ids in enc["input_ids"])
    elif "input_ids" in sample.column_names:
        print("Input IDs column found")
        # already tokenized dataset
        total_tokens = sum(len(x) for x in sample["input_ids"])
        total_chars = 0  # unknown; skip
    else:
        raise ValueError(f"Expected 'text' or 'input_ids' columns, found {sample.column_names}")

    avg_tokens = total_tokens / max(1, n)
    return {
        "n_train_rows": len(train_split),
        "est_total_chars": total_chars,     # 0 if already tokenized
        "est_total_tokens": int(avg_tokens * len(train_split)),
        "avg_chars_per_row": (total_chars / max(1, n)) if total_chars else 0,
        "avg_tokens_per_row": avg_tokens,
    }

def load_any_dataset(path, validation_split=0.01):
    # Case A: saved HF dataset
    if os.path.exists(os.path.join(path, "dataset_info.json")):
        return load_from_disk(path)

    # Case B: arrow shards only
    split_dirs = ["train", "validation"]
    if all(os.path.isdir(os.path.join(path, s)) for s in split_dirs):
        def load_arrow_split(split_dir):
            files = sorted(glob.glob(os.path.join(split_dir, "data-*.arrow")))
            parts = [Dataset.from_file(f) for f in files]
            return concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        return DatasetDict(
            train=load_arrow_split(os.path.join(path, "train")),
            validation=load_arrow_split(os.path.join(path, "validation")),
        )

    # Fallback: raw text folder
    return load_text_dataset(path, validation_split=validation_split)

def tokenize_function(examples, tokenizer, max_seq_length: int):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_seq_length,
        padding=False,
        return_special_tokens_mask=False,
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BERT Masked LM pretraining on local text datasets")

    # Paths & naming
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to folder or file with raw .txt files")
    parser.add_argument("--output_dir", type=str, default="./runs/bert-mlm", help="Where to save checkpoints and logs")

    # Model selection
    # parser.add_argument(
    #     "--model_name",
    #     type=str,
    #     default="bert-base-uncased",
    #     help=(
    #         "HF model name or path. Common options: "
    #         "'bert-base-uncased' (default), 'bert-base-cased', "
    #         "'prajjwal1/bert-tiny', 'prajjwal1/bert-mini', 'prajjwal1/bert-small', 'prajjwal1/bert-medium'"
    #     ),
    # )

    # Tokenization / sequence settings (BERT defaults)
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max sequence length (original BERT used up to 512)")
    parser.add_argument("--mlm_probability", type=float, default=0.15, help="Masking probability (default BERT: 0.15)")
    
    parser.add_argument("--tokenizer_dir", type=str, required=True,
                        help="Local dir with tokenizer files (vocab.txt/tokenizer.json). No internet used.")
    parser.add_argument("--arch_size", type=str, default="mini",
                        choices=["tiny","mini","small","medium","base","potam"],
                        help="Model architecture size (random init).")
    
    parser.add_argument("--validation_split", type=float, default=0.01, help="Fraction for validation split from train texts")

    # Training hyperparameters (pretraining-style defaults)
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="AdamW learning rate (BERT pretraining used 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay, default BERT used 0.01")
    parser.add_argument("--warmup_ratio", type=float, default=0.01, help="Warmup ratio of total steps (approx BERT-scale warmup)")
    parser.add_argument("--num_train_epochs", type=float, default=3.0, help="Number of epochs to train")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Batch size per device (adjust for your GPU)")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32, help="Eval batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Optional cap on train rows for quick runs")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Optional cap on eval rows for quick runs")

    # Runtime
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training if supported")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Evaluate every N steps (if >0 and logging strategy=steps)")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every N steps (strategy=steps)")
    parser.add_argument(
        "--logging_steps", type=int, default=100, help="Log training metrics every N steps (strategy=steps)"
    )
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, help="Dropout probability for hidden layers")
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        help="Reporting integration: 'none', 'tensorboard', 'wandb', etc.",
    )
    ##Y-drop related args can be added here if needed
    parser.add_argument("--ydrop", action="store_true", help="Use Y-drop training",default=False)
    parser.add_argument("--no-ydrop", action="store_false", dest="ydrop", help="Do not use Y-drop training")
    parser.add_argument("--update_batches", type=int, default=1, help="Number of batches to use for mask update",)
    parser.add_argument("--n_steps", type=int, default=4, help="Conductance calcillation linear steps",)
    parser.add_argument("--update_samples", type=int, default=4, help="Frequency of updating dropout masks",)
    parser.add_argument("--update_freq", type=int, default=1, help="Frequency of updating dropout masks",)
    parser.add_argument("--mask_type", type=str, default="rank_loss_inverse", help="Type of masks to update",)
    parser.add_argument("--scoring_type", type=str, default="Conductance_alt", help="Score for mask updates",)
    parser.add_argument("--after_norm", type=bool, default=False, help="Calculate_score_after_normalization",)
    parser.add_argument("--mode", type=str, default="mean", help="How to handle token conductance",)
    parser.add_argument("--baseline", type=str, default=None, help="Baseline for conductance calculation",)
    parser.add_argument("--annealing_factor", type=float, default=0.1, help="Fraction of training before starting Y-drop",)
    parser.add_argument("--after_relu", type=bool, default=False, help="Calculate score after ReLU",)
    parser.add_argument("--elasticity", type=float, default=0.01, help="Elasticity for MyDropout",)
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    #datasets.disable_caching()   # not set_caching_enabled


# A writable place for any explicit cache files we pass below
    CACHE_DIR = os.environ.get("HF_DATASETS_CACHE", "/tmp")
    os.makedirs(CACHE_DIR, exist_ok=True)
    set_seed(args.seed)

    # Load tokenizer and model
    print(f"\nLoading model: {args.arch_size}")
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_dir)

    if tokenizer.pad_token is None:
        # For some miniature checkpoints, pad token may not be set
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token
    def print_model_tree(model):
        """Prints the full submodule hierarchy as a tree."""
        root_name = model.__class__.__name__
        print(f"\nModel tree: {root_name}")
        def _recur(mod, prefix=""):
            children = list(mod.named_children())
            for i, (name, child) in enumerate(children):
                is_last = (i == len(children) - 1)
                branch = "└─" if is_last else "├─"
                print(f"{prefix}{branch} {name}: {child.__class__.__name__}")
                _recur(child, prefix + ("   " if is_last else "│  "))
        _recur(model)
    size = args.arch_size  # "tiny" | "mini" | "small" | "medium" | "base"

    cfgs = {
        "potam" :   dict(hidden_size=128,  num_hidden_layers=8,  num_attention_heads=4,  intermediate_size=512),
        "tiny":   dict(hidden_size=128,  num_hidden_layers=2,  num_attention_heads=2,  intermediate_size=512),
        "mini":   dict(hidden_size=256,  num_hidden_layers=4,  num_attention_heads=4,  intermediate_size=1024),
        "small":  dict(hidden_size=512,  num_hidden_layers=4,  num_attention_heads=8,  intermediate_size=2048),
        "medium": dict(hidden_size=512,  num_hidden_layers=8,  num_attention_heads=8,  intermediate_size=2048),
        "base":   dict(hidden_size=768,  num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072),
    }
    print("Vocab size:", len(tokenizer))
    config = BertConfig(
        vocab_size=len(tokenizer),              # IMPORTANT: match your local tokenizer
        max_position_embeddings=args.max_seq_length + 2,  # +2 for [CLS]/[SEP] safety
        type_vocab_size=2,
        pad_token_id=tokenizer.pad_token_id or 0,
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.hidden_dropout_prob,
        **cfgs[size],
    )

    model = BertForMaskedLM(config)
    if args.ydrop:
        selected_layers = []
        drop_list = []
        print("Applying Y-drop settings to the model")
        print("Total number of layers:", len(model.bert.encoder.layer))
        for  layer in model.bert.encoder.layer:
            layer.attention.output.dropout = MyDropout(elasticity=args.elasticity, p=args.hidden_dropout_prob, tied_layer=None, mask_type=args.mask_type, scaler=1.0,
                                transformer_mean=True,rescaling_type="linear")
            layer.output.dropout = MyDropout(elasticity=args.elasticity, p=args.hidden_dropout_prob, tied_layer=None, mask_type=args.mask_type, scaler=1.0,
                                transformer_mean=True,rescaling_type="linear")

            if args.after_norm:
                selected_layers.append(layer.attention.output.LayerNorm)
                selected_layers.append(layer.output.LayerNorm)
            else:
                selected_layers.append(layer.attention.output.dense)
                selected_layers.append(layer.output.dense)

            drop_list.append(layer.attention.output.dropout)
            drop_list.append(layer.output.dropout)
        model.selected_layers = selected_layers
        print("Number of selected layers for Y-drop:", len(model.selected_layers))
        model.drop_list = drop_list
        print("Number of dropout layers for Y-drop:", len(model.drop_list))
        for drop in drop_list:
            drop.use_normal_dropout()
            
        model.crit_for = MethodType(mlm_crit_for, model)
        model.pred_scalar_forward = MethodType(pred_scalar_forward, model)
    print_model_tree(model)

    # Load dataset from local text files
    print(f"\nLoading dataset from: {args.dataset_path}")
    # ds = load_any_dataset(args.dataset_path, validation_split=args.validation_split)

    # if isinstance(ds, Dataset):
    #     print("Single dataset split found, assigning to 'train'")
    #     ds = DatasetDict(train=ds)
    train = load_from_disk(os.path.join(args.dataset_path, "train"))
    validation = load_from_disk(os.path.join(args.dataset_path, "validation"))
    ds = DatasetDict(train=train, validation=validation)


# If no 'validation', create a small split
    # if "validation" not in ds:
    #     print(f"No 'validation' split found. Creating a small validation split from 'train' ({args.validation_split*100:.1f}%)")
    #     n_test = max(1, int(0.01 * len(ds["train"])))
    #     tmp = ds["train"].train_test_split(
    #         test_size=n_test,
    #         seed=42,
    #         # write index maps in a path you can write to
    #         train_indices_cache_file_name=os.path.join(CACHE_DIR, "train_idx.arrow"),
    #         test_indices_cache_file_name=os.path.join(CACHE_DIR, "val_idx.arrow"),
    #     )
    #     ds = DatasetDict(train=tmp["train"], validation=tmp["test"])



    # Show quick stats
    # stats = compute_dataset_stats(tokenizer, ds)
    # print("\nDataset stats (estimated):")
    # for k, v in stats.items():
    #     print(f"  - {k}: {v}")

    # Tokenize
    def _tok(batch):
        return tokenize_function(batch, tokenizer, args.max_seq_length)

    train_cols  = ds["train"].column_names
    needs_tok = ("text" in train_cols) and ("input_ids" not in train_cols)
    if needs_tok:
        tokenized = ds.map(_tok, batched=True, num_proc=os.cpu_count() or 1, remove_columns=train_cols, load_from_cache_file=False, keep_in_memory=True)
    else:
        # already has input_ids/attention_mask
        tokenized = ds

    if "validation" in tokenized:
        eval_dataset = tokenized["validation"]
        if args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(min(len(eval_dataset), args.max_eval_samples)))
    else:
        eval_dataset = None

    train_dataset = tokenized["train"]
    if args.max_train_samples:
        train_dataset = train_dataset.select(range(min(len(train_dataset), args.max_train_samples)))

    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    # Training arguments
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
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        report_to=None if args.report_to == "none" else [args.report_to],
        save_total_limit=2,
    )

    # Trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    # )
    trainer = HookedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    if args.ydrop:
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

    # Evaluate
    if eval_dataset is not None:
        metrics = trainer.evaluate()
        if "eval_loss" in metrics:
            try:
                metrics["perplexity"] = float(math.exp(metrics["eval_loss"]))
            except OverflowError:
                metrics["perplexity"] = float("inf")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        print("\nValidation metrics:")
        for k, v in metrics.items():
            print(f"  - {k}: {v}")

    print("\nDone.")


if __name__ == "__main__":
    main()
