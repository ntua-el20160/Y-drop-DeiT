# pretrain_mlm.py
# ------------------------------------------------------------
# BERT MLM pretraining on Wikitext with Y-Drop support,
# logging MLM loss, masked-token top-k accuracy, and PPPL.
# Optional: quick fine-tuning on SST-2 & MRPC after pretrain.
# ------------------------------------------------------------

import os
import argparse
from typing import Optional

import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    TrainingArguments,
)

# Your local trainer subclass
from bert_helpers.custom_trainer import MyTrainer

# Local utils
from bert_helpers.ydrop_utils import replace_dropout, collect_encoder_dropouts_and_selected_layers
from bert_helpers.callbacks import CSVLoggerCallback, PseudoPerplexityCallback
from bert_helpers.metrics import mlm_compute_metrics


# -----------------------------
# Data helpers
# -----------------------------
def load_wikitext(which: str = "wikitext-2-raw-v1") -> DatasetDict:
    """
    Load Wikitext (WT-2 or WT-103). Keeps train/validation splits.
    """
    ds = load_dataset("wikitext", which)
    return DatasetDict({"train": ds["train"], "validation": ds["validation"]})


def tokenize_and_pack(
    raw_ds: DatasetDict,
    tokenizer: BertTokenizerFast,
    block_size: int,
    num_proc: Optional[int] = None,
) -> DatasetDict:
    """
    Tokenize 'text' then pack into fixed-length blocks for *all* token-level columns
    (e.g., input_ids, attention_mask, token_type_ids, special_tokens_mask if present).
    Drops the last short remainder to avoid padding.
    """

    def tok_fn(examples):
        # Keep or drop token_type_ids depending on your preference.
        # If you don't need them, set return_token_type_ids=False to lighten the batch.
        return tokenizer(
            examples["text"],
            add_special_tokens=True,
            return_token_type_ids=True,  # set False if you prefer Option A
        )

    tokenized = raw_ds.map(
        tok_fn,
        batched=True,
        remove_columns=raw_ds["train"].column_names,
        num_proc=num_proc,
    )

    def pack_fn(examples):
        # Figure out which columns are token-level tensors we should pack.
        # (Anything that is a list-of-lists with same length as input_ids should be packed.)
        keys = [k for k in examples.keys() if k in {
            "input_ids", "attention_mask", "token_type_ids", "special_tokens_mask"
        } and k in examples]

        # Concatenate each column, then split into equal blocks.
        concatenated = {k: sum(examples[k], []) for k in keys}
        total_len = (len(concatenated["input_ids"]) // block_size) * block_size

        result = {
            k: [seq[i:i + block_size] for i in range(0, total_len, block_size)]
            for k, seq in concatenated.items()
        }
        return result

    lm = tokenized.map(pack_fn, batched=True, num_proc=num_proc)
    return lm



def make_collator(tokenizer: BertTokenizerFast, p: float, whole_word: bool):
    """
    Dynamic masking collator (per-batch).
    """
    if whole_word:
        return DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm_probability=p)
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=p)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser("BERT MLM pretraining (Wikitext) with Y-Drop + metrics logging")

    # Model/tokenizer
    ap.add_argument("--model_name", type=str, default="prajjwal1/bert-small")

    # Data
    ap.add_argument("--wikitext", type=str, default="wikitext-2-raw-v1",
                    choices=["wikitext-2-raw-v1", "wikitext-103-raw-v1"])
    ap.add_argument("--block_size", type=int, default=512)
    ap.add_argument("--preprocessing_num_workers", type=int, default=4)

    # Masking
    ap.add_argument("--mlm_probability", type=float, default=0.15)
    ap.add_argument("--whole_word_mask", action="store_true")

    # Train schedule (single phase for simplicity)
    ap.add_argument("--max_steps", type=int, default=20000)
    ap.add_argument("--epochs", type=float, default=0.0, help="If >0, trains by epochs instead of steps")

    # Batching
    ap.add_argument("--per_device_train_batch_size", type=int, default=32)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=32)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # Optim
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--adam_beta1", type=float, default=0.9)
    ap.add_argument("--adam_beta2", type=float, default=0.999)
    ap.add_argument("--adam_epsilon", type=float, default=1e-6)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--lr_scheduler_type", type=str, default="linear",
                    choices=["linear","cosine","cosine_with_restarts","polynomial","constant","constant_with_warmup"])

    # Runtime & logging
    ap.add_argument("--output_dir", type=str, default="./mlm-wikitext-run")
    ap.add_argument("--save_total_limit", type=int, default=3)
    ap.add_argument("--save_steps", type=int, default=2000)
    ap.add_argument("--eval_steps", type=int, default=1000)
    ap.add_argument("--logging_steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataloader_num_workers", type=int, default=4)
    ap.add_argument("--report_to", type=str, nargs="*", default=["none"])

    # Precision / memory
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--tf32", action="store_true")

    # Y-Drop knobs (kept similar to your original)
    ap.add_argument("--ydrop", action="store_true",default =False, help="Enable MyDropout replacement")
    ap.add_argument('--drop_rate', type=float, default=0.1)
    ap.add_argument('--elasticity', type=float, default=0.01)
    ap.add_argument('--annealing_factor', type=float, default=0.01)
    ap.add_argument('--n_steps', type=int, default=5)
    ap.add_argument('--update_batches', type=int, default=1)
    ap.add_argument('--update_freq', type=int, default=1)
    ap.add_argument('--mask_type', default='sigmoid', type=str)
    ap.add_argument('--scoring_type', choices=['Conductance', 'Conductance_alt'], default='Conductance')
    ap.add_argument('--stats', action='store_true', default=False)
    ap.add_argument('--baseline', type=str, default='mask', choices=['mask', 'pad'])
    ap.add_argument('--target_mode', type=str, default='pred', choices=['pred', 'label'])
    ap.add_argument('--mode', type=str, default='masked', choices=[None, 'masked', 'cls', 'mean', 'sum', 'topk'])
    ap.add_argument('--epoch_gap', type=int, default=1)
    ap.add_argument('--conductance_batch_size', type=int, default=None)
    ap.add_argument('--switch_epoch', type=int, default=None)

    # PPPL eval config
    ap.add_argument("--pppl_num_texts", type=int, default=64, help="How many validation texts to score for PPPL")
    ap.add_argument("--pppl_max_len", type=int, default=128, help="Max tokens per text for PPPL")
    ap.add_argument("--pppl_chunk_size", type=int, default=64, help="Mask-positions per forward chunk (controls VRAM)")
    ap.add_argument("--pppl_log_key", type=str, default="eval_pppl", help="Key under which PPPL is logged")

    # Optional: run quick fine-tuning at the end
    ap.add_argument("--run_finetune", action="store_true")
    ap.add_argument("--finetune_max_steps", type=int, default=1000, help="Steps per downstream task for a quick check")

    args = ap.parse_args()

    # Perf toggles
    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # 1) Model & tokenizer
    # -----------------------------
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    model = BertForMaskedLM.from_pretrained(args.model_name)
    model.config._attn_implementation = "eager"
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Optional Y-Drop swap
    drop_list = selected_layers = None
    if args.ydrop:
        replace_dropout(
            model,
            drop_rate=args.drop_rate,
            elasticity=args.elasticity,
            mask_type=args.mask_type,
            transformer_mean=True,
        )
        drop_list, selected_layers = collect_encoder_dropouts_and_selected_layers(model)

    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    print(model)

    # -----------------------------
    # 2) Data
    # -----------------------------
    raw = load_wikitext(args.wikitext)
    # Keep validation texts for PPPL
    val_texts = raw["validation"]["text"]

    lm_ds = tokenize_and_pack(
        raw_ds=raw,
        tokenizer=tokenizer,
        block_size=args.block_size,
        num_proc=args.preprocessing_num_workers,
    )

    collator = make_collator(tokenizer, args.mlm_probability, args.whole_word_mask)

    # -----------------------------
    # 3) TrainingArguments & Trainer
    # -----------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    common = dict(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=args.report_to,
        ddp_find_unused_parameters=False,
        output_dir=args.output_dir,
    )

    if args.epochs and args.epochs > 0:
        targs = TrainingArguments(num_train_epochs=args.epochs, **common)
    else:
        targs = TrainingArguments(max_steps=args.max_steps, **common)

    trainer = MyTrainer(
        model=model,
        args=targs,
        train_dataset=lm_ds["train"],
        eval_dataset=lm_ds.get("validation", None),
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=mlm_compute_metrics,  # top-1/top-5 masked-token accuracy
    )

    # Attach Y-Drop knobs for your subclass logic
    trainer.ydrop = args.ydrop
    trainer.annealing_factor = args.annealing_factor
    trainer.switch_epoch = args.switch_epoch
    trainer.update_freq = args.update_freq
    trainer.update_batches = args.update_batches
    trainer.scoring_type = args.scoring_type
    trainer.mask_type = args.mask_type
    trainer.conductance_batch_size = args.conductance_batch_size
    trainer.epoch_gap = args.epoch_gap
    trainer.baseline = args.baseline
    trainer.target_mode = args.target_mode
    trainer.n_steps = args.n_steps
    trainer.mode = args.mode
    trainer.drop_list = drop_list
    trainer.selected_layers = selected_layers

    # -----------------------------
    # 4) Callbacks: CSV log + PPPL on eval
    # -----------------------------
    csv_cb = CSVLoggerCallback(csv_path=os.path.join(args.output_dir, "metrics.csv"))
    pppl_cb = PseudoPerplexityCallback(
        tokenizer=tokenizer,
        texts=val_texts,
        max_texts=args.pppl_num_texts,
        max_len=args.pppl_max_len,
        chunk_size=args.pppl_chunk_size,
        log_key=args.pppl_log_key,
    )
    trainer.add_callback(csv_cb)
    trainer.add_callback(pppl_cb)

    # -----------------------------
    # 5) Train & save
    # -----------------------------
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "checkpoint-final"))

    # -----------------------------
    # 6) Optional: quick fine-tuning
    # -----------------------------
    if args.run_finetune:
        from finetune_tasks import run_quick_glue_finetuning
        run_quick_glue_finetuning(
            base_model_path=os.path.join(args.output_dir, "checkpoint-final"),
            out_dir=os.path.join(args.output_dir, "finetune"),
            max_steps=args.finetune_max_steps,
            per_device_batch_size=32,
            seed=args.seed,
            fp16=args.fp16,
            bf16=args.bf16,
            report_to=args.report_to,
        )


if __name__ == "__main__":
    main()
