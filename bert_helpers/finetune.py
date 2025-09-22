# finetune_tasks.py
# --------------------------------------------------------
# Quick fine-tuning on two simple GLUE tasks:
#  - SST-2 (sentiment)
#  - MRPC (paraphrase)
# Loads a fresh head and initializes from your pretrained base.
# --------------------------------------------------------

import os
from typing import Dict, Tuple

from datasets import load_dataset
import numpy as np
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score


def glue_tokenize(tokenizer, examples, sentence_keys):
    """
    Tokenize with truncation for a GLUE example.
    sentence_keys: ("sentence",) for SST-2 or ("sentence1","sentence2") for MRPC.
    """
    if len(sentence_keys) == 1:
        return tokenizer(examples[sentence_keys[0]], truncation=True)
    a, b = sentence_keys
    return tokenizer(examples[a], examples[b], truncation=True)


def glue_compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds) if len(set(labels)) == 2 else 0.0
    return {"accuracy": acc, "f1": f1}


def run_quick_glue_finetuning(
    base_model_path: str,
    out_dir: str,
    *,
    max_steps: int = 1000,
    per_device_batch_size: int = 32,
    seed: int = 42,
    fp16: bool = False,
    bf16: bool = False,
    report_to = ("none",),
):
    """
    Fine-tune SST-2 and MRPC with small budgets to sanity-check pretraining.
    """
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = BertTokenizerFast.from_pretrained(base_model_path)

    tasks: Dict[str, Tuple[str, Tuple[str, ...]]] = {
        "sst2": ("glue", ("sentence",)),
        "mrpc": ("glue", ("sentence1", "sentence2")),
    }

    for task, (hf_name, fields) in tasks.items():
        print(f"\n=== Fine-tuning on {task.upper()} ===")
        ds = load_dataset(hf_name, task)
        num_labels = 2  # both SST-2 and MRPC are binary

        model = BertForSequenceClassification.from_pretrained(base_model_path, num_labels=num_labels)

        tokenized = ds.map(
            lambda ex: glue_tokenize(tokenizer, ex, fields),
            batched=True,
            remove_columns=ds["train"].column_names,
        )

        targs = TrainingArguments(
            output_dir=os.path.join(out_dir, task),
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_batch_size,
            max_steps=max_steps,
            evaluation_strategy="steps",
            eval_steps=max(50, max_steps // 10),
            logging_steps=50,
            save_steps=max_steps,  # save only at the end
            load_best_model_at_end=False,
            learning_rate=2e-5,
            weight_decay=0.0,
            report_to=list(report_to),
            seed=seed,
            fp16=fp16,
            bf16=bf16,
        )

        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"] if "validation" in tokenized else tokenized["validation_matched"],
            tokenizer=tokenizer,
            compute_metrics=glue_compute_metrics,
        )

        trainer.train()
        metrics = trainer.evaluate()
        print(f"{task.upper()} metrics:", metrics)
        trainer.save_model(os.path.join(out_dir, task, "checkpoint-final"))
