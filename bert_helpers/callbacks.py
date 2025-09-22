# callbacks.py
# -------------------------------------------------------
# Trainer callbacks:
#  - CSVLoggerCallback: appends metrics to metrics.csv
#  - PseudoPerplexityCallback: computes PPPL on evaluate
# -------------------------------------------------------

import csv
import os
from typing import Iterable

import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, PreTrainedTokenizerBase

from bert_helpers.metrics import pseudo_perplexity_for_texts


class CSVLoggerCallback(TrainerCallback):
    """
    Appends every logged dict (train/eval) to a CSV file.
    """
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._fieldnames = None
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return
        # add global_step if available
        rec = dict(step=state.global_step, **logs)
        # lazy header init
        if self._fieldnames is None:
            self._fieldnames = list(rec.keys())
            header_needed = not os.path.exists(self.csv_path)
            with open(self.csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self._fieldnames)
                if header_needed:
                    w.writeheader()
                w.writerow(rec)
        else:
            # ensure all fields exist
            for k in self._fieldnames:
                if k not in rec:
                    rec[k] = ""
            with open(self.csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self._fieldnames)
                w.writerow(rec)


class PseudoPerplexityCallback(TrainerCallback):
    """
    On each evaluation, compute PPPL on a small set of validation texts
    and log it via trainer.log({log_key: value}).
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        texts: Iterable[str],
        *,
        max_texts: int = 64,
        max_len: int = 128,
        chunk_size: int = 64,
        log_key: str = "eval_pppl",
    ):
        self.tokenizer = tokenizer
        self.texts = list(texts)  # snapshot
        self.max_texts = max_texts
        self.max_len = max_len
        self.chunk_size = chunk_size
        self.log_key = log_key

    def on_evaluate(self, args, state, control, **kwargs):
        trainer = kwargs["model"], kwargs.get("trainer", None)
        # HF passes model, but not trainer; get it from kwargs if present
        model = kwargs["model"]
        trainer_obj = kwargs.get("trainer", None)

        device = next(model.parameters()).device
        pppl = pseudo_perplexity_for_texts(
            model=model,
            tokenizer=self.tokenizer,
            texts=self.texts,
            max_texts=self.max_texts,
            max_len=self.max_len,
            chunk_size=self.chunk_size,
            device=device,
        )
        # Log through the trainer if available, otherwise via control
        if trainer_obj is not None:
            trainer_obj.log({self.log_key: pppl})
        else:
            # Fallback: this will still be picked up by CSVLoggerCallback via on_log
            control.should_log = True
            return {self.log_key: pppl}
