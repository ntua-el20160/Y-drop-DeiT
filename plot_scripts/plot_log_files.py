
#!/usr/bin/env python3
"""
transformer_log_viz.py

Read 1+ training log files (JSON per line or loosely-formatted JSON blobs),
compute summaries, and produce pretty, thesis-ready plots using matplotlib only.
No seaborn, no explicit colors or styles.

Expected keys (best-effort & optional): 
- epoch (int)
- train_loss (float)
- test_loss (float)
- test_acc (float)
- time (float; seconds elapsed from start)
- best_acc (float)
- best_loss (float)
- patience_counter (int)

Usage examples:
  python transformer_log_viz.py \
      --logs run_baseline.jsonl run_conductance.jsonl \
      --titles "Baseline" "Conductance-aware Dropout" \
      --outdir figs \
      --ema 0.9

  # If titles omitted, filenames will be used as labels.

Outputs (in --outdir):
  - summary.csv : comparison table with key metrics
  - *.png figures:
      loss_curves.png
      acc_curve.png
      gen_gap.png
      loss_vs_time.png
      best_so_far.png
      patience.png        (only if patience_counter present)
      final_scatter.png   (final accuracy vs final loss)

Created by: ChatGPT
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------

def robust_json_objects(text: str) -> List[dict]:
    """
    Extract JSON objects from a string that may contain one JSON per line
    or concatenated JSON blobs. Returns list of dicts. Skips unparsable bits.
    """
    objs = []
    # Fast path: try line-by-line JSONL
    line_mode_success = False
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                objs.append(obj)
                line_mode_success = True
        except Exception:
            continue
    if line_mode_success and objs:
        return objs

    # Fallback: regex to find {...} blocks and parse each
    # This is naive but works for typical logs without nested braces in strings.
    pattern = re.compile(r'\{[^{}]*\}')
    for m in pattern.finditer(text):
        snippet = m.group(0)
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                objs.append(obj)
        except Exception:
            continue
    return objs


def read_log_file(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    objs = robust_json_objects(text)
    if not objs:
        raise ValueError(f"No JSON objects found in {path}.")
    # Ensure sorting by epoch if present; otherwise preserve order.
    if "epoch" in objs[0]:
        try:
            objs = sorted(objs, key=lambda d: d.get("epoch", 0))
        except Exception:
            pass
    return objs


def ema(series: np.ndarray, alpha: Optional[float]) -> np.ndarray:
    """Exponential moving average smoothing. alpha in (0,1). If None, returns series."""
    if alpha is None:
        return series
    if not (0.0 < alpha < 1.0):
        raise ValueError("EMA alpha must be in (0,1).")
    out = np.zeros_like(series, dtype=float)
    val = series[0]
    out[0] = val
    for i in range(1, len(series)):
        val = alpha * series[i] + (1 - alpha) * val
        out[i] = val
    return out


def safe_array(values: List[Optional[float]]) -> np.ndarray:
    """Convert to numpy array with nan for missing values."""
    return np.array([float(v) if v is not None else np.nan for v in values], dtype=float)


@dataclass
class RunData:
    label: str
    epochs: np.ndarray
    train_loss: np.ndarray
    test_loss: np.ndarray
    test_acc: np.ndarray
    time: np.ndarray
    best_acc: np.ndarray
    best_loss: np.ndarray
    patience: np.ndarray
    # Derived
    gen_gap: np.ndarray = field(default_factory=lambda: np.array([]))
    final_epoch: int = 0
    final_train_loss: float = math.nan
    final_test_loss: float = math.nan
    final_test_acc: float = math.nan
    min_test_loss: float = math.nan
    min_test_loss_epoch: int = 0
    max_test_acc: float = math.nan
    max_test_acc_epoch: int = 0
    total_time_sec: float = math.nan
    time_to_min_loss_sec: float = math.nan
    time_to_max_acc_sec: float = math.nan


def build_run(objs: List[dict], label: str) -> RunData:
    def take(key, default=None):
        return [o.get(key, default) for o in objs]

    epochs = safe_array(take("epoch"))
    train_loss = safe_array(take("train_loss"))
    test_loss = safe_array(take("test_loss"))
    test_acc = safe_array(take("test_acc"))
    time_sec = safe_array(take("time"))
    best_acc = safe_array(take("best_acc"))
    best_loss = safe_array(take("best_loss"))
    patience = safe_array(take("patience_counter"))

    # Derived
    gen_gap = train_loss - test_loss

    # Final & extrema
    final_idx = len(epochs) - 1
    final_epoch = int(epochs[final_idx]) if len(epochs) else 0
    final_train_loss = float(train_loss[final_idx]) if len(train_loss) else math.nan
    final_test_loss = float(test_loss[final_idx]) if len(test_loss) else math.nan
    final_test_acc = float(test_acc[final_idx]) if len(test_acc) else math.nan

    # min test loss
    if np.all(np.isnan(test_loss)):
        min_test_loss = math.nan
        min_test_loss_epoch = 0
        time_to_min_loss = math.nan
    else:
        i_min = int(np.nanargmin(test_loss))
        min_test_loss = float(test_loss[i_min])
        min_test_loss_epoch = int(epochs[i_min]) if i_min < len(epochs) else i_min
        time_to_min_loss = float(time_sec[i_min]) if i_min < len(time_sec) else math.nan

    # max test acc
    if np.all(np.isnan(test_acc)):
        max_test_acc = math.nan
        max_test_acc_epoch = 0
        time_to_max_acc = math.nan
    else:
        i_max = int(np.nanargmax(test_acc))
        max_test_acc = float(test_acc[i_max])
        max_test_acc_epoch = int(epochs[i_max]) if i_max < len(epochs) else i_max
        time_to_max_acc = float(time_sec[i_max]) if i_max < len(time_sec) else math.nan

    total_time_sec = float(np.nanmax(time_sec)) if len(time_sec) else math.nan

    return RunData(
        label=label,
        epochs=epochs,
        train_loss=train_loss,
        test_loss=test_loss,
        test_acc=test_acc,
        time=time_sec,
        best_acc=best_acc,
        best_loss=best_loss,
        patience=patience,
        gen_gap=gen_gap,
        final_epoch=final_epoch,
        final_train_loss=final_train_loss,
        final_test_loss=final_test_loss,
        final_test_acc=final_test_acc,
        min_test_loss=min_test_loss,
        min_test_loss_epoch=min_test_loss_epoch,
        max_test_acc=max_test_acc,
        max_test_acc_epoch=max_test_acc_epoch,
        total_time_sec=total_time_sec,
        time_to_min_loss_sec=time_to_min_loss,
        time_to_max_acc_sec=time_to_max_acc,
    )


def seconds_to_hms(sec: float) -> str:
    if math.isnan(sec):
        return ""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def save_summary_csv(runs: List[RunData], outdir: str) -> str:
    path = os.path.join(outdir, "summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "label", "final_epoch", "final_train_loss", "final_test_loss", "final_test_acc",
            "min_test_loss", "min_test_loss_epoch", "time_to_min_loss(h:m:s)",
            "max_test_acc", "max_test_acc_epoch", "time_to_max_acc(h:m:s)",
            "total_time(h:m:s)"
        ])
        for r in runs:
            w.writerow([
                r.label, r.final_epoch, f"{r.final_train_loss:.6f}" if not math.isnan(r.final_train_loss) else "",
                f"{r.final_test_loss:.6f}" if not math.isnan(r.final_test_loss) else "",
                f"{r.final_test_acc:.6f}" if not math.isnan(r.final_test_acc) else "",
                f"{r.min_test_loss:.6f}" if not math.isnan(r.min_test_loss) else "",
                r.min_test_loss_epoch if r.min_test_loss_epoch else "",
                seconds_to_hms(r.time_to_min_loss_sec),
                f"{r.max_test_acc:.6f}" if not math.isnan(r.max_test_acc) else "",
                r.max_test_acc_epoch if r.max_test_acc_epoch else "",
                seconds_to_hms(r.time_to_max_acc_sec),
                seconds_to_hms(r.total_time_sec),
            ])
    return path


def _finalize(ax, title: str, xlabel: str, ylabel: str, legend_outside: bool=True):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    if legend_outside:
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=True)
    else:
        ax.legend(frameon=True)


def plot_loss_curves(runs: List[RunData], outdir: str, alpha_ema: Optional[float]):
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in runs:
        epochs = r.epochs
        tr = np.copy(r.train_loss)
        te = np.copy(r.test_loss)
        if len(tr) and not np.all(np.isnan(tr)):
            tr = ema(tr, alpha_ema) if alpha_ema else tr
            ax.plot(epochs, tr, label=f"{r.label} – train")
        if len(te) and not np.all(np.isnan(te)):
            te = ema(te, alpha_ema) if alpha_ema else te
            ax.plot(epochs, te, linestyle=":", label=f"{r.label} – test")
    _finalize(ax, "Loss over epochs", "Epoch", "Loss")
    fig.tight_layout()
    path = os.path.join(outdir, "loss_curves.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_acc_curve(runs: List[RunData], outdir: str, alpha_ema: Optional[float]):
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in runs:
        if len(r.test_acc) and not np.all(np.isnan(r.test_acc)):
            y = ema(r.test_acc, alpha_ema) if alpha_ema else r.test_acc
            ax.plot(r.epochs, y, label=r.label)
    _finalize(ax, "Test accuracy over epochs", "Epoch", "Accuracy")
    fig.tight_layout()
    path = os.path.join(outdir, "acc_curve.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_gen_gap(runs: List[RunData], outdir: str, alpha_ema: Optional[float]):
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in runs:
        if len(r.gen_gap) and not np.all(np.isnan(r.gen_gap)):
            y = ema(r.gen_gap, alpha_ema) if alpha_ema else r.gen_gap
            ax.plot(r.epochs, y, label=r.label)
    _finalize(ax, "Generalization gap (train_loss − test_loss)", "Epoch", "Gap")
    fig.tight_layout()
    path = os.path.join(outdir, "gen_gap.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_loss_vs_time(runs: List[RunData], outdir: str, alpha_ema: Optional[float]):
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in runs:
        if len(r.test_loss) and len(r.time) and not np.all(np.isnan(r.test_loss)):
            y = ema(r.test_loss, alpha_ema) if alpha_ema else r.test_loss
            ax.plot(r.time / 3600.0, y, label=r.label)
    _finalize(ax, "Test loss vs wall-clock time", "Time (hours)", "Test loss")
    fig.tight_layout()
    path = os.path.join(outdir, "loss_vs_time.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_best_so_far(runs: List[RunData], outdir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in runs:
        # Prefer 'best_acc' if present; otherwise compute cummax of test_acc
        if len(r.best_acc) and not np.all(np.isnan(r.best_acc)):
            y = r.best_acc
            ax.plot(r.epochs, y, label=f"{r.label} – best_acc")
        elif len(r.test_acc) and not np.all(np.isnan(r.test_acc)):
            y = np.maximum.accumulate(np.nan_to_num(r.test_acc, nan=-np.inf))
            y[y == -np.inf] = np.nan
            ax.plot(r.epochs, y, label=f"{r.label} – best(test_acc)")
    _finalize(ax, "Best-so-far accuracy", "Epoch", "Accuracy")
    fig.tight_layout()
    path = os.path.join(outdir, "best_so_far.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_patience(runs: List[RunData], outdir: str):
    # Only plot if any run has patience data
    any_patience = any(r.patience is not None and len(r.patience) and not np.all(np.isnan(r.patience)) for r in runs)
    if not any_patience:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in runs:
        if len(r.patience) and not np.all(np.isnan(r.patience)):
            ax.step(r.epochs, r.patience, where="post", label=r.label)
    _finalize(ax, "Early stopping patience counter", "Epoch", "Patience")
    fig.tight_layout()
    path = os.path.join(outdir, "patience.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_final_scatter(runs: List[RunData], outdir: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in runs:
        x = r.final_test_loss
        y = r.final_test_acc
        if not (math.isnan(x) or math.isnan(y)):
            ax.scatter([x], [y], label=r.label, s=60, marker="o")
            ax.annotate(r.label, (x, y), xytext=(5, 5), textcoords="offset points")
    _finalize(ax, "Final metrics", "Final test loss", "Final test accuracy", legend_outside=False)
    fig.tight_layout()
    path = os.path.join(outdir, "final_scatter.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize transformer training logs.")
    parser.add_argument("--logs", nargs="+", required=True, help="Paths to 1+ log files (JSONL or loose JSON).")
    parser.add_argument("--titles", nargs="*", help="Optional labels for each log (same order). If omitted, filenames are used.")
    parser.add_argument("--outdir", default="figs", help="Output directory for figures & summary.csv")
    parser.add_argument("--ema", type=float, default=None, help="EMA smoothing factor in (0,1). Example: 0.9")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    args = parser.parse_args()

    if args.titles and len(args.titles) != len(args.logs):
        print("If provided, --titles must match the number of --logs.", file=sys.stderr)
        sys.exit(2)

    os.makedirs(args.outdir, exist_ok=True)

    # Set DPI globally
    plt.rcParams["figure.dpi"] = args.dpi
    plt.rcParams["savefig.dpi"] = args.dpi
    plt.rcParams["font.size"] = 11

    runs: List[RunData] = []
    for i, path in enumerate(args.logs):
        label = args.titles[i] if args.titles else os.path.basename(path)
        try:
            objs = read_log_file(path)
        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}", file=sys.stderr)
            continue
        run = build_run(objs, label=label)
        runs.append(run)

    if not runs:
        print("No valid runs loaded. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Save summary CSV
    csv_path = save_summary_csv(runs, args.outdir)
    print(f"Wrote summary: {csv_path}")

    # Plots
    p1 = plot_loss_curves(runs, args.outdir, args.ema)
    print(f"Wrote {p1}")
    p2 = plot_acc_curve(runs, args.outdir, args.ema)
    print(f"Wrote {p2}")
    p3 = plot_gen_gap(runs, args.outdir, args.ema)
    print(f"Wrote {p3}")
    p4 = plot_loss_vs_time(runs, args.outdir, args.ema)
    print(f"Wrote {p4}")
    p5 = plot_best_so_far(runs, args.outdir)
    print(f"Wrote {p5}")
    p6 = plot_patience(runs, args.outdir)
    if p6:
        print(f"Wrote {p6}")
    else:
        print("No patience_counter found; skipping patience plot.")
    p7 = plot_final_scatter(runs, args.outdir)
    print(f"Wrote {p7}")

    print("Done.")


if __name__ == "__main__":
    main()
