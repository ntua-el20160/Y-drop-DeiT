# plot_layer_metrics.py
import argparse
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


HEADER_RE = re.compile(
    r"\[block_(\d+)_layer_(\d+)\]\s+LAST\s+(Standard Dropout|Y-drop)@0"
)
KVS_RE = re.compile(r"(\w+)=([\-0-9\.]+)")

DEFAULT_METRICS = ["mean", "median", "var", "IQR", "CV", "p95", "skew", "gini"]


def parse_log_text(text: str) -> pd.DataFrame:
    """
    Parse the log text and return a tidy DataFrame with columns:
    block, layer, label, method, <metrics...>, x_idx, x_label
    """
    lines = text.splitlines()
    records = []
    i = 0
    seen_order = OrderedDict()

    while i < len(lines):
        m = HEADER_RE.search(lines[i])
        if m:
            block = int(m.group(1))
            layer = int(m.group(2))
            method = m.group(3)
            # Expect metrics on the next line
            if i + 1 < len(lines):
                kvs = dict((k, float(v)) for k, v in KVS_RE.findall(lines[i + 1]))
                label = f"b{block:03d}_l{layer:02d}"
                rec = {"block": block, "layer": layer, "label": label, "method": method}
                rec.update(kvs)
                records.append(rec)
                seen_order.setdefault(label, len(seen_order))
                i += 2
                continue
        i += 1

    if not records:
        raise ValueError("No metrics found. Is this the right log file format?")

    df = pd.DataFrame(records)
    # sort by block, layer, method for deterministic plotting
    df = df.sort_values(["block", "layer", "method"]).reset_index(drop=True)

    # Build layer indices (x-axis) based on unique (block, layer) order
    unique_layers = df[["block", "layer"]].drop_duplicates().reset_index(drop=True)
    index_map = {(int(r.block), int(r.layer)): idx for idx, r in unique_layers.iterrows()}
    df["x_idx"] = df.apply(lambda r: index_map[(int(r["block"]), int(r["layer"]))], axis=1)
    df["x_label"] = df.apply(lambda r: f"b{int(r['block']):02d}-l{int(r['layer']):02d}", axis=1)

    return df


# --- NEW: helper to map layer -> multiplicative factor for sum-from-mean
def _layer_factor(layer: int) -> float:
    """
    Return multiplicative factor per layer when converting mean -> sum.
    Layers 1 and 3 multiply by 192; layer 2 by 576.
    """
    if layer == 1 or layer == 3:
        return 192.0
    if layer == 2:
        return 576.0
    # If other layer indices appear, default to 1.0 but warn via NaN-safe handling upstream
    return 1.0


def make_plots(
    df: pd.DataFrame,
    out_dir: Path,
    metrics=DEFAULT_METRICS,
    title_prefix: str = "",
    save_pdf: bool = False,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    unique_layers = df[["block", "layer"]].drop_duplicates().reset_index(drop=True)
    xticks = list(unique_layers.index)
    xticklabels = [f"b{int(r.block):02d}-l{int(r.layer):02d}" for _, r in unique_layers.iterrows()]

    pdf = PdfPages(out_dir / "layer_metrics_plots.pdf") if save_pdf else None

    for metric in metrics:
        if metric not in df.columns:
            print(f"[WARN] Metric '{metric}' not found in data. Skipping.")
            continue

        # --- NEW: if metric is 'mean', we'll plot a sum-from-mean instead
        plot_metric_name = metric
        y_label = metric
        title_metric_for_display = metric
        filename_metric = metric

        convert_mean_to_sum = (metric.lower() == "mean")
        if convert_mean_to_sum:
            plot_metric_name = "sum"
            y_label = "sum"
            title_metric_for_display = "sum"
            filename_metric = "sum"

        plt.figure(figsize=(12, 6))

        for method in ["Standard Dropout", "Y-drop"]:
            sub = df[df["method"] == method].sort_values("x_idx")
            if sub.empty:
                continue

            if convert_mean_to_sum:
                # multiply each row's mean by layer-dependent factor
                factors = sub["layer"].apply(_layer_factor).values
                y = sub["mean"].values * factors
            else:
                y = sub[metric].values

            x = sub["x_idx"].values
            plt.plot(x, y, marker="o", label=f"{method}")

            # mean (across layers) line for this plotted quantity
            mval = float(np.mean(y))
            plt.axhline(mval, linestyle="--", alpha=0.6, label=f"{method} mean={mval:.3g}")

        plt.xticks(xticks, xticklabels, rotation=90)
        plt.xlabel("Layers (block-layer)")
        plt.ylabel(y_label)

        title = (
            f"{title_prefix}{title_metric_for_display} across layers"
            if title_prefix
            else f"{title_metric_for_display} across layers"
        )
        plt.title(title)
        plt.legend()
        plt.tight_layout()

        out_png = out_dir / f"{filename_metric}_across_layers.png"
        plt.savefig(out_png, dpi=220)
        if pdf is not None:
            pdf.savefig()  # adds current figure as a page
        plt.close()
        print(f"[OK] saved {out_png}")

    if pdf is not None:
        pdf.close()
        print(f"[OK] saved {out_dir / 'layer_metrics_plots.pdf'}")


def main():
    ap = argparse.ArgumentParser(description="Plot per-layer metrics from LAST-epoch comparison log.")
    ap.add_argument("--log", required=True, help="Path to the raw log file.")
    ap.add_argument("--out", required=True, help="Directory to write plots (and optional CSV/PDF).")
    ap.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS,
                    help=f"Metrics to plot (default: {', '.join(DEFAULT_METRICS)})")
    ap.add_argument("--prefix", default="", help="Optional title/filename prefix, e.g., 'CIFAR100 - '")
    ap.add_argument("--csv", action="store_true", help="Also save tidy CSV of parsed values.")
    ap.add_argument("--pdf", action="store_true", help="Also save a single multi-page PDF with all plots.")
    args = ap.parse_args()

    log_path = Path(args.log)
    out_dir = Path(args.out)

    if not log_path.is_file():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    df = parse_log_text(text)

    # Optionally save tidy CSV
    if args.csv:
        csv_path = out_dir / "layer_metrics_tidy.csv"
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"[OK] saved {csv_path}")

    make_plots(
        df=df,
        out_dir=out_dir,
        metrics=args.metrics,
        title_prefix=args.prefix,
        save_pdf=args.pdf,
    )


if __name__ == "__main__":
    main()
