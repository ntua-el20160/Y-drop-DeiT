# # plot_layer_metrics.py
# import argparse
# import re
# from collections import OrderedDict
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages


# HEADER_RE = re.compile(
#     r"\[block_(\d+)_layer_(\d+)\]\s+LAST\s+(Standard Dropout|Y-drop)@0"
# )
# KVS_RE = re.compile(r"(\w+)=([\-0-9\.]+)")

# DEFAULT_METRICS = ["mean", "median", "var", "IQR", "CV", "p95", "skew", "gini"]


# def parse_log_text(text: str) -> pd.DataFrame:
#     """
#     Parse the log text and return a tidy DataFrame with columns:
#     block, layer, label, method, <metrics...>, x_idx, x_label
#     """
#     lines = text.splitlines()
#     records = []
#     i = 0
#     seen_order = OrderedDict()

#     while i < len(lines):
#         m = HEADER_RE.search(lines[i])
#         if m:
#             block = int(m.group(1))
#             layer = int(m.group(2))
#             method = m.group(3)
#             # Expect metrics on the next line
#             if i + 1 < len(lines):
#                 kvs = dict((k, float(v)) for k, v in KVS_RE.findall(lines[i + 1]))
#                 label = f"b{block:03d}_l{layer:02d}"
#                 rec = {"block": block, "layer": layer, "label": label, "method": method}
#                 rec.update(kvs)
#                 records.append(rec)
#                 seen_order.setdefault(label, len(seen_order))
#                 i += 2
#                 continue
#         i += 1

#     if not records:
#         raise ValueError("No metrics found. Is this the right log file format?")

#     df = pd.DataFrame(records)
#     # sort by block, layer, method for deterministic plotting
#     df = df.sort_values(["block", "layer", "method"]).reset_index(drop=True)

#     # Build layer indices (x-axis) based on unique (block, layer) order
#     unique_layers = df[["block", "layer"]].drop_duplicates().reset_index(drop=True)
#     index_map = {(int(r.block), int(r.layer)): idx for idx, r in unique_layers.iterrows()}
#     df["x_idx"] = df.apply(lambda r: index_map[(int(r["block"]), int(r["layer"]))], axis=1)
#     df["x_label"] = df.apply(lambda r: f"b{int(r['block']):02d}-l{int(r['layer']):02d}", axis=1)

#     return df


# def _layer_factor(layer: int) -> float:
#     """
#     Return multiplicative factor per layer when converting mean -> sum.
#     Layers 1 and 3 multiply by 192; layer 2 by 576.
#     """
#     if layer == 1 or layer == 3:
#         return 192.0
#     if layer == 2:
#         return 576.0
#     return 1.0


# def make_plots(
#     df: pd.DataFrame,
#     out_dir: Path,
#     metrics=DEFAULT_METRICS,
#     title_prefix: str = "",
#     save_pdf: bool = False,
# ):
#     out_dir.mkdir(parents=True, exist_ok=True)

#     unique_layers = df[["block", "layer"]].drop_duplicates().reset_index(drop=True)
#     xticks = list(unique_layers.index)
#     xticklabels = [f"b{int(r.block):02d}-l{int(r.layer):02d}" for _, r in unique_layers.iterrows()]

#     pdf = PdfPages(out_dir / "layer_metrics_plots.pdf") if save_pdf else None

#     # Define consistent colors for methods (keep original colors)
#     method_colors = {
#         "Standard Dropout": "#1f77b4",  # blue
#         "Y-drop": "#ff7f0e"  # orange
#     }

#     for metric in metrics:
#         if metric not in df.columns:
#             print(f"[WARN] Metric '{metric}' not found in data. Skipping.")
#             continue

#         # if metric is 'mean', we'll plot a sum-from-mean instead
#         plot_metric_name = metric
#         y_label = metric
#         title_metric_for_display = metric
#         filename_metric = metric

#         convert_mean_to_sum = (metric.lower() == "mean")
#         if convert_mean_to_sum:
#             plot_metric_name = "sum"
#             y_label = "sum (lower is better)"
#             title_metric_for_display = "sum"
#             filename_metric = "sum"
#         else:
#             y_label = f"{metric} (lower is better)"

#         # ===== PLOT 1: Original comparison plot with enhanced quality =====
#         fig, ax = plt.subplots(figsize=(14, 7))

#         method_data = {}
#         for method in ["Standard Dropout", "Y-drop"]:
#             sub = df[df["method"] == method].sort_values("x_idx")
#             if sub.empty:
#                 continue

#             if convert_mean_to_sum:
#                 factors = sub["layer"].apply(_layer_factor).values
#                 y = sub["mean"].values * factors
#             else:
#                 y = sub[metric].values

#             x = sub["x_idx"].values
#             method_data[method] = (x, y)
            
#             color = method_colors[method]
#             ax.plot(x, y, marker="o", markersize=6, linewidth=2, 
#                    label=f"{method}", color=color)

#             # mean line
#             mval = float(np.mean(y))
#             ax.axhline(mval, linestyle="--", alpha=0.5, linewidth=1.5, color=color,
#                       label=f"{method} mean={mval:.4g}")

#         ax.set_xticks(xticks)
#         ax.set_xticklabels(xticklabels, rotation=90, fontsize=9)
#         ax.set_xlabel("Layers (block-layer)", fontsize=12, fontweight='bold')
#         ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
#         ax.grid(True, alpha=0.3, linestyle='--')

#         title = (
#             f"{title_prefix}{title_metric_for_display} across layers"
#             if title_prefix
#             else f"{title_metric_for_display} across layers"
#         )
#         ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
#         ax.legend(fontsize=10, loc='best', framealpha=0.9)
#         plt.tight_layout()

#         out_png = out_dir / f"{filename_metric}_across_layers.png"
#         plt.savefig(out_png, dpi=300, bbox_inches='tight')
#         if pdf is not None:
#             pdf.savefig()
#         plt.close()
#         print(f"[OK] saved {out_png}")

#         # ===== PLOT 2: Difference plot (Y-drop - Standard Dropout) =====
#         if len(method_data) == 2:
#             fig, ax = plt.subplots(figsize=(14, 7))
            
#             x_std, y_std = method_data["Standard Dropout"]
#             x_ydrop, y_ydrop = method_data["Y-drop"]
            
#             # Calculate difference (negative means Y-drop is better/lower)
#             diff = y_ydrop - y_std
            
#             colors = ['green' if d < 0 else 'red' for d in diff]
#             ax.bar(x_ydrop, diff, color=colors, alpha=0.7, width=0.6)
#             ax.axhline(0, color='black', linestyle='-', linewidth=1.5)
            
#             # Add mean difference line
#             mean_diff = np.mean(diff)
#             ax.axhline(mean_diff, color='purple', linestyle='--', linewidth=2,
#                       label=f"Mean difference = {mean_diff:.4g}")
            
#             # Add text annotations for significant differences
#             for i, (x, d) in enumerate(zip(x_ydrop, diff)):
#                 if abs(d) > abs(mean_diff) * 1.5:  # Highlight large differences
#                     ax.text(x, d, f'{d:.3g}', ha='center', 
#                            va='bottom' if d > 0 else 'top', fontsize=7, fontweight='bold')
            
#             ax.set_xticks(xticks)
#             ax.set_xticklabels(xticklabels, rotation=90, fontsize=9)
#             ax.set_xlabel("Layers (block-layer)", fontsize=12, fontweight='bold')
#             ax.set_ylabel(f"Difference (Y-drop − Standard Dropout)\nNegative is better", 
#                          fontsize=12, fontweight='bold')
#             ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            
#             title_diff = (
#                 f"{title_prefix}{title_metric_for_display} - Method Difference"
#                 if title_prefix
#                 else f"{title_metric_for_display} - Method Difference"
#             )
#             ax.set_title(title_diff, fontsize=14, fontweight='bold', pad=15)
#             ax.legend(fontsize=10, loc='best', framealpha=0.9)
#             plt.tight_layout()
            
#             out_png_diff = out_dir / f"{filename_metric}_difference.png"
#             plt.savefig(out_png_diff, dpi=300, bbox_inches='tight')
#             if pdf is not None:
#                 pdf.savefig()
#             plt.close()
#             print(f"[OK] saved {out_png_diff}")

#             # ===== PLOT 3: Relative improvement percentage =====
#             fig, ax = plt.subplots(figsize=(14, 7))
            
#             # Calculate percentage improvement (negative means Y-drop is better)
#             with np.errstate(divide='ignore', invalid='ignore'):
#                 pct_change = ((y_ydrop - y_std) / np.abs(y_std)) * 100
#                 pct_change = np.nan_to_num(pct_change, nan=0.0, posinf=0.0, neginf=0.0)
            
#             colors = ['green' if p < 0 else 'red' for p in pct_change]
#             ax.bar(x_ydrop, pct_change, color=colors, alpha=0.7, width=0.6)
#             ax.axhline(0, color='black', linestyle='-', linewidth=1.5)
            
#             mean_pct = np.mean(pct_change)
#             ax.axhline(mean_pct, color='purple', linestyle='--', linewidth=2,
#                       label=f"Mean improvement = {mean_pct:.2f}%")
            
#             # Add text annotations for significant changes
#             for i, (x, p) in enumerate(zip(x_ydrop, pct_change)):
#                 if abs(p) > abs(mean_pct) * 1.5:
#                     ax.text(x, p, f'{p:.1f}%', ha='center', 
#                            va='bottom' if p > 0 else 'top', fontsize=7, fontweight='bold')
            
#             ax.set_xticks(xticks)
#             ax.set_xticklabels(xticklabels, rotation=90, fontsize=9)
#             ax.set_xlabel("Layers (block-layer)", fontsize=12, fontweight='bold')
#             ax.set_ylabel("Relative Change (%)\nNegative is better", 
#                          fontsize=12, fontweight='bold')
#             ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            
#             title_pct = (
#                 f"{title_prefix}{title_metric_for_display} - Relative Improvement"
#                 if title_prefix
#                 else f"{title_metric_for_display} - Relative Improvement"
#             )
#             ax.set_title(title_pct, fontsize=14, fontweight='bold', pad=15)
#             ax.legend(fontsize=10, loc='best', framealpha=0.9)
#             plt.tight_layout()
            
#             out_png_pct = out_dir / f"{filename_metric}_relative_improvement.png"
#             plt.savefig(out_png_pct, dpi=300, bbox_inches='tight')
#             if pdf is not None:
#                 pdf.savefig()
#             plt.close()
#             print(f"[OK] saved {out_png_pct}")

#             # ===== PLOT 4: Cumulative advantage plot =====
#             fig, ax = plt.subplots(figsize=(14, 7))
            
#             cumulative_diff = np.cumsum(diff)
            
#             color = 'green' if cumulative_diff[-1] < 0 else 'red'
#             ax.plot(x_ydrop, cumulative_diff, marker='o', linewidth=2.5, 
#                    markersize=6, color=color, label='Cumulative difference')
#             ax.axhline(0, color='black', linestyle='-', linewidth=1.5)
#             ax.fill_between(x_ydrop, cumulative_diff, 0, alpha=0.3, color=color)
            
#             # Final cumulative value
#             final_val = cumulative_diff[-1]
#             ax.text(x_ydrop[-1], final_val, f'  Final: {final_val:.3g}', 
#                    va='center', fontsize=11, fontweight='bold', 
#                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
#             ax.set_xticks(xticks)
#             ax.set_xticklabels(xticklabels, rotation=90, fontsize=9)
#             ax.set_xlabel("Layers (block-layer)", fontsize=12, fontweight='bold')
#             ax.set_ylabel(f"Cumulative Difference\nNegative is better", 
#                          fontsize=12, fontweight='bold')
#             ax.grid(True, alpha=0.3, linestyle='--')
            
#             title_cum = (
#                 f"{title_prefix}{title_metric_for_display} - Cumulative Advantage"
#                 if title_prefix
#                 else f"{title_metric_for_display} - Cumulative Advantage"
#             )
#             ax.set_title(title_cum, fontsize=14, fontweight='bold', pad=15)
#             ax.legend(fontsize=10, loc='best', framealpha=0.9)
#             plt.tight_layout()
            
#             out_png_cum = out_dir / f"{filename_metric}_cumulative.png"
#             plt.savefig(out_png_cum, dpi=300, bbox_inches='tight')
#             if pdf is not None:
#                 pdf.savefig()
#             plt.close()
#             print(f"[OK] saved {out_png_cum}")

#     # ===== ADDITIONAL PLOTS: Multi-metric comparisons =====
    
#     # PLOT: Summary statistics table as visualization
#     fig, ax = plt.subplots(figsize=(14, 8))
#     ax.axis('tight')
#     ax.axis('off')
    
#     summary_data = []
#     for metric in metrics:
#         if metric not in df.columns:
#             continue
            
#         convert_mean_to_sum = (metric.lower() == "mean")
#         display_metric = "sum" if convert_mean_to_sum else metric
        
#         for method in ["Standard Dropout", "Y-drop"]:
#             sub = df[df["method"] == method].sort_values("x_idx")
#             if sub.empty:
#                 continue
                
#             if convert_mean_to_sum:
#                 factors = sub["layer"].apply(_layer_factor).values
#                 values = sub["mean"].values * factors
#             else:
#                 values = sub[metric].values
            
#             summary_data.append([
#                 display_metric,
#                 method,
#                 f"{np.mean(values):.4g}",
#                 f"{np.median(values):.4g}",
#                 f"{np.std(values):.4g}",
#                 f"{np.min(values):.4g}",
#                 f"{np.max(values):.4g}"
#             ])
    
#     if summary_data:
#         table = ax.table(cellText=summary_data,
#                         colLabels=['Metric', 'Method', 'Mean', 'Median', 'Std Dev', 'Min', 'Max'],
#                         cellLoc='center',
#                         loc='center',
#                         colWidths=[0.15, 0.2, 0.13, 0.13, 0.13, 0.13, 0.13])
#         table.auto_set_font_size(False)
#         table.set_fontsize(10)
#         table.scale(1, 2.5)
        
#         # Color header
#         for i in range(7):
#             table[(0, i)].set_facecolor('#4472C4')
#             table[(0, i)].set_text_props(weight='bold', color='white')
        
#         # Color rows by method
#         for i, row in enumerate(summary_data, start=1):
#             color = '#D6E9F8' if row[1] == "Standard Dropout" else '#FFE6CC'
#             for j in range(7):
#                 table[(i, j)].set_facecolor(color)
        
#         title_summary = f"{title_prefix}Summary Statistics" if title_prefix else "Summary Statistics"
#         plt.title(title_summary, fontsize=16, fontweight='bold', pad=20)
        
#         out_png_summary = out_dir / "summary_statistics.png"
#         plt.savefig(out_png_summary, dpi=300, bbox_inches='tight')
#         if pdf is not None:
#             pdf.savefig()
#         plt.close()
#         print(f"[OK] saved {out_png_summary}")

#     # PLOT: Per-block aggregation
#     if 'block' in df.columns:
#         for metric in metrics:
#             if metric not in df.columns:
#                 continue
            
#             convert_mean_to_sum = (metric.lower() == "mean")
#             display_metric = "sum" if convert_mean_to_sum else metric
            
#             fig, ax = plt.subplots(figsize=(12, 7))
            
#             blocks = sorted(df['block'].unique())
#             width = 0.35
#             x_pos = np.arange(len(blocks))
            
#             for i, method in enumerate(["Standard Dropout", "Y-drop"]):
#                 block_means = []
#                 for block in blocks:
#                     sub = df[(df["method"] == method) & (df["block"] == block)]
#                     if sub.empty:
#                         block_means.append(0)
#                         continue
                    
#                     if convert_mean_to_sum:
#                         factors = sub["layer"].apply(_layer_factor).values
#                         values = sub["mean"].values * factors
#                     else:
#                         values = sub[metric].values
                    
#                     block_means.append(np.mean(values))
                
#                 offset = width * (i - 0.5)
#                 color = method_colors[method]
#                 bars = ax.bar(x_pos + offset, block_means, width, 
#                              label=method, color=color, alpha=0.8)
                
#                 # Add value labels on bars
#                 for bar in bars:
#                     height = bar.get_height()
#                     ax.text(bar.get_x() + bar.get_width()/2., height,
#                            f'{height:.3g}',
#                            ha='center', va='bottom', fontsize=9, fontweight='bold')
            
#             ax.set_xlabel('Block', fontsize=12, fontweight='bold')
#             ax.set_ylabel(f'Mean {display_metric} (lower is better)', 
#                          fontsize=12, fontweight='bold')
#             ax.set_title(f"{title_prefix}Mean {display_metric} per Block" if title_prefix 
#                         else f"Mean {display_metric} per Block",
#                         fontsize=14, fontweight='bold', pad=15)
#             ax.set_xticks(x_pos)
#             ax.set_xticklabels([f'Block {b}' for b in blocks])
#             ax.legend(fontsize=10, loc='best', framealpha=0.9)
#             ax.grid(True, alpha=0.3, linestyle='--', axis='y')
#             plt.tight_layout()
            
#             out_png_block = out_dir / f"{display_metric}_per_block.png"
#             plt.savefig(out_png_block, dpi=300, bbox_inches='tight')
#             if pdf is not None:
#                 pdf.savefig()
#             plt.close()
#             print(f"[OK] saved {out_png_block}")

#     # PLOT: Win/Loss summary across all metrics
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     win_loss_data = []
#     for metric in metrics:
#         if metric not in df.columns:
#             continue
        
#         convert_mean_to_sum = (metric.lower() == "mean")
#         display_metric = "sum" if convert_mean_to_sum else metric
        
#         std_sub = df[df["method"] == "Standard Dropout"].sort_values("x_idx")
#         ydrop_sub = df[df["method"] == "Y-drop"].sort_values("x_idx")
        
#         if std_sub.empty or ydrop_sub.empty:
#             continue
        
#         if convert_mean_to_sum:
#             std_vals = std_sub["mean"].values * std_sub["layer"].apply(_layer_factor).values
#             ydrop_vals = ydrop_sub["mean"].values * ydrop_sub["layer"].apply(_layer_factor).values
#         else:
#             std_vals = std_sub[metric].values
#             ydrop_vals = ydrop_sub[metric].values
        
#         wins = np.sum(ydrop_vals < std_vals)
#         losses = np.sum(ydrop_vals > std_vals)
#         ties = np.sum(ydrop_vals == std_vals)
        
#         win_loss_data.append([display_metric, wins, ties, losses])
    
#     if win_loss_data:
#         metrics_list = [row[0] for row in win_loss_data]
#         wins = [row[1] for row in win_loss_data]
#         ties = [row[2] for row in win_loss_data]
#         losses = [row[3] for row in win_loss_data]
        
#         x = np.arange(len(metrics_list))
#         width = 0.6
        
#         p1 = ax.bar(x, wins, width, label='Y-drop wins (better)', color='green', alpha=0.8)
#         p2 = ax.bar(x, ties, width, bottom=wins, label='Ties', color='gray', alpha=0.6)
#         p3 = ax.bar(x, losses, width, bottom=np.array(wins)+np.array(ties), 
#                    label='Y-drop losses', color='red', alpha=0.8)
        
#         # Add counts on bars
#         for i, (w, t, l) in enumerate(zip(wins, ties, losses)):
#             if w > 0:
#                 ax.text(i, w/2, str(w), ha='center', va='center', 
#                        fontweight='bold', color='white', fontsize=10)
#             if t > 0:
#                 ax.text(i, w + t/2, str(t), ha='center', va='center', 
#                        fontweight='bold', color='white', fontsize=10)
#             if l > 0:
#                 ax.text(i, w + t + l/2, str(l), ha='center', va='center', 
#                        fontweight='bold', color='white', fontsize=10)
        
#         ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
#         ax.set_ylabel('Number of Layers', fontsize=12, fontweight='bold')
#         ax.set_title(f"{title_prefix}Y-drop Performance per Layer (Win/Tie/Loss)" 
#                     if title_prefix else "Y-drop Performance per Layer (Win/Tie/Loss)",
#                     fontsize=14, fontweight='bold', pad=15)
#         ax.set_xticks(x)
#         ax.set_xticklabels(metrics_list, rotation=45, ha='right')
#         ax.legend(fontsize=10, loc='best', framealpha=0.9)
#         ax.grid(True, alpha=0.3, linestyle='--', axis='y')
#         plt.tight_layout()
        
#         out_png_winloss = out_dir / "win_loss_summary.png"
#         plt.savefig(out_png_winloss, dpi=300, bbox_inches='tight')
#         if pdf is not None:
#             pdf.savefig()
#         plt.close()
#         print(f"[OK] saved {out_png_winloss}")

#     if pdf is not None:
#         pdf.close()
#         print(f"[OK] saved {out_dir / 'layer_metrics_plots.pdf'}")


# def main():
#     ap = argparse.ArgumentParser(description="Plot per-layer metrics from LAST-epoch comparison log.")
#     ap.add_argument("--log", required=True, help="Path to the raw log file.")
#     ap.add_argument("--out", required=True, help="Directory to write plots (and optional CSV/PDF).")
#     ap.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS,
#                     help=f"Metrics to plot (default: {', '.join(DEFAULT_METRICS)})")
#     ap.add_argument("--prefix", default="", help="Optional title/filename prefix, e.g., 'CIFAR100 - '")
#     ap.add_argument("--csv", action="store_true", help="Also save tidy CSV of parsed values.")
#     ap.add_argument("--pdf", action="store_true", help="Also save a single multi-page PDF with all plots.")
#     args = ap.parse_args()

#     log_path = Path(args.log)
#     out_dir = Path(args.out)

#     if not log_path.is_file():
#         raise FileNotFoundError(f"Log file not found: {log_path}")

#     text = log_path.read_text(encoding="utf-8", errors="ignore")
#     df = parse_log_text(text)

#     # Optionally save tidy CSV
#     if args.csv:
#         csv_path = out_dir / "layer_metrics_tidy.csv"
#         out_dir.mkdir(parents=True, exist_ok=True)
#         df.to_csv(csv_path, index=False)
#         print(f"[OK] saved {csv_path}")

#     make_plots(
#         df=df,
#         out_dir=out_dir,
#         metrics=args.metrics,
#         title_prefix=args.prefix,
#         save_pdf=args.pdf,
#     )


# if __name__ == "__main__":
#     main()
# plot_layer_metrics.py
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
    r"\[b?lock_(\d+)_layer_(\d+)\]\s+LAST\s+(.+?)@0"
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


def _layer_factor(layer: int) -> float:
    """
    Return multiplicative factor per layer when converting mean -> sum.
    Layers 1 and 3 multiply by 192; layer 2 by 576.
    """
    if layer == 1 or layer == 3:
        return 192.0
    if layer == 2:
        return 576.0
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

    # Define colors - will assign automatically if new methods appear
    default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
                      "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    
    unique_methods = df["method"].unique()
    method_colors = {}
    
    # Prioritize known methods with their original colors
    known_mapping = {
        "Standard Dropout": "#1f77b4",  # blue
        "Y-drop": "#ff7f0e"  # orange
    }
    
    color_idx = 0
    for method in unique_methods:
        if method in known_mapping:
            method_colors[method] = known_mapping[method]
        else:
            # Assign next available color, skip already used ones
            while color_idx < len(default_colors) and default_colors[color_idx] in known_mapping.values():
                color_idx += 1
            if color_idx < len(default_colors):
                method_colors[method] = default_colors[color_idx]
                color_idx += 1
            else:
                # Fallback to cycling through colors
                method_colors[method] = default_colors[len(method_colors) % len(default_colors)]

    for metric in metrics:
        if metric not in df.columns:
            print(f"[WARN] Metric '{metric}' not found in data. Skipping.")
            continue

        # if metric is 'mean', we'll plot a sum-from-mean instead
        plot_metric_name = metric
        y_label = metric
        title_metric_for_display = metric
        filename_metric = metric

        convert_mean_to_sum = (metric.lower() == "mean")
        if convert_mean_to_sum:
            plot_metric_name = "sum"
            y_label = "sum (lower is better)"
            title_metric_for_display = "sum"
            filename_metric = "sum"
        else:
            y_label = f"{metric} (lower is better)"

        # ===== PLOT 1: Original comparison plot with enhanced quality =====
        fig, ax = plt.subplots(figsize=(14, 7))

        method_data = {}
        for method in unique_methods:
            sub = df[df["method"] == method].sort_values("x_idx")
            if sub.empty:
                continue

            if convert_mean_to_sum:
                factors = sub["layer"].apply(_layer_factor).values
                y = sub["mean"].values * factors
            else:
                y = sub[metric].values

            x = sub["x_idx"].values
            method_data[method] = (x, y)
            
            color = method_colors.get(method, "#333333")
            ax.plot(x, y, marker="o", markersize=6, linewidth=2, 
                   label=f"{method}", color=color)

            # mean line
            mval = float(np.mean(y))
            ax.axhline(mval, linestyle="--", alpha=0.5, linewidth=1.5, color=color,
                      label=f"{method} mean={mval:.4g}")

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=90, fontsize=9)
        ax.set_xlabel("Layers (block-layer)", fontsize=12, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')

        title = (
            f"{title_prefix}{title_metric_for_display} across layers"
            if title_prefix
            else f"{title_metric_for_display} across layers"
        )
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.legend(fontsize=10, loc='best', framealpha=0.9)
        plt.tight_layout()

        out_png = out_dir / f"{filename_metric}_across_layers.png"
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        if pdf is not None:
            pdf.savefig()
        plt.close()
        print(f"[OK] saved {out_png}")

        # ===== PLOT 2: Difference plot (Method2 - Method1) =====
        if len(method_data) >= 2:
            # Get first two methods for comparison
            methods_list = list(method_data.keys())
            method1, method2 = methods_list[0], methods_list[1]
            
            fig, ax = plt.subplots(figsize=(14, 7))
            
            x_m1, y_m1 = method_data[method1]
            x_m2, y_m2 = method_data[method2]
            
            # Calculate difference (negative means method2 is better/lower)
            diff = y_m2 - y_m1
            
            colors = ['green' if d < 0 else 'red' for d in diff]
            ax.bar(x_m2, diff, color=colors, alpha=0.7, width=0.6)
            ax.axhline(0, color='black', linestyle='-', linewidth=1.5)
            
            # Add mean difference line
            mean_diff = np.mean(diff)
            ax.axhline(mean_diff, color='purple', linestyle='--', linewidth=2,
                      label=f"Mean difference = {mean_diff:.4g}")
            
            # Add text annotations for significant differences
            for i, (x, d) in enumerate(zip(x_m2, diff)):
                if abs(d) > abs(mean_diff) * 1.5:  # Highlight large differences
                    ax.text(x, d, f'{d:.3g}', ha='center', 
                           va='bottom' if d > 0 else 'top', fontsize=7, fontweight='bold')
            
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation=90, fontsize=9)
            ax.set_xlabel("Layers (block-layer)", fontsize=12, fontweight='bold')
            ax.set_ylabel(f"Difference ({method2} − {method1})\nNegative is better", 
                         fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            
            title_diff = (
                f"{title_prefix}{title_metric_for_display} - Method Difference"
                if title_prefix
                else f"{title_metric_for_display} - Method Difference"
            )
            ax.set_title(title_diff, fontsize=14, fontweight='bold', pad=15)
            ax.legend(fontsize=10, loc='best', framealpha=0.9)
            plt.tight_layout()
            
            out_png_diff = out_dir / f"{filename_metric}_difference.png"
            plt.savefig(out_png_diff, dpi=300, bbox_inches='tight')
            if pdf is not None:
                pdf.savefig()
            plt.close()
            print(f"[OK] saved {out_png_diff}")

            # ===== PLOT 3: Relative improvement percentage =====
            fig, ax = plt.subplots(figsize=(14, 7))
            
            # Calculate percentage improvement (negative means method2 is better)
            with np.errstate(divide='ignore', invalid='ignore'):
                pct_change = ((y_m2 - y_m1) / np.abs(y_m1)) * 100
                pct_change = np.nan_to_num(pct_change, nan=0.0, posinf=0.0, neginf=0.0)
            
            colors = ['green' if p < 0 else 'red' for p in pct_change]
            ax.bar(x_m2, pct_change, color=colors, alpha=0.7, width=0.6)
            ax.axhline(0, color='black', linestyle='-', linewidth=1.5)
            
            mean_pct = np.mean(pct_change)
            ax.axhline(mean_pct, color='purple', linestyle='--', linewidth=2,
                      label=f"Mean improvement = {mean_pct:.2f}%")
            
            # Add text annotations for significant changes
            for i, (x, p) in enumerate(zip(x_m2, pct_change)):
                if abs(p) > abs(mean_pct) * 1.5:
                    ax.text(x, p, f'{p:.1f}%', ha='center', 
                           va='bottom' if p > 0 else 'top', fontsize=7, fontweight='bold')
            
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation=90, fontsize=9)
            ax.set_xlabel("Layers (block-layer)", fontsize=12, fontweight='bold')
            ax.set_ylabel("Relative Change (%)\nNegative is better", 
                         fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            
            title_pct = (
                f"{title_prefix}{title_metric_for_display} - Relative Improvement"
                if title_prefix
                else f"{title_metric_for_display} - Relative Improvement"
            )
            ax.set_title(title_pct, fontsize=14, fontweight='bold', pad=15)
            ax.legend(fontsize=10, loc='best', framealpha=0.9)
            plt.tight_layout()
            
            out_png_pct = out_dir / f"{filename_metric}_relative_improvement.png"
            plt.savefig(out_png_pct, dpi=300, bbox_inches='tight')
            if pdf is not None:
                pdf.savefig()
            plt.close()
            print(f"[OK] saved {out_png_pct}")

            # ===== PLOT 4: Cumulative advantage plot =====
            fig, ax = plt.subplots(figsize=(14, 7))
            
            cumulative_diff = np.cumsum(diff)
            
            color = 'green' if cumulative_diff[-1] < 0 else 'red'
            ax.plot(x_m2, cumulative_diff, marker='o', linewidth=2.5, 
                   markersize=6, color=color, label='Cumulative difference')
            ax.axhline(0, color='black', linestyle='-', linewidth=1.5)
            ax.fill_between(x_m2, cumulative_diff, 0, alpha=0.3, color=color)
            
            # Final cumulative value
            final_val = cumulative_diff[-1]
            ax.text(x_m2[-1], final_val, f'  Final: {final_val:.3g}', 
                   va='center', fontsize=11, fontweight='bold', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation=90, fontsize=9)
            ax.set_xlabel("Layers (block-layer)", fontsize=12, fontweight='bold')
            ax.set_ylabel(f"Cumulative Difference\nNegative is better", 
                         fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            
            title_cum = (
                f"{title_prefix}{title_metric_for_display} - Cumulative Advantage ({method2} vs {method1})"
                if title_prefix
                else f"{title_metric_for_display} - Cumulative Advantage ({method2} vs {method1})"
            )
            ax.set_title(title_cum, fontsize=14, fontweight='bold', pad=15)
            ax.legend(fontsize=10, loc='best', framealpha=0.9)
            plt.tight_layout()
            
            out_png_cum = out_dir / f"{filename_metric}_cumulative.png"
            plt.savefig(out_png_cum, dpi=300, bbox_inches='tight')
            if pdf is not None:
                pdf.savefig()
            plt.close()
            print(f"[OK] saved {out_png_cum}")

    # ===== ADDITIONAL PLOTS: Multi-metric comparisons =====
    
    # PLOT: Summary statistics table as visualization
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    summary_data = []
    for metric in metrics:
        if metric not in df.columns:
            continue
            
        convert_mean_to_sum = (metric.lower() == "mean")
        display_metric = "sum" if convert_mean_to_sum else metric
        
        for method in unique_methods:
            sub = df[df["method"] == method].sort_values("x_idx")
            if sub.empty:
                continue
                
            if convert_mean_to_sum:
                factors = sub["layer"].apply(_layer_factor).values
                values = sub["mean"].values * factors
            else:
                values = sub[metric].values
            
            summary_data.append([
                display_metric,
                method,
                f"{np.mean(values):.4g}",
                f"{np.median(values):.4g}",
                f"{np.std(values):.4g}",
                f"{np.min(values):.4g}",
                f"{np.max(values):.4g}"
            ])
    
    if summary_data:
        table = ax.table(cellText=summary_data,
                        colLabels=['Metric', 'Method', 'Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.15, 0.2, 0.13, 0.13, 0.13, 0.13, 0.13])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Color header
        for i in range(7):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows by method - assign colors dynamically
        method_table_colors = {}
        table_color_palette = ['#D6E9F8', '#FFE6CC', '#E6F4EA', '#FCE8E8', '#F3E5F5']
        for idx, method in enumerate(unique_methods):
            method_table_colors[method] = table_color_palette[idx % len(table_color_palette)]
        
        for i, row in enumerate(summary_data, start=1):
            color = method_table_colors.get(row[1], '#F0F0F0')
            for j in range(7):
                table[(i, j)].set_facecolor(color)
        
        title_summary = f"{title_prefix}Summary Statistics" if title_prefix else "Summary Statistics"
        plt.title(title_summary, fontsize=16, fontweight='bold', pad=20)
        
        out_png_summary = out_dir / "summary_statistics.png"
        plt.savefig(out_png_summary, dpi=300, bbox_inches='tight')
        if pdf is not None:
            pdf.savefig()
        plt.close()
        print(f"[OK] saved {out_png_summary}")

    # PLOT: Per-block aggregation
    if 'block' in df.columns:
        for metric in metrics:
            if metric not in df.columns:
                continue
            
            convert_mean_to_sum = (metric.lower() == "mean")
            display_metric = "sum" if convert_mean_to_sum else metric
            
            fig, ax = plt.subplots(figsize=(12, 7))
            
            blocks = sorted(df['block'].unique())
            width = 0.8 / len(unique_methods)  # Dynamic width based on number of methods
            x_pos = np.arange(len(blocks))
            
            for i, method in enumerate(unique_methods):
                block_means = []
                for block in blocks:
                    sub = df[(df["method"] == method) & (df["block"] == block)]
                    if sub.empty:
                        block_means.append(0)
                        continue
                    
                    if convert_mean_to_sum:
                        factors = sub["layer"].apply(_layer_factor).values
                        values = sub["mean"].values * factors
                    else:
                        values = sub[metric].values
                    
                    block_means.append(np.mean(values))
                
                offset = width * (i - len(unique_methods)/2 + 0.5)
                color = method_colors.get(method, "#333333")
                bars = ax.bar(x_pos + offset, block_means, width, 
                             label=method, color=color, alpha=0.8)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3g}',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_xlabel('Block', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'Mean {display_metric} (lower is better)', 
                         fontsize=12, fontweight='bold')
            ax.set_title(f"{title_prefix}Mean {display_metric} per Block" if title_prefix 
                        else f"Mean {display_metric} per Block",
                        fontsize=14, fontweight='bold', pad=15)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'Block {b}' for b in blocks])
            ax.legend(fontsize=10, loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            plt.tight_layout()
            
            out_png_block = out_dir / f"{display_metric}_per_block.png"
            plt.savefig(out_png_block, dpi=300, bbox_inches='tight')
            if pdf is not None:
                pdf.savefig()
            plt.close()
            print(f"[OK] saved {out_png_block}")

    # PLOT: Win/Loss summary across all metrics (comparing first two methods)
    if len(unique_methods) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        method1, method2 = unique_methods[0], unique_methods[1]
        
        win_loss_data = []
        for metric in metrics:
            if metric not in df.columns:
                continue
            
            convert_mean_to_sum = (metric.lower() == "mean")
            display_metric = "sum" if convert_mean_to_sum else metric
            
            m1_sub = df[df["method"] == method1].sort_values("x_idx")
            m2_sub = df[df["method"] == method2].sort_values("x_idx")
            
            if m1_sub.empty or m2_sub.empty:
                continue
            
            if convert_mean_to_sum:
                m1_vals = m1_sub["mean"].values * m1_sub["layer"].apply(_layer_factor).values
                m2_vals = m2_sub["mean"].values * m2_sub["layer"].apply(_layer_factor).values
            else:
                m1_vals = m1_sub[metric].values
                m2_vals = m2_sub[metric].values
            
            wins = np.sum(m2_vals < m1_vals)
            losses = np.sum(m2_vals > m1_vals)
            ties = np.sum(m2_vals == m1_vals)
            
            win_loss_data.append([display_metric, wins, ties, losses])
    
    if win_loss_data:
        metrics_list = [row[0] for row in win_loss_data]
        wins = [row[1] for row in win_loss_data]
        ties = [row[2] for row in win_loss_data]
        losses = [row[3] for row in win_loss_data]
        
        x = np.arange(len(metrics_list))
        width = 0.6
        
        p1 = ax.bar(x, wins, width, label=f'{method2} wins (better)', color='green', alpha=0.8)
        p2 = ax.bar(x, ties, width, bottom=wins, label='Ties', color='gray', alpha=0.6)
        p3 = ax.bar(x, losses, width, bottom=np.array(wins)+np.array(ties), 
                   label=f'{method2} losses', color='red', alpha=0.8)
        
        # Add counts on bars
        for i, (w, t, l) in enumerate(zip(wins, ties, losses)):
            if w > 0:
                ax.text(i, w/2, str(w), ha='center', va='center', 
                       fontweight='bold', color='white', fontsize=10)
            if t > 0:
                ax.text(i, w + t/2, str(t), ha='center', va='center', 
                       fontweight='bold', color='white', fontsize=10)
            if l > 0:
                ax.text(i, w + t + l/2, str(l), ha='center', va='center', 
                       fontweight='bold', color='white', fontsize=10)
        
        ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Layers', fontsize=12, fontweight='bold')
        ax.set_title(f"{title_prefix}{method2} vs {method1} Performance per Layer (Win/Tie/Loss)" 
                    if title_prefix else f"{method2} vs {method1} Performance per Layer (Win/Tie/Loss)",
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_list, rotation=45, ha='right')
        ax.legend(fontsize=10, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        plt.tight_layout()
        
        out_png_winloss = out_dir / "win_loss_summary.png"
        plt.savefig(out_png_winloss, dpi=300, bbox_inches='tight')
        if pdf is not None:
            pdf.savefig()
        plt.close()
        print(f"[OK] saved {out_png_winloss}")

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