from __future__ import annotations
import os, json, sys
from typing import Dict, Optional, Tuple, List
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

# ============== small math helpers ==============

def _to_1d_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().reshape(-1).cpu().numpy().astype(np.float32)

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _load_manifest_epochs(root: str) -> List[str]:
    return sorted([d for d in os.listdir(root) if d.startswith("epoch_")])

def _mad_fast(x):
    n = x.size
    if n == 0:
        return float("nan")
    xs = np.sort(x)
    i = np.arange(1, n + 1)
    total_abs_diff = 2.0 * np.sum(xs * (2*i - n - 1))
    return total_abs_diff / (n**2)

def gini_signed(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    mabs = np.mean(np.abs(x))
    if mabs == 0:
        return 0.0
    g  = _mad_fast(x) / (2.0 * mabs)
    return float(g)

def skewness_sample(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    n = x.size
    if n < 3: return float("nan")
    m = x.mean()
    s = x.std(ddof=1)
    if s == 0: return 0.0
    g1 = np.mean(((x - m) / s) ** 3)
    return float(np.sqrt(n*(n-1)) / (n-2) * g1)

def coeff_variation_signed_or_abs(x: np.ndarray, mode: str = "abs") -> float:
    if mode == "signed":
        mu = x.mean()
        sd = x.std(ddof=1)
        return float("nan") if np.isclose(mu, 0.0) else float(abs(sd / mu))
    ax = np.abs(x)
    mu = ax.mean()
    sd = ax.std(ddof=1)
    return float("nan") if np.isclose(mu, 0.0) else float(sd / mu)

def spearman_r(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0: return float("nan")
    def _rank(x):
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(x)+1)
        u, first = np.unique(x[order], return_index=True)
        for i in range(len(u)):
            s = first[i]
            e = first[i+1] if i+1 < len(u) else len(x)
            ranks[s:e] = ranks[s:e].mean()
        out = np.empty_like(ranks)
        out[order] = ranks
        return out
    ra, rb = _rank(a), _rank(b)
    r = np.corrcoef(ra, rb)[0, 1]
    return float(r)

# ============== naming (plain vs transformer blocks) ==============

def layer_name(layer_idx: int, transformer: bool, mod: int = 4) -> str:
    if not transformer:
        return f"layer_{layer_idx:03d}"
    block = layer_idx // mod + 1
    inside = layer_idx % mod + 1
    return f"block_{block:03d}_layer_{inside:02d}"

# ============== streaming epoch tracker (unchanged) ==============

class StreamingConductanceEpochTracker:
    def __init__(self,
                 output_dir: str,
                 transformer: bool = False,
                 block_mod: int = 4,
                 cv_mode: str = "abs",
                 sign_eps: float = 0.0):
        self.output_dir = output_dir
        self.transformer = transformer
        self.block_mod = block_mod
        self.cv_mode = cv_mode
        self.sign_eps = float(sign_eps)
        self._epoch: Optional[int] = None
        self._state: Dict[int, Dict[str, np.ndarray]] = {}
        self._counts: Dict[int, int] = {}

    def begin_epoch(self, epoch: int):
        if self._epoch is not None:
            raise RuntimeError("end_epoch() before starting a new epoch.")
        self._epoch = int(epoch)
        self._state.clear()
        self._counts.clear()

    def _init_layer(self, layer: int, v: np.ndarray):
        n = v.size
        self._state[layer] = {
            "mean": v.copy(),
            "M2": np.zeros(n, dtype=np.float32),
            "last": v.copy(),
            "sum_abs_diff": np.zeros(n, np.float32),
            "max_abs_jump": np.zeros(n, np.float32),
            "sign_flips": np.zeros(n, np.int32),
            "min": v.copy(),
            "max": v.copy(),
        }
        self._counts[layer] = 1

    def update(self, new_scores: Dict[int, torch.Tensor]):
        if self._epoch is None:
            raise RuntimeError("Call begin_epoch(epoch) first.")
        for layer, t in new_scores.items():
            v = _to_1d_numpy(t)
            if layer not in self._state:
                self._init_layer(layer, v)
                continue
            st = self._state[layer]
            cnt = self._counts[layer] + 1

            mean_prev = st["mean"]
            delta = v - mean_prev
            mean_new = mean_prev + delta / cnt
            st["M2"] += delta * (v - mean_new)
            st["mean"] = mean_new

            jump = np.abs(v - st["last"])
            st["sum_abs_diff"] += jump
            st["max_abs_jump"] = np.maximum(st["max_abs_jump"], jump)

            last_eff = st["last"].copy()
            v_eff = v.copy()
            last_eff[np.abs(last_eff) <= self.sign_eps] = 0.0
            v_eff[np.abs(v_eff) <= self.sign_eps] = 0.0
            st["sign_flips"] += (np.sign(last_eff) != np.sign(v_eff)).astype(np.int32)

            st["min"] = np.minimum(st["min"], v)
            st["max"] = np.maximum(st["max"], v)

            st["last"] = v
            self._counts[layer] = cnt

    def end_epoch(self):
        if self._epoch is None:
            raise RuntimeError("No active epoch.")
        ep_dir = os.path.join(self.output_dir, f"epoch_{self._epoch:04d}")
        _ensure_dir(ep_dir)

        manifest = {
            "epoch": self._epoch,
            "transformer": self.transformer,
            "block_mod": self.block_mod,
            "layers": [],
        }

        for layer, st in sorted(self._state.items()):
            name = layer_name(layer, self.transformer, self.block_mod)
            lay_dir = os.path.join(ep_dir, name)
            _ensure_dir(lay_dir)
            cnt = self._counts[layer]
            mean = st["mean"]
            var = st["M2"] / max(cnt - 1, 1)
            mean_abs_diff = st["sum_abs_diff"] / max(cnt - 1, 1)

            np.save(os.path.join(lay_dir, "per_neuron_mean.npy"), mean)
            np.save(os.path.join(lay_dir, "per_neuron_var.npy"), var)
            np.save(os.path.join(lay_dir, "per_neuron_mean_abs_diff.npy"), mean_abs_diff)
            np.save(os.path.join(lay_dir, "per_neuron_max_abs_jump.npy"), st["max_abs_jump"])
            np.save(os.path.join(lay_dir, "per_neuron_sign_flips.npy"), st["sign_flips"])
            np.save(os.path.join(lay_dir, "per_neuron_min.npy"), st["min"])
            np.save(os.path.join(lay_dir, "per_neuron_max.npy"), st["max"])

            manifest["layers"].append({
                "layer_index": layer,
                "name": name,
                "count_iterations": cnt,
                "n_neurons": int(mean.size),
            })

        with open(os.path.join(ep_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        self._epoch = None
        self._state.clear()
        self._counts.clear()

# ============== enhanced statistics ==============

def _basic_stats(x: np.ndarray, cv_mode: str) -> Dict[str, float]:
    if x.size == 0:
        return {k: float("nan") for k in ["mean", "median", "variance", "std", "IQR", 
                "CV", "p5", "p25", "p75", "p95", "min", "max", "skewness", "gini", "n"]}
    
    q25, q50, q75 = np.quantile(x, [0.25, 0.5, 0.75])
    p5, p95 = np.percentile(x, [5, 95])
    cv = coeff_variation_signed_or_abs(x, cv_mode)
    skew = skewness_sample(x)
    g = gini_signed(x)
    
    return {
        "mean": float(np.mean(x)),
        "median": float(q50),
        "variance": float(np.var(x, ddof=1)) if x.size > 1 else 0.0,
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "IQR": float(q75 - q25),
        "CV": float(cv),
        "p5": float(p5),
        "p25": float(q25),
        "p75": float(q75),
        "p95": float(p95),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "skewness": float(skew),
        "gini": float(g),
        "n": int(x.size),
    }

def _ks_distance(x: np.ndarray, y: np.ndarray) -> float:
    xs = np.sort(x); ys = np.sort(y)
    nx, ny = xs.size, ys.size
    i = j = 0
    cdf_x = cdf_y = 0.0
    d = 0.0
    while i < nx and j < ny:
        if xs[i] <= ys[j]:
            cdf_x = (i+1)/nx
            d = max(d, abs(cdf_x - cdf_y))
            i += 1
        else:
            cdf_y = (j+1)/ny
            d = max(d, abs(cdf_x - cdf_y))
            j += 1
    while i < nx:
        cdf_x = (i+1)/nx
        d = max(d, abs(cdf_x - cdf_y))
        i += 1
    while j < ny:
        cdf_y = (j+1)/ny
        d = max(d, abs(cdf_x - cdf_y))
        j += 1
    return float(d)

def _wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    xs = np.sort(x).astype(np.float64); ys = np.sort(y).astype(np.float64)
    nx, ny = xs.size, ys.size
    if nx == 0 or ny == 0: return np.nan
    i = j = 0
    wx, wy = 1.0/nx, 1.0/ny
    rx = wx; ry = wy
    dist = 0.0
    while i < nx and j < ny:
        moved = min(rx, ry)
        dist += moved * abs(xs[i] - ys[j])
        rx -= moved; ry -= moved
        if rx <= 1e-18: i += 1; rx = wx
        if ry <= 1e-18: j += 1; ry = wy
    return float(dist)

# ============== ENHANCED PLOTTING FUNCTIONS ==============

def _plot_comprehensive_comparison(arrays: List[np.ndarray],
                                   labels: List[str],
                                   out_png: str,
                                   layer_name: str,
                                   epochs: List[int],
                                   bins: int = 200):
    """
    Create a comprehensive multi-panel comparison plot.
    """
    n_runs = len(arrays)
    arrays = [a for a in arrays if a is not None and a.size > 0]
    if not arrays:
        return
    
    # Use default colors for consistency
    colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_runs]
    
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Histogram overlay
    ax1 = fig.add_subplot(gs[0, 0])
    lo = float(min(a.min() for a in arrays))
    hi = float(max(a.max() for a in arrays))
    edges = np.linspace(lo, hi, bins+1)
    for k, (a, lab) in enumerate(zip(arrays, labels)):
        ax1.hist(a, bins=edges, alpha=0.6, label=f"{lab} (ep{epochs[k]})", 
                color=colors[k], edgecolor='black', linewidth=0.5)
    ax1.set_xlabel("Loss Conductance (lower is better)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Count", fontsize=11, fontweight='bold')
    ax1.set_title("Distribution Comparison", fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. CDF comparison
    ax2 = fig.add_subplot(gs[0, 1])
    for k, (a, lab) in enumerate(zip(arrays, labels)):
        sorted_a = np.sort(a)
        cdf = np.arange(1, len(sorted_a) + 1) / len(sorted_a)
        ax2.plot(sorted_a, cdf, label=f"{lab}", color=colors[k], linewidth=2)
    ax2.set_xlabel("Loss Conductance", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Cumulative Probability", fontsize=11, fontweight='bold')
    ax2.set_title("Cumulative Distribution Functions", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot comparison
    ax3 = fig.add_subplot(gs[0, 2])
    bp = ax3.boxplot(arrays, labels=labels, patch_artist=True, 
                     showfliers=False, widths=0.6)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax3.set_ylabel("Loss Conductance", fontsize=11, fontweight='bold')
    ax3.set_title("Distribution Summary (Box Plot)", fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Ranked values (gap from minimum)
    ax4 = fig.add_subplot(gs[1, 0])
    nmin = min(a.size for a in arrays)
    idx = np.arange(nmin)
    for k, (a, lab) in enumerate(zip(arrays, labels)):
        sorted_a = np.sort(a.ravel())[:nmin]
        gap = -sorted_a  # Negative because lower is better
        ax4.plot(idx, gap, label=f"{lab}", color=colors[k], linewidth=2, alpha=0.8)
    ax4.set_xlabel("Ranked Neuron (0=best)", fontsize=11, fontweight='bold')
    ax4.set_ylabel("Loss Conductance (inverted)", fontsize=11, fontweight='bold')
    ax4.set_title("Ranked Neuron Performance", fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 5. Percentile comparison
    ax5 = fig.add_subplot(gs[1, 1])
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    x_pos = np.arange(len(percentiles))
    width = 0.8 / n_runs
    for k, (a, lab) in enumerate(zip(arrays, labels)):
        vals = np.percentile(a, percentiles)
        offset = (k - (n_runs-1)/2) * width
        ax5.bar(x_pos + offset, vals, width, label=f"{lab}", 
               color=colors[k], alpha=0.7, edgecolor='black', linewidth=0.5)
    ax5.set_xlabel("Percentile", fontsize=11, fontweight='bold')
    ax5.set_ylabel("Loss Conductance", fontsize=11, fontweight='bold')
    ax5.set_title("Percentile Comparison", fontsize=12, fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels([f"p{p}" for p in percentiles])
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Statistical summary table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    stats_data = []
    headers = ["Metric"] + labels
    metrics = ["Mean", "Median", "Std", "Min", "Max", "p5", "p95"]
    
    for metric in metrics:
        row = [metric]
        for a in arrays:
            if metric == "Mean":
                val = np.mean(a)
            elif metric == "Median":
                val = np.median(a)
            elif metric == "Std":
                val = np.std(a, ddof=1)
            elif metric == "Min":
                val = np.min(a)
            elif metric == "Max":
                val = np.max(a)
            elif metric == "p5":
                val = np.percentile(a, 5)
            elif metric == "p95":
                val = np.percentile(a, 95)
            row.append(f"{val:.4f}")
        stats_data.append(row)
    
    table = ax6.table(cellText=stats_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     colWidths=[0.15] + [0.15]*n_runs)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax6.set_title("Statistical Summary", fontsize=12, fontweight='bold', pad=20)
    
    # 7. Top-k worst neurons comparison
    ax7 = fig.add_subplot(gs[2, 0])
    k_worst = min(50, nmin)
    x = np.arange(k_worst)
    for i, (a, lab) in enumerate(zip(arrays, labels)):
        worst = np.sort(a.ravel())[::-1][:k_worst]  # Highest (worst) values
        ax7.plot(x, worst, marker='o', markersize=3, label=f"{lab}", 
                color=colors[i], linewidth=2, alpha=0.8)
    ax7.set_xlabel(f"Rank (worst neurons)", fontsize=11, fontweight='bold')
    ax7.set_ylabel("Loss Conductance", fontsize=11, fontweight='bold')
    ax7.set_title(f"Top-{k_worst} Worst Performing Neurons", fontsize=12, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    
    # 8. Violin plot
    ax8 = fig.add_subplot(gs[2, 1])
    parts = ax8.violinplot(arrays, positions=range(n_runs), showmeans=True, 
                          showmedians=True, widths=0.7)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
    ax8.set_xticks(range(n_runs))
    ax8.set_xticklabels(labels)
    ax8.set_ylabel("Loss Conductance", fontsize=11, fontweight='bold')
    ax8.set_title("Distribution Shape (Violin Plot)", fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Difference plot (if 2 runs)
    ax9 = fig.add_subplot(gs[2, 2])
    if n_runs == 2:
        nmin = min(arrays[0].size, arrays[1].size)
        diff = arrays[0][:nmin] - arrays[1][:nmin]
        sorted_diff = np.sort(diff)
        
        colors_diff = ['green' if d < 0 else 'red' for d in sorted_diff]
        ax9.barh(np.arange(nmin), sorted_diff, color=colors_diff, alpha=0.6, 
                edgecolor='black', linewidth=0.3)
        ax9.axvline(x=0, color='black', linestyle='--', linewidth=2)
        ax9.set_xlabel(f"{labels[0]} - {labels[1]}\n(negative=better for {labels[0]})", 
                      fontsize=10, fontweight='bold')
        ax9.set_ylabel("Neuron (sorted by difference)", fontsize=11, fontweight='bold')
        ax9.set_title("Per-Neuron Difference", fontsize=12, fontweight='bold')
        ax9.grid(True, alpha=0.3, axis='x')
        
        # Add statistics
        better_count = np.sum(diff < 0)
        worse_count = np.sum(diff > 0)
        mean_diff = np.mean(diff)
        ax9.text(0.02, 0.98, 
                f"{labels[0]} better: {better_count}/{nmin}\n" +
                f"{labels[0]} worse: {worse_count}/{nmin}\n" +
                f"Mean diff: {mean_diff:.4f}",
                transform=ax9.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        # Show pairwise distance matrix
        dist_matrix = np.zeros((n_runs, n_runs))
        for i in range(n_runs):
            for j in range(n_runs):
                if i != j:
                    nmin = min(arrays[i].size, arrays[j].size)
                    dist_matrix[i, j] = _wasserstein_1d(arrays[i][:nmin], arrays[j][:nmin])
        
        im = ax9.imshow(dist_matrix, cmap='YlOrRd', aspect='auto')
        ax9.set_xticks(range(n_runs))
        ax9.set_yticks(range(n_runs))
        ax9.set_xticklabels(labels, rotation=45, ha='right')
        ax9.set_yticklabels(labels)
        ax9.set_title("Wasserstein Distance Matrix", fontsize=12, fontweight='bold')
        
        # Add text annotations
        for i in range(n_runs):
            for j in range(n_runs):
                text = ax9.text(j, i, f'{dist_matrix[i, j]:.3f}',
                              ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax9, label="W1 Distance")
    
    fig.suptitle(f"{layer_name} - Comprehensive Comparison (Lower is Better)", 
                fontsize=14, fontweight='bold', y=0.995)
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()

def _plot_difference_analysis(arr1: np.ndarray, arr2: np.ndarray,
                              label1: str, label2: str,
                              out_png: str, layer_name: str,
                              bins: int = 100):
    """
    Detailed difference analysis for two-run comparison.
    """
    nmin = min(arr1.size, arr2.size)
    a1 = arr1[:nmin]
    a2 = arr2[:nmin]
    diff = a1 - a2  # Positive means arr1 is worse
    
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Use consistent colors
    color1 = plt.cm.tab10(0)
    color2 = plt.cm.tab10(1)
    
    # 1. Difference histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(diff, bins=bins, alpha=0.7, color='purple', edgecolor='black', linewidth=0.5)
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=2, label='No difference')
    ax1.axvline(x=np.mean(diff), color='red', linestyle='-', linewidth=2, label=f'Mean: {np.mean(diff):.4f}')
    ax1.axvline(x=np.median(diff), color='orange', linestyle='-', linewidth=2, label=f'Median: {np.median(diff):.4f}')
    ax1.set_xlabel(f"Difference ({label1} - {label2})", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Count", fontsize=11, fontweight='bold')
    ax1.set_title("Difference Distribution", fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Add text with improvement statistics
    better = np.sum(diff < 0)
    worse = np.sum(diff > 0)
    same = np.sum(diff == 0)
    ax1.text(0.02, 0.98, 
            f"{label1} better: {better} ({100*better/nmin:.1f}%)\n" +
            f"{label1} worse: {worse} ({100*worse/nmin:.1f}%)\n" +
            f"Same: {same} ({100*same/nmin:.1f}%)",
            transform=ax1.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # 2. Scatter plot with diagonal
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(a2, a1, alpha=0.5, s=10, c='blue', edgecolors='none')
    lim_min = min(a1.min(), a2.min())
    lim_max = max(a1.max(), a2.max())
    ax2.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2, label='y=x')
    ax2.set_xlabel(f"{label2} conductance", fontsize=11, fontweight='bold')
    ax2.set_ylabel(f"{label1} conductance", fontsize=11, fontweight='bold')
    ax2.set_title("Per-Neuron Scatter (below line = better for " + label1 + ")", 
                 fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # 3. Bland-Altman plot
    ax3 = fig.add_subplot(gs[0, 2])
    mean_vals = (a1 + a2) / 2
    ax3.scatter(mean_vals, diff, alpha=0.5, s=10, c='green', edgecolors='none')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax3.axhline(y=np.mean(diff), color='red', linestyle='--', linewidth=2, 
               label=f'Mean diff: {np.mean(diff):.4f}')
    std_diff = np.std(diff, ddof=1)
    ax3.axhline(y=np.mean(diff) + 1.96*std_diff, color='orange', linestyle=':', 
               linewidth=2, label=f'+1.96 SD')
    ax3.axhline(y=np.mean(diff) - 1.96*std_diff, color='orange', linestyle=':', 
               linewidth=2, label=f'-1.96 SD')
    ax3.set_xlabel("Mean conductance", fontsize=11, fontweight='bold')
    ax3.set_ylabel(f"Difference ({label1} - {label2})", fontsize=11, fontweight='bold')
    ax3.set_title("Bland-Altman Plot", fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Sorted difference plot
    ax4 = fig.add_subplot(gs[1, 0])
    sorted_idx = np.argsort(diff)
    sorted_diff = diff[sorted_idx]
    colors_bar = ['green' if d < 0 else 'red' for d in sorted_diff]
    ax4.bar(np.arange(nmin), sorted_diff, color=colors_bar, alpha=0.6, 
           edgecolor='black', linewidth=0.2)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax4.set_xlabel("Neuron (sorted by difference)", fontsize=11, fontweight='bold')
    ax4.set_ylabel(f"Difference ({label1} - {label2})", fontsize=11, fontweight='bold')
    ax4.set_title(f"Sorted Differences (green={label1} better)", fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Quantile-quantile plot
    ax5 = fig.add_subplot(gs[1, 1])
    sorted_a1 = np.sort(a1)
    sorted_a2 = np.sort(a2)
    ax5.scatter(sorted_a2, sorted_a1, alpha=0.5, s=10, c='purple', edgecolors='none')
    ax5.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2, label='y=x')
    ax5.set_xlabel(f"{label2} quantiles", fontsize=11, fontweight='bold')
    ax5.set_ylabel(f"{label1} quantiles", fontsize=11, fontweight='bold')
    ax5.set_title("Q-Q Plot", fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal')
    
    # 6. Summary statistics table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    stats_data = [
        ["Mean difference", f"{np.mean(diff):.6f}"],
        ["Median difference", f"{np.median(diff):.6f}"],
        ["Std difference", f"{np.std(diff, ddof=1):.6f}"],
        ["Min difference", f"{np.min(diff):.6f}"],
        ["Max difference", f"{np.max(diff):.6f}"],
        ["", ""],
        [f"{label1} better count", f"{better} ({100*better/nmin:.1f}%)"],
        [f"{label1} worse count", f"{worse} ({100*worse/nmin:.1f}%)"],
        ["", ""],
        ["KS distance", f"{_ks_distance(a1, a2):.6f}"],
        ["Wasserstein distance", f"{_wasserstein_1d(a1, a2):.6f}"],
        ["Spearman correlation", f"{spearman_r(a1, a2):.6f}"],
    ]
    
    table = ax6.table(cellText=stats_data, cellLoc='left', loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Color alternating rows
    for i in range(len(stats_data)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax6.set_title("Detailed Statistics", fontsize=12, fontweight='bold', pad=20)
    
    fig.suptitle(f"{layer_name} - Detailed Difference Analysis\n" + 
                f"(Negative difference = {label1} is better)", 
                fontsize=14, fontweight='bold', y=0.995)
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()

# ============== IO helpers ==============

def _layer_dirs(epoch_dir: str) -> Dict[str, str]:
    out = {}
    for name in os.listdir(epoch_dir):
        if name.startswith("layer_") or name.startswith("block_"):
            p = os.path.join(epoch_dir, name)
            if os.path.isdir(p): out[name] = p
    return out

def _read_layer_means_last_epoch(run_dir: str) -> Tuple[int, Dict[str, np.ndarray]]:
    """
    Return (last_epoch_index, {layer_name: mean_vector}) for the run.
    """
    epoch_dirs = _load_manifest_epochs(run_dir)
    if not epoch_dirs:
        return -1, {}
    ep_idx = max(int(d.split("_")[1]) for d in epoch_dirs)
    ep_name = f"epoch_{ep_idx:04d}"
    ep_dir = os.path.join(run_dir, ep_name)

    man_path = os.path.join(ep_dir, "manifest.json")
    if os.path.exists(man_path):
        with open(man_path, "r") as f:
            manifest = json.load(f)
        layers = [l["name"] for l in manifest.get("layers", [])]
        lay_dirs = {nm: os.path.join(ep_dir, nm) for nm in layers}
    else:
        lay_dirs = _layer_dirs(ep_dir)

    layer_means: Dict[str, np.ndarray] = {}
    for name, ldir in lay_dirs.items():
        mp = os.path.join(ldir, "per_neuron_mean.npy")
        if os.path.exists(mp):
            layer_means[name] = np.load(mp)
    return ep_idx, layer_means

# ============== LAST-EPOCH-ONLY COMPARATOR ==============

def compare_runs_last_only(run_dirs: List[str],
                           out_dir: str,
                           labels: Optional[List[str]] = None,
                           bins: int = 200,
                           cv_mode: str = "abs"):
    _ensure_dir(out_dir)
    assert len(run_dirs) >= 1
    if labels is None or len(labels) == 0:
        labels = [f"R{i+1}" for i in range(len(run_dirs))]
    assert len(labels) == len(run_dirs)

    # Read last-epoch means from all runs
    per_run_epoch: List[int] = []
    per_run_layers: List[Dict[str, np.ndarray]] = []
    for rd in run_dirs:
        ep_idx, means = _read_layer_means_last_epoch(rd)
        if ep_idx < 0 or not means:
            print(f"[WARN] No epoch data found in: {rd}", file=sys.stderr)
        per_run_epoch.append(ep_idx)
        per_run_layers.append(means)

    # Intersect common layers
    common_layers = sorted(set.intersection(*(set(d.keys()) for d in per_run_layers if d)))
    if not common_layers:
        raise RuntimeError("No common layers across runs at their last epochs.")

    last_log: List[str] = []

    # Per-layer comprehensive analysis
    for layer in common_layers:
        layer_dir = os.path.join(out_dir, layer)
        _ensure_dir(layer_dir)

        arrays = [d[layer] for d in per_run_layers]
        
        # Create comprehensive comparison plot
        _plot_comprehensive_comparison(
            arrays, labels,
            os.path.join(layer_dir, "comprehensive_comparison.png"),
            layer, per_run_epoch, bins
        )
        
        # If exactly 2 runs, create detailed difference analysis
        if len(arrays) == 2:
            _plot_difference_analysis(
                arrays[0], arrays[1],
                labels[0], labels[1],
                os.path.join(layer_dir, "difference_analysis.png"),
                layer, bins
            )

        # Compute and log detailed statistics
        last_log.append(f"\n{'='*80}\n")
        last_log.append(f"LAYER: {layer}\n")
        last_log.append(f"{'='*80}\n\n")
        
        for lab, arr, ep in zip(labels, arrays, per_run_epoch):
            s = _basic_stats(arr, cv_mode=cv_mode)
            last_log.append(f"[{lab}] @ epoch {ep}\n")
            last_log.append(f"  Neurons: {s['n']}\n")
            last_log.append(f"  Mean:    {s['mean']:.8f}  |  Median: {s['median']:.8f}\n")
            last_log.append(f"  Std:     {s['std']:.8f}  |  Var:    {s['variance']:.8f}\n")
            last_log.append(f"  Min:     {s['min']:.8f}  |  Max:    {s['max']:.8f}\n")
            last_log.append(f"  p5:      {s['p5']:.8f}  |  p95:    {s['p95']:.8f}\n")
            last_log.append(f"  IQR:     {s['IQR']:.8f}  |  CV:     {s['CV']:.8f}\n")
            last_log.append(f"  Skew:    {s['skewness']:.8f}  |  Gini:   {s['gini']:.8f}\n")
            last_log.append("\n")

        # Pairwise comparisons
        if len(labels) > 1:
            last_log.append("PAIRWISE COMPARISONS:\n")
            last_log.append("-" * 80 + "\n")
            n = len(labels)
            for i in range(n):
                for j in range(i+1, n):
                    ai, bj = arrays[i], arrays[j]
                    nmin = min(ai.size, bj.size)
                    ks = _ks_distance(ai, bj)
                    w1 = _wasserstein_1d(ai, bj)
                    rho = spearman_r(ai[:nmin], bj[:nmin])
                    
                    # Additional metrics
                    mean_diff = np.mean(ai[:nmin] - bj[:nmin])
                    median_diff = np.median(ai[:nmin] - bj[:nmin])
                    better_i = np.sum((ai[:nmin] - bj[:nmin]) < 0)
                    better_j = np.sum((ai[:nmin] - bj[:nmin]) > 0)
                    
                    last_log.append(f"\n{labels[i]} vs {labels[j]}:\n")
                    last_log.append(f"  KS distance:        {ks:.8f}\n")
                    last_log.append(f"  Wasserstein (W1):   {w1:.8f}\n")
                    last_log.append(f"  Spearman ρ:         {rho:.8f}\n")
                    last_log.append(f"  Mean difference:    {mean_diff:.8f}  ({labels[i]} - {labels[j]})\n")
                    last_log.append(f"  Median difference:  {median_diff:.8f}\n")
                    last_log.append(f"  {labels[i]} better: {better_i}/{nmin} ({100*better_i/nmin:.1f}%)\n")
                    last_log.append(f"  {labels[j]} better: {better_j}/{nmin} ({100*better_j/nmin:.1f}%)\n")

    # Write comprehensive log file
    with open(os.path.join(out_dir, "comparison_last_epochs.txt"), "w") as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE CONDUCTANCE COMPARISON - LAST EPOCHS ONLY\n")
        f.write("="*80 + "\n")
        f.write(f"Number of runs: {len(run_dirs)}\n")
        f.write(f"Labels: {', '.join(labels)}\n")
        f.write(f"Epochs: {', '.join(f'{lab}={ep}' for lab, ep in zip(labels, per_run_epoch))}\n")
        f.write(f"Common layers: {len(common_layers)}\n")
        f.write("="*80 + "\n\n")
        f.write("NOTE: Lower conductance values indicate better performance\n")
        f.write("      (less loss when removing the neuron)\n\n")
        f.writelines(last_log)
    
    # Create summary CSV
    import csv
    summary_csv = os.path.join(out_dir, "summary_statistics.csv")
    with open(summary_csv, 'w', newline='') as csvfile:
        fieldnames = ['Layer', 'Run', 'Epoch', 'Mean', 'Median', 'Std', 'Min', 'Max', 
                     'p5', 'p95', 'IQR', 'CV', 'Skewness', 'Gini']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for layer in common_layers:
            arrays = [d[layer] for d in per_run_layers]
            for lab, arr, ep in zip(labels, arrays, per_run_epoch):
                s = _basic_stats(arr, cv_mode=cv_mode)
                writer.writerow({
                    'Layer': layer,
                    'Run': lab,
                    'Epoch': ep,
                    'Mean': f"{s['mean']:.8f}",
                    'Median': f"{s['median']:.8f}",
                    'Std': f"{s['std']:.8f}",
                    'Min': f"{s['min']:.8f}",
                    'Max': f"{s['max']:.8f}",
                    'p5': f"{s['p5']:.8f}",
                    'p95': f"{s['p95']:.8f}",
                    'IQR': f"{s['IQR']:.8f}",
                    'CV': f"{s['CV']:.8f}",
                    'Skewness': f"{s['skewness']:.8f}",
                    'Gini': f"{s['gini']:.8f}",
                })

# ============== CLI ==============

import argparse

def _die(msg: str, code: int = 2):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Compare conductance runs (LAST EPOCH ONLY - ENHANCED)")
    p.add_argument("--runs", nargs="+", required=True,
                   help="One or more stats dirs (each has epoch_XXXX folders)")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Optional labels for runs (same length as --runs)")
    p.add_argument("--out",  required=True, help="Output directory for plots & logs")
    p.add_argument("--bins", type=int, default=200, help="Bins for hist overlays")
    p.add_argument("--cv-mode", choices=["abs", "signed"], default="abs",
                   help="CV definition: abs -> std(|x|)/mean(|x|), signed -> std/|mean|")
    return p.parse_args()

def main():
    args = parse_args()

    # sanity checks
    for i, path in enumerate(args.runs):
        if not os.path.isdir(path):
            _die(f"Run #{i+1} path does not exist or is not a directory: {path}")
        has_epochs = any(d.startswith("epoch_") for d in os.listdir(path))
        if not has_epochs:
            print(f"[WARN] Run #{i+1} has no epoch_* folders: {path}", file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)

    compare_runs_last_only(
        run_dirs=args.runs,
        out_dir=args.out,
        labels=args.labels,
        bins=args.bins,
        cv_mode=args.cv_mode,
    )

    print(f"[OK] Enhanced last-epoch comparison complete. Outputs in: {args.out}")
    print(f"     - comprehensive_comparison.png: 9-panel overview for each layer")
    print(f"     - difference_analysis.png: detailed 2-run comparison (if applicable)")
    print(f"     - comparison_last_epochs.txt: detailed text statistics")
    print(f"     - summary_statistics.csv: tabular summary")

if __name__ == "__main__":
    main()