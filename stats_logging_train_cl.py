from __future__ import annotations
import os, json, math
from typing import Dict, Optional, Tuple, List
import numpy as np
import torch
import matplotlib.pyplot as plt

import argparse
import os
import sys
# ============== small math helpers ==============

def _to_1d_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().reshape(-1).cpu().numpy().astype(np.float32)

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _load_manifest_epochs(root: str) -> List[str]:
    return sorted([d for d in os.listdir(root) if d.startswith("epoch_")])
def _mad_fast(x):
    """
    Mean absolute difference (MAD) = average of |xi - xj|.
    Computed in O(n log n) using a sorted-sum identity.
    """
    n = x.size
    if n == 0:
        return float("nan")
    xs = np.sort(x)
    i = np.arange(1, n + 1)
    total_abs_diff = 2.0 * np.sum(xs * (2*i - n - 1))
    return total_abs_diff / (n**2)

def gini_signed(x: np.ndarray) -> float:
    # =0 perfectly equal, # =1 one neuron dominates
    x = np.asarray(x, dtype=np.float64).ravel()
    mabs = np.mean(np.abs(x))
    if mabs == 0:
        return 0.0
    g  = _mad_fast(x) / (2.0 * mabs)

    return float(g)

def skewness_sample(x: np.ndarray) -> float:
    # <0 very few large values, >0 very few small values, =0 symmetric
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
    # numpy fallback (average ranks)
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

# ============== streaming epoch tracker ==============

class StreamingConductanceEpochTracker:
    """
    Online (streaming) tracker:
      - Call `begin_epoch(epoch)`
      - Every time you compute conductance (dict[layer]->tensor), call `update(new_scores)`
      - Call `end_epoch()` to save per-layer arrays/statistics in output_dir/epoch_xxxx/
    It stores only running state; no per-iteration history.
    """
    def __init__(self,
                 output_dir: str,
                 transformer: bool = False,
                 block_mod: int = 4,
                 cv_mode: str = "abs",
                 sign_eps: float = 0.0):
        """
        output_dir : root folder where epoch folders will be created
        transformer: if True, layers are named as block_i/layer_j with modulo `block_mod`
        cv_mode    : 'abs' (default) uses CV(|x|); 'signed' uses std/mean (undefined near 0)
        sign_eps   : values with |x|<=sign_eps are considered 0 for sign-flip counting
        """
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
            "mean": v.copy(),                     # running mean per neuron
            "M2": np.zeros(n, dtype=np.float32),  # for running variance
            "last": v.copy(),                     # last value per neuron
            "sum_abs_diff": np.zeros(n, np.float32),  # Σ|Δ|
            "max_abs_jump": np.zeros(n, np.float32),  # max_t |Δ|
            "sign_flips": np.zeros(n, np.int32),      # count of sign flips
            "min": v.copy(),
            "max": v.copy(),
        }
        self._counts[layer] = 1  # we have one observation already

    def update(self, new_scores: Dict[int, torch.Tensor]):
        """
        new_scores: dict[layer_idx] -> tensor with per-neuron conductance for *this iteration*.
                    Any shape is fine; it's flattened internally.
        """
        if self._epoch is None:
            raise RuntimeError("Call begin_epoch(epoch) first.")
        for layer, t in new_scores.items():
            v = _to_1d_numpy(t)
            if layer not in self._state:
                self._init_layer(layer, v)
                continue
            st = self._state[layer]
            cnt = self._counts[layer] + 1
 
            # Welford update (vectorized)
            mean_prev = st["mean"]
            delta = v - mean_prev
            mean_new = mean_prev + delta / cnt
            st["M2"] += delta * (v - mean_new)
            st["mean"] = mean_new

            # dynamics
            jump = np.abs(v - st["last"])
            st["sum_abs_diff"] += jump
            st["max_abs_jump"] = np.maximum(st["max_abs_jump"], jump)

            # sign flips (ignore tiny magnitudes)
            last_eff = st["last"].copy()
            v_eff = v.copy()
            last_eff[np.abs(last_eff) <= self.sign_eps] = 0.0
            v_eff[np.abs(v_eff) <= self.sign_eps] = 0.0
            st["sign_flips"] += (np.sign(last_eff) != np.sign(v_eff)).astype(np.int32)

            # bounds
            st["min"] = np.minimum(st["min"], v)
            st["max"] = np.maximum(st["max"], v)

            st["last"] = v
            self._counts[layer] = cnt

    def end_epoch(self):
        """
        Saves per-layer arrays & a lightweight manifest into output_dir/epoch_xxxx/.
        Per-layer files (npy):
          - per_neuron_mean.npy
          - per_neuron_var.npy       (variance across iterations within the epoch)
          - per_neuron_mean_abs_diff.npy   (mean |Δ| per iteration)
          - per_neuron_max_abs_jump.npy    (max |Δ| within epoch)
          - per_neuron_sign_flips.npy
          - per_neuron_min.npy / per_neuron_max.npy
        """
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

        # reset
        self._epoch = None
        self._state.clear()
        self._counts.clear()

# ============== reporting / logs & plots ==============

def _silent_saturated(x: np.ndarray,
                      eps: Optional[float], tau: Optional[float],
                      eps_q: float, tau_q: float) -> Tuple[float,float,float,float]:
    ax = np.abs(x)
    if eps is None: eps = float(np.quantile(ax, eps_q)) if ax.size else 0.0
    if tau is None: tau = float(np.quantile(ax, tau_q)) if ax.size else 0.0
    pct_silent = (ax < eps).mean()*100.0 if ax.size else float("nan")
    pct_sat = (ax > tau).mean()*100.0 if ax.size else float("nan")
    return pct_silent, pct_sat, eps, tau

def _basic_stats(x: np.ndarray, cv_mode: str) -> Dict[str, float]:
    q25, q75 = np.quantile(x, [0.25, 0.75]) if x.size else (float("nan"), float("nan"))
    p5, p95 = np.percentile(x, [5, 95]) if x.size else (float("nan"), float("nan"))
    cv = coeff_variation_signed_or_abs(x, cv_mode)
    skew = skewness_sample(x)
    g = gini_signed(x)
    return {
        "mean": float(np.mean(x)) if x.size else float("nan"),
        "median": float(np.median(x)) if x.size else float("nan"),
        "variance": float(np.var(x, ddof=1)) if x.size > 1 else 0.0,
        "IQR": float(q75 - q25) if x.size else float("nan"),
        "CV": float(cv),
        "p5": float(p5), "p95": float(p95),
        "skewness": float(skew),
        "gini": float(g),
        "n": int(x.size),
    }

def _ks_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sample KS distance without SciPy."""
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
    # drain remainder
    if i < nx:
        # y is at 1.0 now
        while i < nx:
            cdf_x = (i+1)/nx
            d = max(d, abs(cdf_x - cdf_y))
            i += 1
    if j < ny:
        while j < ny:
            cdf_y = (j+1)/ny
            d = max(d, abs(cdf_x - cdf_y))
            j += 1
    return float(d)

def _wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    """Exact 1D Wasserstein-1 (Earth Mover's) for two samples with equal weights."""
    xs = np.sort(x).astype(np.float64); ys = np.sort(y).astype(np.float64)
    nx, ny = xs.size, ys.size
    if nx == 0 or ny == 0: return np.nan
    wx, wy = 1.0/nx, 1.0/ny
    i = j = 0
    rx = wx; ry = wy   # remaining mass at current atoms
    dist = 0.0
    while i < nx and j < ny:
        moved = min(rx, ry)
        dist += moved * abs(xs[i] - ys[j])
        rx -= moved; ry -= moved
        if rx <= 1e-18: i += 1; rx = wx
        if ry <= 1e-18: j += 1; ry = wy
    return float(dist)

def _downsample_epochs(sorted_epochs: List[int], gap: int, include_last: bool = True) -> List[int]:
    if gap <= 1:
        return sorted_epochs[:]  # no downsampling
    selected = []
    for e in sorted_epochs:
        if e == 0 or (e % gap == 0):
            selected.append(e)
    if include_last and selected and selected[-1] != sorted_epochs[-1]:
        selected.append(sorted_epochs[-1])
    elif include_last and not selected:
        selected = [sorted_epochs[0], sorted_epochs[-1]] if sorted_epochs else []
    return sorted(set(selected))
    
def _plot_timeseries_single(metric_name: str,
                            epochs: List[int],
                            values: List[float],
                            out_png: str,
                            title: str,
                            ylabel: str = None):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, values, marker="o", linestyle="-", linewidth=2, markersize=6)
    plt.xlabel("Training Epoch", fontsize=12, fontweight='bold')
    plt.ylabel(ylabel if ylabel else metric_name, fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close()

def _plot_timeseries(metric_name: str,
                     epochs_A: List[int], values_A: List[float], label_A: str,
                     epochs_B: List[int], values_B: List[float], label_B: str,
                     out_png: str, title: str,
                     ylabel: str = None):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_A, values_A, marker="o", linestyle="-", label=label_A, linewidth=2, markersize=6)
    plt.plot(epochs_B, values_B, marker="o", linestyle="-", label=label_B, linewidth=2, markersize=6)
    plt.xlabel("Training Epoch", fontsize=12, fontweight='bold')
    plt.ylabel(ylabel if ylabel else metric_name, fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close()

def _plot_hist(x: np.ndarray, out_png: str, title: str, bins: int = 50):
    plt.figure(figsize=(10, 6))
    counts, edges = np.histogram(x, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    plt.bar(centers, counts, width=np.diff(edges), align="center")
    plt.xlabel("Conductance Value (signed)", fontsize=12, fontweight='bold')
    plt.ylabel("Neuron Count", fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close()

def _plot_values_scatter(x: np.ndarray, out_png: str, title: str):
    plt.figure(figsize=(10, 6))
    plt.scatter(np.arange(x.size), x, s=8, alpha=0.6)
    plt.xlabel("Neuron Index", fontsize=12, fontweight='bold')
    plt.ylabel("Conductance Value", fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close()

def _overlay_hist(a: np.ndarray, b: np.ndarray, out_png: str, title: str, bins: int = 50):
    lo = float(min(a.min() if a.size else 0, b.min() if b.size else 0))
    hi = float(max(a.max() if a.size else 1, b.max() if b.size else 1))
    edges = np.linspace(lo, hi, bins+1)
    ca, _ = np.histogram(a, bins=edges, density=False)
    cb, _ = np.histogram(b, bins=edges, density=False)
    centers = (edges[:-1] + edges[1:]) / 2
    width = np.diff(edges) * 0.45
    plt.figure(figsize=(10, 6))
    plt.bar(centers - width/2, ca, width=width, alpha=0.6, label="Method A")
    plt.bar(centers + width/2, cb, width=width, alpha=0.6, label="Method B")
    plt.xlabel("Conductance Value (signed)", fontsize=12, fontweight='bold')
    plt.ylabel("Neuron Count", fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close()

def _overlay_scatter(a: np.ndarray, b: np.ndarray, out_png: str, title: str):
    plt.figure(figsize=(10, 6))
    plt.scatter(np.arange(a.size), a, s=8, label="Method A", alpha=0.6)
    plt.scatter(np.arange(b.size), b, s=8, marker="x", label="Method B", alpha=0.6)
    plt.xlabel("Neuron Index", fontsize=12, fontweight='bold')
    plt.ylabel("Conductance Value", fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close()

def _overlay_maxgap(a: np.ndarray, b: np.ndarray, out_png: str, title: str):
    if a.size == 0 or b.size == 0:
        return
    sa = np.sort(a)[::-1]; sb = np.sort(b)[::-1]
    ga = sa[0]-sa; gb = sb[0] - sb
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(ga.size), ga, label="Method A", linewidth=2)
    plt.plot(np.arange(gb.size), gb, label="Method B", linewidth=2)
    plt.xlabel("Neuron Rank (0 = highest conductance)", fontsize=12, fontweight='bold')
    plt.ylabel("Gap from Maximum", fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close()

def _plot_maxgap(x: np.ndarray, out_png: str, title: str):
    if x.size == 0:
        return
    xs = np.sort(x)[::-1]
    gaps = xs[0] - xs
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(xs.size), gaps, linewidth=2)
    plt.xlabel("Neuron Rank (0 = highest conductance)", fontsize=12, fontweight='bold')
    plt.ylabel("Gap from Maximum", fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close()

def build_reports(output_dir: str,
                  transformer: bool = False,
                  block_mod: int = 4,
                  eps: Optional[float] = None,
                  tau: Optional[float] = None,
                  eps_q: float = 0.05,
                  tau_q: float = 0.98,
                  cv_mode: str = "abs",
                  bins: int = 50,
                  epoch_gap: int = 1):
    """
    Walks all epoch_* folders under output_dir, loads per-layer arrays,
    computes requested metrics, writes readable logs, and saves plots.
    """
    # ---- 1) Per-epoch reporting (downsampled) ----
    all_epoch_dirs = {int(d.split("_")[1]): d for d in os.listdir(output_dir) if d.startswith("epoch_")}
    all_epoch_idx = sorted(all_epoch_dirs.keys())
    if not all_epoch_idx:
        return

    selected_idx = _downsample_epochs(all_epoch_idx, gap=epoch_gap, include_last=True)

    prev_layer_means: Dict[str, np.ndarray] = {}

    for eidx in selected_idx:
        ep = all_epoch_dirs[eidx]
        ep_dir = os.path.join(output_dir, ep)
        try:
            with open(os.path.join(ep_dir, "manifest.json"), "r") as f:
                manifest = json.load(f)
        except FileNotFoundError:
            continue

        basic_log_lines: List[str] = []
        dyn_log_lines: List[str] = []
        cmp_log_lines: List[str] = []

        for lay in manifest["layers"]:
            name = lay["name"]
            lay_dir = os.path.join(ep_dir, name)
            mean = np.load(os.path.join(lay_dir, "per_neuron_mean.npy"))

            stats = _basic_stats(mean, cv_mode=cv_mode)
            basic_log_lines.append(
                f"[{name}] n={stats['n']}\n"
                f" mean={stats['mean']:.6f}  median={stats['median']:.6f}  variance={stats['variance']:.6f}\n"
                f" IQR={stats['IQR']:.6f}  CV={stats['CV']:.6f}  p5={stats['p5']:.6f}  p95={stats['p95']:.6f}\n"
                f" skewness={stats['skewness']:.6f}  gini(|x|)={stats['gini']:.6f}\n"
            )

            mad = np.load(os.path.join(lay_dir, "per_neuron_mean_abs_diff.npy"))
            mxj = np.load(os.path.join(lay_dir, "per_neuron_max_abs_jump.npy"))
            flips = np.load(os.path.join(lay_dir, "per_neuron_sign_flips.npy"))
            dyn_log_lines.append(
                f"[{name}] mean(|Δ|) per iter: mean={mad.mean():.6f} median={np.median(mad):.6f} "
                f"max(|Δ|) per neuron: mean={mxj.mean():.6f} max={mxj.max():.6f} "
                f"sign_flips: mean={flips.mean():.3f} max={flips.max():d}\n"
            )

            _plot_hist(mean, os.path.join(lay_dir, "histogram.png"),
                       title=f"Conductance Distribution – Epoch {eidx} – {name}", bins=bins)
            _plot_values_scatter(mean, os.path.join(lay_dir, "values_scatter.png"),
                                 title=f"Per-Neuron Conductance Values – Epoch {eidx} – {name}")
            _plot_maxgap(mean, os.path.join(lay_dir, "maxgap.png"),
                         title=f"Conductance Gap from Maximum – Epoch {eidx} – {name}")

            key = name
            if key in prev_layer_means:
                prev = prev_layer_means[key]
                n = min(prev.size, mean.size)
                diffs = np.abs(mean[:n] - prev[:n])
                avg_diff = diffs.mean()
                rho = spearman_r(mean[:n], prev[:n])
                cmp_log_lines.append(
                    f"[{name}] vs prev-selected-epoch: avg|Δ|={avg_diff:.6f}  Spearman ρ_s={rho:.4f}\n"
                )
            prev_layer_means[key] = mean

        with open(os.path.join(ep_dir, "stats_basic.txt"), "w") as f:
            f.write(f"=== BASIC STATS (1–9) – sampled every {epoch_gap} epochs ===\n")
            f.writelines(basic_log_lines)
        with open(os.path.join(ep_dir, "stats_within_epoch_dynamics.txt"), "w") as f:
            f.write("=== WITHIN-EPOCH DYNAMICS (10: largest jump max_t, 11*: mean|Δ| per-iter, sign flips) ===\n")
            f.writelines(dyn_log_lines)
        if cmp_log_lines:
            with open(os.path.join(ep_dir, "stats_epoch_compare.txt"), "w") as f:
                f.write(f"=== BETWEEN SELECTED EPOCHS (gap≈{epoch_gap}): (11 avg diff, 12 Spearman ρ_s) ===\n")
                f.writelines(cmp_log_lines)

    # ---- 2) Cross-epoch time-series (sampled & delta) ----
    layer_epoch_data: Dict[str, Dict[int, Dict[str, np.ndarray]]] = _read_layer_all_metrics(output_dir)

    ts_root = os.path.join(output_dir, "timeseries")
    _ensure_dir(ts_root)

    # Metrics from mean-based stats
    mean_metrics = ["mean", "median", "variance", "IQR", "CV", "p5", "p95", "skewness", "gini"]
    
    # Additional metrics from saved arrays (averaged over neurons)
    array_metrics = {
        "per_neuron_var": "Within-Epoch Variance",
        "per_neuron_mean_abs_diff": "Mean Absolute Change per Iteration",
        "per_neuron_max_abs_jump": "Maximum Absolute Jump",
        "per_neuron_sign_flips": "Sign Flip Count",
        "per_neuron_min": "Minimum Conductance",
        "per_neuron_max": "Maximum Conductance"
    }

    sampled_rows: List[Tuple[str, int, str, float]] = []
    delta_rows:   List[Tuple[str, int, str, float]] = []

    for layer, ep2data in layer_epoch_data.items():
        lay_ts_dir = os.path.join(ts_root, layer)
        _ensure_dir(lay_ts_dir)

        all_ep = sorted(ep2data.keys())
        ds_ep = _downsample_epochs(all_ep, gap=epoch_gap, include_last=True)

        # Precompute stats on all epochs for mean-based metrics
        stats_all: Dict[int, Dict[str, float]] = {}
        for e in all_ep:
            mean_vec = ep2data[e]["per_neuron_mean"]
            stats_all[e] = _basic_stats(mean_vec, cv_mode=cv_mode)

        # Plot mean-based metrics
        metric_labels = {
            "mean": "Average Conductance",
            "median": "Median Conductance",
            "variance": "Conductance Variance",
            "IQR": "Interquartile Range (IQR)",
            "CV": "Coefficient of Variation",
            "p5": "5th Percentile",
            "p95": "95th Percentile",
            "skewness": "Distribution Skewness",
            "gini": "Gini Coefficient"
        }
        
        for metric in mean_metrics:
            vals_sampled = [stats_all[e][metric] for e in ds_ep]
            out_png = os.path.join(lay_ts_dir, f"timeseries_{metric}_gap{epoch_gap}.png")
            _plot_timeseries_single(
                metric_name=metric,
                epochs=ds_ep,
                values=vals_sampled,
                out_png=out_png,
                title=f"{layer} – {metric_labels.get(metric, metric)} Evolution",
                ylabel=metric_labels.get(metric, metric)
            )

            for e, v in zip(ds_ep, vals_sampled):
                sampled_rows.append((layer, e, metric, v))

            # Δ over epoch_gap
            delta_ep = [e for e in ds_ep if (e - epoch_gap) in stats_all]
            delta_vals = [stats_all[e][metric] - stats_all[e - epoch_gap][metric] for e in delta_ep]
            out_png_delta = os.path.join(lay_ts_dir, f"timeseries_{metric}_delta{epoch_gap}.png")
            if delta_ep:
                _plot_timeseries_single(
                    metric_name=f"Δ{metric}",
                    epochs=delta_ep,
                    values=delta_vals,
                    out_png=out_png_delta,
                    title=f"{layer} – Change in {metric_labels.get(metric, metric)} (Δ over {epoch_gap} epochs)",
                    ylabel=f"Δ{metric_labels.get(metric, metric)}"
                )

            for e, dv in zip(delta_ep, delta_vals):
                delta_rows.append((layer, e, f"delta_{metric}", dv))

        # Plot array-based metrics (averaged over neurons)
        for array_name, metric_label in array_metrics.items():
            metric_key = array_name.replace("per_neuron_", "")
            
            # Compute averaged values for sampled epochs
            vals_sampled = []
            for e in ds_ep:
                arr = ep2data[e].get(array_name)
                if arr is not None:
                    vals_sampled.append(float(np.mean(arr)))
                else:
                    vals_sampled.append(float('nan'))
            
            out_png = os.path.join(lay_ts_dir, f"timeseries_{metric_key}_gap{epoch_gap}.png")
            _plot_timeseries_single(
                metric_name=metric_key,
                epochs=ds_ep,
                values=vals_sampled,
                out_png=out_png,
                title=f"{layer} – {metric_label} Evolution (Layer-Averaged)",
                ylabel=metric_label
            )

            for e, v in zip(ds_ep, vals_sampled):
                sampled_rows.append((layer, e, metric_key, v))

            # Δ over epoch_gap
            metric_dict = {}
            for e in all_ep:
                arr = ep2data[e].get(array_name)
                if arr is not None:
                    metric_dict[e] = float(np.mean(arr))
            
            delta_ep = [e for e in ds_ep if (e - epoch_gap) in metric_dict and e in metric_dict]
            delta_vals = [metric_dict[e] - metric_dict[e - epoch_gap] for e in delta_ep]
            out_png_delta = os.path.join(lay_ts_dir, f"timeseries_{metric_key}_delta{epoch_gap}.png")
            if delta_ep:
                _plot_timeseries_single(
                    metric_name=f"Δ{metric_key}",
                    epochs=delta_ep,
                    values=delta_vals,
                    out_png=out_png_delta,
                    title=f"{layer} – Change in {metric_label} (Δ over {epoch_gap} epochs)",
                    ylabel=f"Δ{metric_label}"
                )

            for e, dv in zip(delta_ep, delta_vals):
                delta_rows.append((layer, e, f"delta_{metric_key}", dv))

    # Write CSVs
    csv_sampled = os.path.join(ts_root, f"metrics_timeseries_gap{epoch_gap}.csv")
    with open(csv_sampled, "w") as f:
        f.write("layer,epoch,metric,value\n")
        for layer, e, metric, v in sampled_rows:
            val_str = "nan" if (v is None or (isinstance(v, float) and not np.isfinite(v))) else f"{v}"
            f.write(f"{layer},{e},{metric},{val_str}\n")

    csv_delta = os.path.join(ts_root, f"metrics_delta_gap{epoch_gap}.csv")
    with open(csv_delta, "w") as f:
        f.write("layer,epoch,metric,value\n")
        for layer, e, metric, v in delta_rows:
            val_str = "nan" if (v is None or (isinstance(v, float) and not np.isfinite(v))) else f"{v}"
            f.write(f"{layer},{e},{metric},{val_str}\n")

def _layer_dirs(epoch_dir: str) -> Dict[str, str]:
    """Return mapping layer_name -> layer_dir for an epoch dir."""
    out = {}
    for name in os.listdir(epoch_dir):
        if name.startswith("layer_") or name.startswith("block_"):
            p = os.path.join(epoch_dir, name)
            if os.path.isdir(p): out[name] = p
    return out

def _read_layer_means(run_dir: str, invert_sign: bool = False) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Returns: {layer_name: {epoch_index: mean_vector}}
    """
    result: Dict[str, Dict[int, np.ndarray]] = {}
    for ep_name in _load_manifest_epochs(run_dir):
        ep_idx = int(ep_name.split("_")[1])
        ep_dir = os.path.join(run_dir, ep_name)
        man_path = os.path.join(ep_dir, "manifest.json")
        if os.path.exists(man_path):
            with open(man_path, "r") as f:
                manifest = json.load(f)
            layers = [l["name"] for l in manifest.get("layers", [])]
            lay_dirs = {nm: os.path.join(ep_dir, nm) for nm in layers}
        else:
            lay_dirs = _layer_dirs(ep_dir)
        for name, ldir in lay_dirs.items():
            mean_path = os.path.join(ldir, "per_neuron_mean.npy")
            if not os.path.exists(mean_path): continue
            mean = np.load(mean_path)
            if invert_sign:
                mean = -mean
            result.setdefault(name, {})[ep_idx] = mean
    return result

def _read_layer_all_metrics(run_dir: str, invert_sign: bool = False) -> Dict[str, Dict[int, Dict[str, np.ndarray]]]:
    """
    Returns: {layer_name: {epoch_index: {metric_name: array}}}
    Loads all saved metrics: mean, var, mean_abs_diff, max_abs_jump, sign_flips, min, max
    """
    result: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}
    metric_files = [
        "per_neuron_mean.npy",
        "per_neuron_var.npy",
        "per_neuron_mean_abs_diff.npy",
        "per_neuron_max_abs_jump.npy",
        "per_neuron_sign_flips.npy",
        "per_neuron_min.npy",
        "per_neuron_max.npy"
    ]
    
    for ep_name in _load_manifest_epochs(run_dir):
        ep_idx = int(ep_name.split("_")[1])
        ep_dir = os.path.join(run_dir, ep_name)
        man_path = os.path.join(ep_dir, "manifest.json")
        if os.path.exists(man_path):
            with open(man_path, "r") as f:
                manifest = json.load(f)
            layers = [l["name"] for l in manifest.get("layers", [])]
            lay_dirs = {nm: os.path.join(ep_dir, nm) for nm in layers}
        else:
            lay_dirs = _layer_dirs(ep_dir)
            
        for name, ldir in lay_dirs.items():
            if name not in result:
                result[name] = {}
            if ep_idx not in result[name]:
                result[name][ep_idx] = {}
                
            for metric_file in metric_files:
                metric_path = os.path.join(ldir, metric_file)
                if os.path.exists(metric_path):
                    arr = np.load(metric_path)
                    if invert_sign and metric_file in ["per_neuron_mean.npy", "per_neuron_min.npy", "per_neuron_max.npy"]:
                        arr = -arr
                    metric_name = metric_file.replace(".npy", "")
                    result[name][ep_idx][metric_name] = arr
    
    return result

# ==== generalized comparator (N runs) + back-compat wrapper ====

def compare_runs_multi(run_dirs: List[str],
                       out_dir: str,
                       labels: Optional[List[str]] = None,
                       best_epochs: Optional[List[Optional[int]]] = None,
                       metrics_to_plot: Optional[List[str]] = None,
                       bins: int = 60,
                       cv_mode: str = "abs",
                       epoch_gap: int = 1,
                       invert_sign: bool = False):
    _ensure_dir(out_dir)
    assert len(run_dirs) >= 1
    if labels is None or len(labels) == 0:
        labels = [f"R{i+1}" for i in range(len(run_dirs))]
    assert len(labels) == len(run_dirs)
    if best_epochs is None:
        best_epochs = [None] * len(run_dirs)
    assert len(best_epochs) == len(run_dirs)

    runs_data = [_read_layer_all_metrics(rd, invert_sign=invert_sign) for rd in run_dirs] 
    layers_common = sorted(set.intersection(*(set(d.keys()) for d in runs_data)))
    if not layers_common:
        raise RuntimeError("No common layers found across runs.")

    if metrics_to_plot is None:
        metrics_to_plot = ["mean","median","variance","IQR","CV","p95","skewness","gini"]

    # Additional array-based metrics
    array_metrics = {
        "per_neuron_var": "Within-Epoch Variance",
        "per_neuron_mean_abs_diff": "Mean Absolute Change",
        "per_neuron_max_abs_jump": "Maximum Absolute Jump",
        "per_neuron_sign_flips": "Sign Flip Count",
        "per_neuron_min": "Minimum Conductance",
        "per_neuron_max": "Maximum Conductance"
    }

    last_epochs = []
    for d in runs_data:
        last_epochs.append(max({e for layer_dict in d.values() for e in layer_dict.keys()}))

    last_log = []
    best_log = []

    for layer in layers_common:
        layer_dir = os.path.join(out_dir, layer)
        _ensure_dir(layer_dir)

        # Downsampled series & deltas per run
        per_metric_series = {m: [] for m in metrics_to_plot}
        per_metric_delta  = {m: [] for m in metrics_to_plot}
        
        # Add array metrics
        for arr_name in array_metrics.keys():
            metric_key = arr_name.replace("per_neuron_", "")
            per_metric_series[metric_key] = []
            per_metric_delta[metric_key] = []

        last_arrays = []
        best_arrays = []
        last_all_metrics = []
        best_all_metrics = []

        for run_idx, (d, lab) in enumerate(zip(runs_data, labels)):
            ep_all = sorted(d[layer].keys())
            ep = _downsample_epochs(ep_all, gap=epoch_gap, include_last=True)

            # Mean-based metrics
            stats = {m: [] for m in metrics_to_plot}
            for e in ep:
                mean_vec = d[layer][e]["per_neuron_mean"]
                s = _basic_stats(mean_vec, cv_mode=cv_mode)
                for m in metrics_to_plot: stats[m].append(s[m])

            for m in metrics_to_plot:
                per_metric_series[m].append((ep, stats[m], lab))
                mp = dict(zip(ep, stats[m]))
                dEp = [t for t in ep if (t - epoch_gap) in mp]
                dVal = [mp[t] - mp[t - epoch_gap] for t in dEp]
                per_metric_delta[m].append((dEp, dVal, f"{lab} Δ{m}"))

            # Array-based metrics
            for arr_name, metric_label in array_metrics.items():
                metric_key = arr_name.replace("per_neuron_", "")
                vals = []
                for e in ep:
                    arr = d[layer][e].get(arr_name)
                    if arr is not None:
                        vals.append(float(np.mean(arr)))
                    else:
                        vals.append(float('nan'))
                
                per_metric_series[metric_key].append((ep, vals, lab))
                
                # Deltas
                metric_dict = {}
                for e in ep_all:
                    arr = d[layer][e].get(arr_name)
                    if arr is not None:
                        metric_dict[e] = float(np.mean(arr))
                
                dEp = [t for t in ep if (t - epoch_gap) in metric_dict and t in metric_dict]
                dVal = [metric_dict[t] - metric_dict[t - epoch_gap] for t in dEp]
                per_metric_delta[metric_key].append((dEp, dVal, f"{lab} Δ{metric_key}"))

            # Store last epoch data
            last_arrays.append(d[layer].get(last_epochs[run_idx], {}).get("per_neuron_mean"))
            last_all_metrics.append(d[layer].get(last_epochs[run_idx], {}))
            
            be = best_epochs[run_idx]
            best_arrays.append(d[layer].get(int(be), {}).get("per_neuron_mean") if be is not None else None)
            best_all_metrics.append(d[layer].get(int(be), {}) if be is not None else {})

        # Plot time-series & deltas for mean-based metrics
        metric_labels_plot = {
            "mean": "Average Conductance",
            "median": "Median Conductance",
            "variance": "Conductance Variance",
            "IQR": "Interquartile Range",
            "CV": "Coefficient of Variation",
            "p5": "5th Percentile",
            "p95": "95th Percentile",
            "skewness": "Distribution Skewness",
            "gini": "Gini Coefficient"
        }
        
        for m in metrics_to_plot:
            _plot_timeseries_multi(
                m, per_metric_series[m],
                os.path.join(layer_dir, f"timeseries_{m}_gap{epoch_gap}.png"),
                title=f"{layer} – {metric_labels_plot.get(m, m)} Evolution Comparison",
                ylabel=metric_labels_plot.get(m, m)
            )
            _plot_timeseries_multi(
                f"Δ{m} (over {epoch_gap})", per_metric_delta[m],
                os.path.join(layer_dir, f"timeseries_{m}_delta{epoch_gap}.png"),
                title=f"{layer} – Change in {metric_labels_plot.get(m, m)} (Δ over {epoch_gap} epochs)",
                ylabel=f"Δ{metric_labels_plot.get(m, m)}"
            )

        # Plot time-series for array-based metrics
        for arr_name, metric_label in array_metrics.items():
            metric_key = arr_name.replace("per_neuron_", "")
            _plot_timeseries_multi(
                metric_key, per_metric_series[metric_key],
                os.path.join(layer_dir, f"timeseries_{metric_key}_gap{epoch_gap}.png"),
                title=f"{layer} – {metric_label} Evolution Comparison (Layer-Averaged)",
                ylabel=metric_label
            )
            _plot_timeseries_multi(
                f"Δ{metric_key}", per_metric_delta[metric_key],
                os.path.join(layer_dir, f"timeseries_{metric_key}_delta{epoch_gap}.png"),
                title=f"{layer} – Change in {metric_label} (Δ over {epoch_gap} epochs)",
                ylabel=f"Δ{metric_label}"
            )

        # Last-epoch overlays
        if all(a is not None for a in last_arrays):
            if len(last_arrays) == 1:
                arr = last_arrays[0]
                _plot_hist(arr, os.path.join(layer_dir, "hist_last.png"),
                           title=f"{layer} – Conductance Distribution (Last Epoch: {labels[0]} @ Epoch {last_epochs[0]})", bins=bins)
                _plot_values_scatter(arr, os.path.join(layer_dir, "scatter_last.png"),
                                     title=f"{layer} – Per-Neuron Conductance (Last Epoch)")
                _plot_maxgap(arr, os.path.join(layer_dir, "maxgap_last.png"),
                             title=f"{layer} – Conductance Gap from Maximum (Last Epoch)")
                s = _basic_stats(arr, cv_mode=cv_mode)
                
                # Add array metrics to log
                metrics_dict = last_all_metrics[0]
                last_log.append(
                    f"[{layer}] LAST  {labels[0]}@{last_epochs[0]}\n"
                    f" Mean Conductance: mean={s['mean']:.6f}  median={s['median']:.6f}  var={s['variance']:.6f}\n"
                    f" Distribution: IQR={s['IQR']:.6f}  CV={s['CV']:.6f}  p95={s['p95']:.6f}\n"
                    f" Shape: skew={s['skewness']:.6f}  gini={s['gini']:.6f}\n"
                )
                for arr_name, metric_label in array_metrics.items():
                    arr_data = metrics_dict.get(arr_name)
                    if arr_data is not None:
                        last_log.append(f" {metric_label}: mean={np.mean(arr_data):.6f}  median={np.median(arr_data):.6f}  max={np.max(arr_data):.6f}\n")
                last_log.append("\n")
            else:
                _overlay_hist_multi(last_arrays, labels,
                                    os.path.join(layer_dir, "overlay_hist_last.png"),
                                    title=f"{layer} – Conductance Distribution Comparison (Last Epochs)",
                                    bins=bins)
                _overlay_scatter_multi(last_arrays, labels,
                                       os.path.join(layer_dir, "overlay_scatter_last.png"),
                                       title=f"{layer} – Per-Neuron Conductance Comparison (Last Epochs)")
                _overlay_maxgap_multi(last_arrays, labels,
                                      os.path.join(layer_dir, "overlay_maxgap_last.png"),
                                      title=f"{layer} – Conductance Gap from Maximum Comparison (Last Epochs)")

                # per-run stats including array metrics
                for lab, arr, metrics_dict in zip(labels, last_arrays, last_all_metrics):
                    s = _basic_stats(arr, cv_mode=cv_mode)
                    last_log.append(
                        f"[{layer}] LAST  {lab}@{last_epochs[labels.index(lab)]}\n"
                        f" Mean Conductance: mean={s['mean']:.6f}  median={s['median']:.6f}  var={s['variance']:.6f}\n"
                        f" Distribution: IQR={s['IQR']:.6f}  CV={s['CV']:.6f}  p95={s['p95']:.6f}\n"
                        f" Shape: skew={s['skewness']:.6f}  gini={s['gini']:.6f}\n"
                    )
                    for arr_name, metric_label in array_metrics.items():
                        arr_data = metrics_dict.get(arr_name)
                        if arr_data is not None:
                            last_log.append(f" {metric_label}: mean={np.mean(arr_data):.6f}  median={np.median(arr_data):.6f}  max={np.max(arr_data):.6f}\n")
                    last_log.append("\n")
                    
                # pairwise distances
                n = len(labels)
                for i in range(n):
                    for j in range(i+1, n):
                        ai, bj = last_arrays[i], last_arrays[j]
                        ks = _ks_distance(ai, bj)
                        w1 = _wasserstein_1d(ai, bj)
                        nmin = min(ai.size, bj.size)
                        rho = np.corrcoef(
                            np.argsort(np.argsort(ai[:nmin])),
                            np.argsort(np.argsort(bj[:nmin]))
                        )[0,1]
                        last_log.append(
                            f"[{layer}] LAST  {labels[i]} vs {labels[j]}  "
                            f"KS={ks:.4f}  W1={w1:.6f}  Spearman ρ_s≈{rho:.4f}\n"
                        )

        # Best-epoch overlays
        have_all_best = all((be is not None and arr is not None)
                            for be, arr in zip(best_epochs, best_arrays))
        if have_all_best:
            if len(best_arrays) == 1:
                arr = best_arrays[0]
                _plot_hist(arr, os.path.join(layer_dir, "hist_best.png"),
                           title=f"{layer} – Conductance Distribution (Best Epoch: {labels[0]} @ Epoch {best_epochs[0]})", bins=bins)
                _plot_values_scatter(arr, os.path.join(layer_dir, "scatter_best.png"),
                                     title=f"{layer} – Per-Neuron Conductance (Best Epoch)")
                _plot_maxgap(arr, os.path.join(layer_dir, "maxgap_best.png"),
                             title=f"{layer} – Conductance Gap from Maximum (Best Epoch)")
                s = _basic_stats(arr, cv_mode=cv_mode)
                
                metrics_dict = best_all_metrics[0]
                best_log.append(
                    f"[{layer}] BEST  {labels[0]}@{best_epochs[0]}\n"
                    f" Mean Conductance: mean={s['mean']:.6f}  median={s['median']:.6f}  var={s['variance']:.6f}\n"
                    f" Distribution: IQR={s['IQR']:.6f}  CV={s['CV']:.6f}  p95={s['p95']:.6f}\n"
                    f" Shape: skew={s['skewness']:.6f}  gini={s['gini']:.6f}\n"
                )
                for arr_name, metric_label in array_metrics.items():
                    arr_data = metrics_dict.get(arr_name)
                    if arr_data is not None:
                        best_log.append(f" {metric_label}: mean={np.mean(arr_data):.6f}  median={np.median(arr_data):.6f}  max={np.max(arr_data):.6f}\n")
                best_log.append("\n")
            else:
                _overlay_hist_multi(best_arrays, labels,
                                    os.path.join(layer_dir, "overlay_hist_best.png"),
                                    title=f"{layer} – Conductance Distribution Comparison (Best Epochs)",
                                    bins=bins)
                _overlay_scatter_multi(best_arrays, labels,
                                       os.path.join(layer_dir, "overlay_scatter_best.png"),
                                       title=f"{layer} – Per-Neuron Conductance Comparison (Best Epochs)")
                _overlay_maxgap_multi(best_arrays, labels,
                                      os.path.join(layer_dir, "overlay_maxgap_best.png"),
                                      title=f"{layer} – Conductance Gap from Maximum Comparison (Best Epochs)")

                # per-run stats + pairwise
                n = len(labels)
                for lab, arr, metrics_dict in zip(labels, best_arrays, best_all_metrics):
                    s = _basic_stats(arr, cv_mode=cv_mode)
                    best_log.append(
                        f"[{layer}] BEST  {lab}@{best_epochs[labels.index(lab)]}\n"
                        f" Mean Conductance: mean={s['mean']:.6f}  median={s['median']:.6f}  var={s['variance']:.6f}\n"
                        f" Distribution: IQR={s['IQR']:.6f}  CV={s['CV']:.6f}  p95={s['p95']:.6f}\n"
                        f" Shape: skew={s['skewness']:.6f}  gini={s['gini']:.6f}\n"
                    )
                    for arr_name, metric_label in array_metrics.items():
                        arr_data = metrics_dict.get(arr_name)
                        if arr_data is not None:
                            best_log.append(f" {metric_label}: mean={np.mean(arr_data):.6f}  median={np.median(arr_data):.6f}  max={np.max(arr_data):.6f}\n")
                    best_log.append("\n")
                    
                for i in range(n):
                    for j in range(i+1, n):
                        ai, bj = best_arrays[i], best_arrays[j]
                        ks = _ks_distance(ai, bj)
                        w1 = _wasserstein_1d(ai, bj)
                        nmin = min(ai.size, bj.size)
                        rho = np.corrcoef(
                            np.argsort(np.argsort(ai[:nmin])),
                            np.argsort(np.argsort(bj[:nmin]))
                        )[0,1]
                        best_log.append(
                            f"[{layer}] BEST  {labels[i]} vs {labels[j]}  "
                            f"KS={ks:.4f}  W1={w1:.6f}  Spearman ρ_s≈{rho:.4f}\n"
                        )

    # Write logs
    with open(os.path.join(out_dir, "comparison_last_epochs.txt"), "w") as f:
        f.write("=== COMPREHENSIVE COMPARISON @ LAST EPOCHS ===\n")
        f.write("Includes: Mean Conductance Stats, Distribution Metrics, and Training Dynamics\n\n")
        f.writelines(last_log)
    if best_log:
        with open(os.path.join(out_dir, "comparison_best_epochs.txt"), "w") as f:
            f.write("=== COMPREHENSIVE COMPARISON @ BEST EPOCHS ===\n")
            f.write("Includes: Mean Conductance Stats, Distribution Metrics, and Training Dynamics\n\n")
            f.writelines(best_log)

def compare_runs(runA_dir: str, runB_dir: str,
                 best_epoch_A: Optional[int],
                 best_epoch_B: Optional[int],
                 out_dir: str,
                 metrics_to_plot: Optional[List[str]] = None,
                 bins: int = 60,
                 cv_mode: str = "abs",
                 labelA: str = "A",
                 labelB: str = "B",
                 epoch_gap: int = 1):
    # Back-compat wrapper → identical outputs for 2 runs
    return compare_runs_multi(
        run_dirs=[runA_dir, runB_dir],
        out_dir=out_dir,
        labels=[labelA, labelB],
        best_epochs=[best_epoch_A, best_epoch_B],
        metrics_to_plot=metrics_to_plot,
        bins=bins,
        cv_mode=cv_mode,
        epoch_gap=epoch_gap,
    )

# ==== helpers (keep originals; add multi-run + wrap originals) ====

def _plot_timeseries_multi(metric_name: str,
                           series: List[Tuple[List[int], List[float], str]],
                           out_png: str,
                           title: str,
                           ylabel: str = None):
    plt.figure(figsize=(10, 6))
    for epochs, values, label in series:
        if epochs:
            plt.plot(epochs, values, marker="o", linestyle="-", label=label, linewidth=2, markersize=6)
    plt.xlabel("Training Epoch", fontsize=12, fontweight='bold')
    plt.ylabel(ylabel if ylabel else metric_name, fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    if len(series) > 1:
        plt.legend(fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close()

def _overlay_hist_multi(arrays: List[np.ndarray],
                        labels: List[str],
                        out_png: str,
                        title: str,
                        bins: int = 50):
    arrays = [a for a in arrays if a is not None and a.size > 0]
    if not arrays: return
    lo = float(min(a.min() for a in arrays))
    hi = float(max(a.max() for a in arrays))
    edges = np.linspace(lo, hi, bins+1)
    centers = (edges[:-1] + edges[1:]) / 2
    wbin = np.diff(edges)

    n = len(arrays)
    per_w = 0.9 * wbin / max(1, n)
    offsets = [((k - (n-1)/2.0) * per_w) for k in range(n)]

    plt.figure(figsize=(10, 6))
    for k, (a, lab) in enumerate(zip(arrays, labels)):
        cnts, _ = np.histogram(a, bins=edges, density=False)
        plt.bar(centers + offsets[k], cnts, width=per_w, align="center", alpha=0.6, label=lab)
    plt.xlabel("Conductance Value (signed)", fontsize=12, fontweight='bold')
    plt.ylabel("Neuron Count", fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    if n > 1: plt.legend(fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close()

def _overlay_scatter_multi(arrays: List[np.ndarray],
                           labels: List[str],
                           out_png: str,
                           title: str):
    arrays = [a for a in arrays if a is not None and a.size > 0]
    if not arrays: return
    nmin = min(a.size for a in arrays)
    idx = np.arange(nmin)
    markers = ['o', 'x', '^', 's', 'd', '+', '*', 'v', '<', '>']

    plt.figure(figsize=(10, 6))
    for k, (a, lab) in enumerate(zip(arrays, labels)):
        v = a.ravel()[:nmin]
        m = markers[k % len(markers)]
        if k == 0: plt.scatter(idx, v, s=8, label=lab, alpha=0.6)
        elif k == 1: plt.scatter(idx, v, s=8, marker='x', label=lab, alpha=0.6)
        else: plt.scatter(idx, v, s=8, marker=m, label=lab, alpha=0.6)
    plt.xlabel("Neuron Index", fontsize=12, fontweight='bold')
    plt.ylabel("Conductance Value", fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    if len(arrays) > 1: plt.legend(fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close()

def _overlay_maxgap_multi(arrays: List[np.ndarray],
                          labels: List[str],
                          out_png: str,
                          title: str):
    pairs = [(a, lab) for a, lab in zip(arrays, labels) if a is not None and a.size > 0]
    if not pairs:
        return
    arrays_f, labels_f = zip(*pairs)

    nmin = min(a.size for a in arrays_f)
    idx = np.arange(nmin)

    plt.figure(figsize=(10, 6))
    for a, lab in zip(arrays_f, labels_f):
        v = np.sort(a.ravel())[::-1][:nmin]
        gap = v[0] - v
        plt.plot(idx, gap, label=lab, linewidth=2)

    plt.xlabel("Neuron Rank (0 = highest conductance)", fontsize=12, fontweight='bold')
    plt.ylabel("Gap from Maximum", fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    if len(arrays_f) > 1:
        plt.legend(fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close()

def _die(msg: str, code: int = 2):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)
    
# ==== CLI (add --runs/--labels/--best; keep legacy flags) ====

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Compare conductance runs (1, 2, or many)")
    p.add_argument("--runs", nargs="+", default=None,
                   help="One or more stats dirs (each has epoch_XXXX folders)")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Optional labels for runs (same length as --runs)")
    p.add_argument("--best", nargs="*", type=int, default=None,
                   help="Optional best epoch per run (same length as --runs). Omit to skip best overlays.")
    p.add_argument("--out",  required=True, help="Output directory for plots & logs")

    # legacy 2-run args
    p.add_argument("--runA", default=None)
    p.add_argument("--runB", default=None)
    p.add_argument("--bestA", type=int, default=None)
    p.add_argument("--bestB", type=int, default=None)
    p.add_argument("--labelA", default="A")
    p.add_argument("--labelB", default="B")

    p.add_argument("--metrics", nargs="+", default=None,
                   help="Subset of metrics: mean median variance IQR CV p5 p95 skewness gini")
    p.add_argument("--bins", type=int, default=200, help="Bins for hist overlays (default: 60)")
    p.add_argument("--cv-mode", choices=["abs", "signed"], default="abs",
                   help="CV definition: abs -> std(|x|)/mean(|x|), signed -> std/|mean|")
    p.add_argument("--epoch-gap", type=int, default=1,
                   help="Downsample & delta window (e.g., 10 => 0,10,20,... and Δ over 10)")
    p.add_argument("--invert-sign", action="store_true",
               help="If set, multiply all loaded per-neuron means by -1 before computing metrics/plots.")

    return p.parse_args()

def main():
    args = parse_args()

    # Resolve runs/labels/best from new or legacy flags
    runs = args.runs
    labels = args.labels
    best = args.best
    if runs is None:
        if args.runA is not None and args.runB is not None:
            runs = [args.runA, args.runB]
            labels = [args.labelA, args.labelB]
            best = [args.bestA, args.bestB]
        else:
            _die("Provide --runs <dirs...> (or legacy --runA/--runB).")

    # sanity checks
    for i, path in enumerate(runs):
        if not os.path.isdir(path):
            _die(f"Run #{i+1} path does not exist or is not a directory: {path}")
        has_epochs = any(d.startswith("epoch_") for d in os.listdir(path))
        if not has_epochs:
            print(f"[WARN] Run #{i+1} has no epoch_* folders: {path}", file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)

    compare_runs_multi(
        run_dirs=runs,
        out_dir=args.out,
        labels=labels,
        best_epochs=best,
        metrics_to_plot=args.metrics,
        bins=args.bins,
        cv_mode=args.cv_mode,
        epoch_gap=args.epoch_gap,
        invert_sign=args.invert_sign,
    )

    print(f"[OK] comparison complete. outputs in: {args.out}")

if __name__ == "__main__":
    main()