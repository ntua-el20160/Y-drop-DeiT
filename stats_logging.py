# streaming_conductance_tracker.py
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
    #pct_silent, pct_sat, eps_used, tau_used = _silent_saturated(x, eps, tau, eps_q, tau_q)
    return {
        "mean": float(np.mean(x)) if x.size else float("nan"),
        "median": float(np.median(x)) if x.size else float("nan"),
        "variance": float(np.var(x, ddof=1)) if x.size > 1 else 0.0,
        "IQR": float(q75 - q25) if x.size else float("nan"),
        "CV": float(cv),
        "p5": float(p5), "p95": float(p95),
        "skewness": float(skew),
        "gini": float(g),
        # "pct_silent": float(pct_silent),
        # "pct_saturated": float(pct_sat),
        # "eps_used": float(eps_used),
        # "tau_used": float(tau_used),
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
    """Exact 1D Wasserstein-1 (Earth Mover’s) for two samples with equal weights."""
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

def _plot_timeseries_single(metric_name: str,
                            epochs: List[int],
                            values: List[float],
                            out_png: str,
                            title: str):
    plt.figure()
    plt.plot(epochs, values, marker="o", linestyle="-")
    plt.xlabel("Epoch"); plt.ylabel(metric_name); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

def _plot_timeseries(metric_name: str,
                     epochs_A: List[int], values_A: List[float], label_A: str,
                     epochs_B: List[int], values_B: List[float], label_B: str,
                     out_png: str, title: str):
    plt.figure()
    plt.plot(epochs_A, values_A, marker="o", linestyle="-", label=label_A)
    plt.plot(epochs_B, values_B, marker="o", linestyle="-", label=label_B)
    plt.xlabel("Epoch"); plt.ylabel(metric_name); plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()
def _plot_hist(x: np.ndarray, out_png: str, title: str, bins: int = 50):
    plt.figure()
    counts, edges = np.histogram(x, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    plt.bar(centers, counts, width=np.diff(edges), align="center")
    plt.xlabel("Conductance (signed)"); plt.ylabel("Count"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

def _plot_values_scatter(x: np.ndarray, out_png: str, title: str):
    plt.figure()
    plt.scatter(np.arange(x.size), x, s=8)
    plt.xlabel("Neuron index"); plt.ylabel("Conductance"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()
def _overlay_hist(a: np.ndarray, b: np.ndarray, out_png: str, title: str, bins: int = 50):
    lo = float(min(a.min() if a.size else 0, b.min() if b.size else 0))
    hi = float(max(a.max() if a.size else 1, b.max() if b.size else 1))
    edges = np.linspace(lo, hi, bins+1)
    ca, _ = np.histogram(a, bins=edges, density=False)
    cb, _ = np.histogram(b, bins=edges, density=False)
    centers = (edges[:-1] + edges[1:]) / 2
    width = np.diff(edges) * 0.45
    plt.figure()
    plt.bar(centers - width/2, ca, width=width, alpha=0.6, label="A")
    plt.bar(centers + width/2, cb, width=width, alpha=0.6, label="B")
    plt.xlabel("Conductance (signed)"); plt.ylabel("Count"); plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

def _overlay_scatter(a: np.ndarray, b: np.ndarray, out_png: str, title: str):
    plt.figure()
    plt.scatter(np.arange(a.size), a, s=8, label="A")
    plt.scatter(np.arange(b.size), b, s=8, marker="x", label="B")
    plt.xlabel("Neuron index"); plt.ylabel("Conductance"); plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

def _overlay_maxgap(a: np.ndarray, b: np.ndarray, out_png: str, title: str):
    if a.size == 0 or b.size == 0:
        return
    sa = np.sort(a)[::-1]; sb = np.sort(b)[::-1]
    ga = sa-sa[0]; gb = sb - sb[0]
    plt.figure()
    plt.plot(np.arange(ga.size), ga, label="A")
    plt.plot(np.arange(gb.size), gb, label="B")
    plt.xlabel("Ranked neuron (0=max)"); plt.ylabel("Gap from max"); plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()
def _plot_maxgap(x: np.ndarray, out_png: str, title: str):
    if x.size == 0:
        return
    xs = np.sort(x)[::-1]
    gaps = xs - xs[0]
    plt.figure()
    plt.plot(np.arange(xs.size), gaps)
    plt.xlabel("Ranked neuron (0=max)"); plt.ylabel("Gap from max"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

def build_reports(output_dir: str,
                  transformer: bool = False,
                  block_mod: int = 4,
                  eps: Optional[float] = None,
                  tau: Optional[float] = None,
                  eps_q: float = 0.05,
                  tau_q: float = 0.98,
                  cv_mode: str = "abs",
                  bins: int = 50):
    """
    Walks all epoch_* folders under output_dir, loads per-layer arrays,
    computes requested metrics, writes readable logs, and saves plots.

    Logs inside each epoch folder:
      - stats_basic.txt             -> (1)–(9) per layer
      - stats_within_epoch_dynamics.txt -> largest jump (max_t), mean |Δ|, sign flips
      - stats_epoch_compare.txt     -> (11) avg diff vs prev epoch, (12) Spearman ρ_s
    Plots per layer:
      - histogram.png               -> (13)
      - values_scatter.png          -> (14)
      - maxgap.png                  -> (15)
    """
    # find epochs
    epochs = sorted([d for d in os.listdir(output_dir) if d.startswith("epoch_")])
    prev_layer_means: Dict[str, np.ndarray] = {}  # layer name -> vector (from previous epoch)

    for ep in epochs:
        ep_dir = os.path.join(output_dir, ep)
        try:
            with open(os.path.join(ep_dir, "manifest.json"), "r") as f:
                manifest = json.load(f)
        except FileNotFoundError:
            continue

        # ----- logs
        basic_log_lines: List[str] = []
        dyn_log_lines: List[str] = []
        cmp_log_lines: List[str] = []

        for lay in manifest["layers"]:
            name = lay["name"]
            lay_dir = os.path.join(ep_dir, name)
            mean = np.load(os.path.join(lay_dir, "per_neuron_mean.npy"))
            # (1)–(9)
            stats = _basic_stats(mean, cv_mode=cv_mode)
            basic_log_lines.append(f"[{name}] n={stats['n']}\n"
                                   f" mean={stats['mean']:.6f}  median={stats['median']:.6f}  variance={stats['variance']:.6f}\n"
                                   f" IQR={stats['IQR']:.6f}  CV={stats['CV']:.6f}  p5={stats['p5']:.6f}  p95={stats['p95']:.6f}\n"
                                   f" skewness={stats['skewness']:.6f}  gini(|x|)={stats['gini']:.6f}\n")
                                #    f" %silent(|x|<ε)={stats['pct_silent']:.2f}%  %saturated(|x|>τ)={stats['pct_saturated']:.2f}%"
                                #    f"  ε={stats['eps_used']:.6g}  τ={stats['tau_used']:.6g}\n")

            # within-epoch dynamics
            mad = np.load(os.path.join(lay_dir, "per_neuron_mean_abs_diff.npy"))
            mxj = np.load(os.path.join(lay_dir, "per_neuron_max_abs_jump.npy"))
            flips = np.load(os.path.join(lay_dir, "per_neuron_sign_flips.npy"))
            dyn_log_lines.append(f"[{name}] mean(|Δ|) per iter: mean={mad.mean():.6f} median={np.median(mad):.6f} "
                                 f"max(|Δ|) per neuron: mean={mxj.mean():.6f} max={mxj.max():.6f} "
                                 f"sign_flips: mean={flips.mean():.3f} max={flips.max():d}\n")

            # plots (13,14,15)
            _plot_hist(mean, os.path.join(lay_dir, "histogram.png"),
                       title=f"{ep} – {name} – histogram", bins=bins)
            _plot_values_scatter(mean, os.path.join(lay_dir, "values_scatter.png"),
                                 title=f"{ep} – {name} – values per neuron")
            _plot_maxgap(mean, os.path.join(lay_dir, "maxgap.png"),
                         title=f"{ep} – {name} – gap from max")

            # (11) & (12) vs previous epoch
            key = name  # stable across epochs due to naming scheme
            if key in prev_layer_means:
                prev = prev_layer_means[key]
                n = min(prev.size, mean.size)
                diffs = np.abs(mean[:n] - prev[:n])
                avg_diff = diffs.mean()
                rho = spearman_r(mean[:n], prev[:n])
                cmp_log_lines.append(f"[{name}] vs prev-epoch: avg|Δ|={avg_diff:.6f}  Spearman ρ_s={rho:.4f}\n")

            prev_layer_means[key] = mean

        # write logs
        with open(os.path.join(ep_dir, "stats_basic.txt"), "w") as f:
            f.write("=== BASIC STATS (1–9) ===\n")
            f.writelines(basic_log_lines)
        with open(os.path.join(ep_dir, "stats_within_epoch_dynamics.txt"), "w") as f:
            f.write("=== WITHIN-EPOCH DYNAMICS (10: largest jump max_t, 11*: mean|Δ| per-iter, sign flips) ===\n")
            f.writelines(dyn_log_lines)
        if cmp_log_lines:
            with open(os.path.join(ep_dir, "stats_epoch_compare.txt"), "w") as f:
                f.write("=== BETWEEN-EPOCH COMPARISON (11: avg diff, 12: Spearman ρ_s) ===\n")
                f.writelines(cmp_log_lines)
    layer_epoch_means: Dict[str, Dict[int, np.ndarray]] = _read_layer_means(output_dir)

    timeseries_root = os.path.join(output_dir, "timeseries")
    _ensure_dir(timeseries_root)

    metrics_to_plot = ["mean", "median", "variance", "IQR", "CV", "p5", "p95", "skewness", "gini"]
    # collect rows for CSV: (layer, epoch, metric, value)
    ts_rows: List[Tuple[str, int, str, float]] = []

    for layer, ep2vec in layer_epoch_means.items():
        lay_ts_dir = os.path.join(timeseries_root, layer)
        _ensure_dir(lay_ts_dir)

        ep_list = sorted(ep2vec.keys())
        # Precompute stats per epoch once
        stats_per_epoch: Dict[int, Dict[str, float]] = {}
        for e in ep_list:
            stats_per_epoch[e] = _basic_stats(ep2vec[e], cv_mode=cv_mode)

        for metric in metrics_to_plot:
            vals = [stats_per_epoch[e][metric] for e in ep_list]
            out_png = os.path.join(lay_ts_dir, f"timeseries_{metric}.png")
            _plot_timeseries_single(metric_name=metric,
                                    epochs=ep_list,
                                    values=vals,
                                    out_png=out_png,
                                    title=f"{layer} – {metric} over epochs")
            # CSV rows
            for e, v in zip(ep_list, vals):
                ts_rows.append((layer, e, metric, v))

    # Write the tidy CSV once
    csv_path = os.path.join(timeseries_root, "metrics_timeseries.csv")
    with open(csv_path, "w") as f:
        f.write("layer,epoch,metric,value\n")
        for layer, e, metric, v in ts_rows:
            # handle NaNs cleanly
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

def _read_layer_means(run_dir: str) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Returns: {layer_name: {epoch_index: mean_vector}}
    """
    result: Dict[str, Dict[int, np.ndarray]] = {}
    for ep_name in _load_manifest_epochs(run_dir):
        ep_idx = int(ep_name.split("_")[1])
        ep_dir = os.path.join(run_dir, ep_name)
        # manifest guides which layers exist
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
            result.setdefault(name, {})[ep_idx] = mean
    return result

def compare_runs(runA_dir: str, runB_dir: str,
                 best_epoch_A: Optional[int],
                 best_epoch_B: Optional[int],
                 out_dir: str,
                 metrics_to_plot: Optional[List[str]] = None,
                 bins: int = 60,
                 cv_mode: str = "abs",
                 labelA: str = "A",
                 labelB: str = "B"):
    """
    Build run-to-run comparisons.
      - Time series of basic stats per layer for both runs
      - Overlays (hist/scatter/maxgap) for last epoch and user-provided best epochs
      - Comparison logs (Δmetrics + KS + Wasserstein) for last & best

    metrics_to_plot: subset of ["mean","median","variance","IQR","CV","p5","p95","skewness","gini"]
    """
    _ensure_dir(out_dir)
    A = _read_layer_means(runA_dir)
    B = _read_layer_means(runB_dir)

    layers_common = sorted(set(A.keys()) & set(B.keys()))
    if not layers_common:
        raise RuntimeError("No common layers found between runs.")

    if metrics_to_plot is None:
        metrics_to_plot = ["mean","median","variance","IQR","CV","p95","skewness","gini"]

    # find last epochs (max) for each run
    last_epoch_A = max({e for d in A.values() for e in d.keys()})
    last_epoch_B = max({e for d in B.values() for e in d.keys()})

    # summary logs
    last_log = []
    best_log = []

    for layer in layers_common:
        layer_dir = os.path.join(out_dir, layer)
        _ensure_dir(layer_dir)

        # --- collect time series of basic stats for each run ---
        epA = sorted(A[layer].keys())
        epB = sorted(B[layer].keys())
        statsA = {m: [] for m in metrics_to_plot}
        statsB = {m: [] for m in metrics_to_plot}

        for e in epA:
            s = _basic_stats(A[layer][e], cv_mode=cv_mode)
            for m in metrics_to_plot: statsA[m].append(s[m])
        for e in epB:
            s = _basic_stats(B[layer][e], cv_mode=cv_mode)
            for m in metrics_to_plot: statsB[m].append(s[m])

        # --- plot time series per metric ---
        for m in metrics_to_plot:
            out_png = os.path.join(layer_dir, f"timeseries_{m}.png")
            _plot_timeseries(
                metric_name=m,
                epochs_A=epA, values_A=statsA[m], label_A=labelA,
                epochs_B=epB, values_B=statsB[m], label_B=labelB,
                out_png=out_png,
                title=f"{layer} – {m} over epochs"
            )

        # --- last-epoch overlays ---
        a_last = A[layer].get(last_epoch_A)
        b_last = B[layer].get(last_epoch_B)
        if a_last is not None and b_last is not None:
            _overlay_hist(a_last, b_last, os.path.join(layer_dir, "overlay_hist_last.png"),
                          title=f"{layer} – Histogram (last epochs {labelA}={last_epoch_A}, {labelB}={last_epoch_B})",
                          bins=bins)
            _overlay_scatter(a_last, b_last, os.path.join(layer_dir, "overlay_scatter_last.png"),
                             title=f"{layer} – Value per neuron (last)")
            _overlay_maxgap(a_last, b_last, os.path.join(layer_dir, "overlay_maxgap_last.png"),
                            title=f"{layer} – Gap from max (last)")

            sA = _basic_stats(a_last, cv_mode=cv_mode)
            sB = _basic_stats(b_last, cv_mode=cv_mode)
            ks = _ks_distance(a_last, b_last)
            w1 = _wasserstein_1d(a_last, b_last)
            # pairwise neuron alignment for rank corr (use min length)
            nmin = min(a_last.size, b_last.size)
            rho = np.corrcoef(
                np.argsort(np.argsort(a_last[:nmin])),
                np.argsort(np.argsort(b_last[:nmin]))
            )[0,1]
            last_log.append(
                f"[{layer}] LAST  {labelA}@{last_epoch_A} vs {labelB}@{last_epoch_B}\n"
                f" Δmean={sA['mean']-sB['mean']:+.6f}  Δmedian={sA['median']-sB['median']:+.6f}  "
                f" Δvar={sA['variance']-sB['variance']:+.6f}  ΔIQR={sA['IQR']-sB['IQR']:+.6f}  "
                f" ΔCV={sA['CV']-sB['CV']:+.6f}  Δp95={sA['p95']-sB['p95']:+.6f}  "
                f" Δskew={sA['skewness']-sB['skewness']:+.6f}  Δgini={sA['gini']-sB['gini']:+.6f}\n"
                f" KS={ks:.4f}  W1={w1:.6f}  Spearman ρ_s≈{rho:.4f}\n"
            )

        # --- best-epoch overlays ---
        if best_epoch_A is not None and best_epoch_B is not None:
            a_best = A[layer].get(int(best_epoch_A))
            b_best = B[layer].get(int(best_epoch_B))
            if a_best is not None and b_best is not None:
                _overlay_hist(a_best, b_best, os.path.join(layer_dir, "overlay_hist_best.png"),
                              title=f"{layer} – Histogram (best {labelA}={best_epoch_A}, {labelB}={best_epoch_B})",
                              bins=bins)
                _overlay_scatter(a_best, b_best, os.path.join(layer_dir, "overlay_scatter_best.png"),
                                 title=f"{layer} – Value per neuron (best)")
                _overlay_maxgap(a_best, b_best, os.path.join(layer_dir, "overlay_maxgap_best.png"),
                                title=f"{layer} – Gap from max (best)")

                sA = _basic_stats(a_best, cv_mode=cv_mode)
                sB = _basic_stats(b_best, cv_mode=cv_mode)
                ks = _ks_distance(a_best, b_best)
                w1 = _wasserstein_1d(a_best, b_best)
                nmin = min(a_best.size, b_best.size)
                rho = np.corrcoef(
                    np.argsort(np.argsort(a_best[:nmin])),
                    np.argsort(np.argsort(b_best[:nmin]))
                )[0,1]
                best_log.append(
                    f"[{layer}] BEST  A@{best_epoch_A} vs B@{best_epoch_B}\n"
                    f" Δmean={sA['mean']-sB['mean']:+.6f}  Δmedian={sA['median']-sB['median']:+.6f}  "
                    f" Δvar={sA['variance']-sB['variance']:+.6f}  ΔIQR={sA['IQR']-sB['IQR']:+.6f}  "
                    f" ΔCV={sA['CV']-sB['CV']:+.6f}  Δp95={sA['p95']-sB['p95']:+.6f}  "
                    f" Δskew={sA['skewness']-sB['skewness']:+.6f}  Δgini={sA['gini']-sB['gini']:+.6f}\n"
                    f" KS={ks:.4f}  W1={w1:.6f}  Spearman ρ_s≈{rho:.4f}\n"
                )

    # write comparison logs
    with open(os.path.join(out_dir, "comparison_last_epochs.txt"), "w") as f:
        f.write("=== BETWEEN-RUN COMPARISON @ LAST EPOCHS ===\n")
        f.writelines(last_log)
    if best_log:
        with open(os.path.join(out_dir, "comparison_best_epochs.txt"), "w") as f:
            f.write("=== BETWEEN-RUN COMPARISON @ BEST EPOCHS ===\n")
            f.writelines(best_log)

    # optional: aggregate across layers barplots of KS/W1 at last & best
    def _aggregate_plot(entries: List[str], out_png: str, label: str):
        # parse lines to get KS and W1
        layers, kss, w1s = [], [], []
        for line in entries:
            if line.startswith("[") and "KS=" in line:
                parts = line.split()
                layer = line.split("]")[0][1:]
                KS = float([p for p in parts if p.startswith("KS=")][0].split("=")[1])
                W1 = float([p for p in parts if p.startswith("W1=")][0].split("=")[1])
                layers.append(layer); kss.append(KS); w1s.append(W1)
        if not layers: return
        x = np.arange(len(layers))
        plt.figure(figsize=(max(6, 0.4*len(layers)), 3))
        plt.bar(x-0.18, kss, width=0.36, label="KS")
        plt.bar(x+0.18, w1s, width=0.36, label="W1")
        plt.xticks(x, layers, rotation=60, ha="right")
        plt.ylabel(label); plt.title(f"{label} per layer")
        plt.legend(); plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

    _aggregate_plot(last_log, os.path.join(out_dir, "aggregate_last_KS_W1.png"),
                    label="Distance (last epochs)")
    if best_log:
        _aggregate_plot(best_log, os.path.join(out_dir, "aggregate_best_KS_W1.png"),
                        label="Distance (best epochs)")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Compare two conductance runs")
    p.add_argument("--runA", required=True, help="Path to RUN A stats dir (contains epoch_XXXX folders)")
    p.add_argument("--runB", required=True, help="Path to RUN B stats dir (contains epoch_XXXX folders)")
    p.add_argument("--out",  required=True, help="Output directory for comparison plots & logs")
    p.add_argument("--bestA", type=int, default=None, help="Best epoch index for RUN A (optional)")
    p.add_argument("--bestB", type=int, default=None, help="Best epoch index for RUN B (optional)")
    p.add_argument("--metrics", nargs="+", default=None,
                   help="Subset of metrics to plot over time. "
                        "Choose from: mean median variance IQR CV p5 p95 skewness gini")
    p.add_argument("--bins", type=int, default=200, help="Bins for hist overlays (default: 60)")
    p.add_argument("--cv-mode", choices=["abs", "signed"], default="abs",
                   help="CV definition: abs -> std(|x|)/mean(|x|), signed -> std/|mean|")
    return p.parse_args()

def _die(msg: str, code: int = 2):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)

def main():
    args = parse_args()


    # sanity checks
    for label, path in [("RUN A", args.runA), ("RUN B", args.runB)]:
        if not os.path.isdir(path):
            _die(f"{label} path does not exist or is not a directory: {path}")
        # quick check for epoch folders
        has_epochs = any(d.startswith("epoch_") for d in os.listdir(path))
        if not has_epochs:
            print(f"[WARN] {label} has no epoch_* folders at: {path} — is this the correct stats dir?", file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)

    # call the comparator
    compare_runs(
        runA_dir=args.runA,
        runB_dir=args.runB,
        best_epoch_A=args.bestA,
        best_epoch_B=args.bestB,
        out_dir=args.out,
        metrics_to_plot=args.metrics,
        bins=args.bins,
        cv_mode=args.cv_mode,
    )

    print(f"[OK] comparison complete. outputs in: {args.out}")

if __name__ == "__main__":
    main()