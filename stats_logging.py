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
                  bins: int = 50,
                  epoch_gap: int = 1):
    """
    Walks all epoch_* folders under output_dir, loads per-layer arrays,
    computes requested metrics, writes readable logs, and saves plots.

    Per-epoch (downsampled by epoch_gap) logs inside each epoch folder:
      - stats_basic.txt
      - stats_within_epoch_dynamics.txt
      - stats_epoch_compare.txt (vs previous selected epoch)

    Per-epoch (downsampled) plots per layer:
      - histogram.png
      - values_scatter.png
      - maxgap.png

    Cross-epoch time-series per layer in output_dir/timeseries/<layer>/ :
      - timeseries_<metric>_gap{epoch_gap}.png       (sampled)
      - timeseries_<metric>_delta{epoch_gap}.png     (Δ over the gap)
      - metrics_timeseries_gap{epoch_gap}.csv        (sampled)
      - metrics_delta_gap{epoch_gap}.csv             (deltas)
    """
    # ---- 1) Per-epoch reporting (downsampled) ----
    # map epoch_idx -> folder name
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
        cmp_log_lines: List[str] = []  # vs previous selected epoch

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
                       title=f"epoch_{eidx:04d} – {name} – histogram", bins=bins)
            _plot_values_scatter(mean, os.path.join(lay_dir, "values_scatter.png"),
                                 title=f"epoch_{eidx:04d} – {name} – values per neuron")
            _plot_maxgap(mean, os.path.join(lay_dir, "maxgap.png"),
                         title=f"epoch_{eidx:04d} – {name} – gap from max")

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
    layer_epoch_means: Dict[str, Dict[int, np.ndarray]] = _read_layer_means(output_dir)

    ts_root = os.path.join(output_dir, "timeseries")
    _ensure_dir(ts_root)

    metrics = ["mean", "median", "variance", "IQR", "CV", "p5", "p95", "skewness", "gini"]

    sampled_rows: List[Tuple[str, int, str, float]] = []
    delta_rows:   List[Tuple[str, int, str, float]] = []

    for layer, ep2vec in layer_epoch_means.items():
        lay_ts_dir = os.path.join(ts_root, layer)
        _ensure_dir(lay_ts_dir)

        all_ep = sorted(ep2vec.keys())
        ds_ep = _downsample_epochs(all_ep, gap=epoch_gap, include_last=True)

        # Precompute stats on all epochs, then slice
        stats_all: Dict[int, Dict[str, float]] = {e: _basic_stats(ep2vec[e], cv_mode=cv_mode) for e in all_ep}

        for metric in metrics:
            vals_sampled = [stats_all[e][metric] for e in ds_ep]
            out_png = os.path.join(lay_ts_dir, f"timeseries_{metric}_gap{epoch_gap}.png")
            _plot_timeseries_single(metric_name=metric,
                                    epochs=ds_ep,
                                    values=vals_sampled,
                                    out_png=out_png,
                                    title=f"{layer} – {metric} (every {epoch_gap} epochs)")

            for e, v in zip(ds_ep, vals_sampled):
                sampled_rows.append((layer, e, metric, v))

            # Δ over epoch_gap (only where e-gap exists)
            delta_ep = [e for e in ds_ep if (e - epoch_gap) in stats_all]
            delta_vals = [stats_all[e][metric] - stats_all[e - epoch_gap][metric] for e in delta_ep]
            out_png_delta = os.path.join(lay_ts_dir, f"timeseries_{metric}_delta{epoch_gap}.png")
            if delta_ep:  # only plot if we have something
                _plot_timeseries_single(metric_name=f"Δ{metric} (over {epoch_gap})",
                                        epochs=delta_ep,
                                        values=delta_vals,
                                        out_png=out_png_delta,
                                        title=f"{layer} – Δ{metric} over {epoch_gap}")

            for e, dv in zip(delta_ep, delta_vals):
                delta_rows.append((layer, e, f"delta_{metric}", dv))

    # Write CSVS
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

# ==== generalized comparator (N runs) + back-compat wrapper ====

def compare_runs_multi(run_dirs: List[str],
                       out_dir: str,
                       labels: Optional[List[str]] = None,
                       best_epochs: Optional[List[Optional[int]]] = None,
                       metrics_to_plot: Optional[List[str]] = None,
                       bins: int = 60,
                       cv_mode: str = "abs",
                       epoch_gap: int = 1):
    _ensure_dir(out_dir)
    assert len(run_dirs) >= 1
    if labels is None or len(labels) == 0:
        labels = [f"R{i+1}" for i in range(len(run_dirs))]
    assert len(labels) == len(run_dirs)
    if best_epochs is None:
        best_epochs = [None] * len(run_dirs)
    assert len(best_epochs) == len(run_dirs)

    runs_data = [_read_layer_means(rd) for rd in run_dirs]
    layers_common = sorted(set.intersection(*(set(d.keys()) for d in runs_data)))
    if not layers_common:
        raise RuntimeError("No common layers found across runs.")

    if metrics_to_plot is None:
        metrics_to_plot = ["mean","median","variance","IQR","CV","p95","skewness","gini"]

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

        last_arrays = []
        best_arrays = []

        for run_idx, (d, lab) in enumerate(zip(runs_data, labels)):
            ep_all = sorted(d[layer].keys())
            ep = _downsample_epochs(ep_all, gap=epoch_gap, include_last=True)

            stats = {m: [] for m in metrics_to_plot}
            for e in ep:
                s = _basic_stats(d[layer][e], cv_mode=cv_mode)
                for m in metrics_to_plot: stats[m].append(s[m])

            for m in metrics_to_plot:
                per_metric_series[m].append((ep, stats[m], lab))
                # deltas
                mp = dict(zip(ep, stats[m]))
                dEp = [t for t in ep if (t - epoch_gap) in mp]
                dVal = [mp[t] - mp[t - epoch_gap] for t in dEp]
                per_metric_delta[m].append((dEp, dVal, f"{lab} Δ{m}"))

            last_arrays.append(d[layer].get(last_epochs[run_idx]))
            be = best_epochs[run_idx]
            best_arrays.append(d[layer].get(int(be)) if be is not None else None)

        # Plot time-series & deltas (multi)
        for m in metrics_to_plot:
            _plot_timeseries_multi(m, per_metric_series[m],
                                   os.path.join(layer_dir, f"timeseries_{m}_gap{epoch_gap}.png"),
                                   title=f"{layer} – {m} (every {epoch_gap})")
            _plot_timeseries_multi(f"Δ{m} (over {epoch_gap})", per_metric_delta[m],
                                   os.path.join(layer_dir, f"timeseries_{m}_delta{epoch_gap}.png"),
                                   title=f"{layer} – Δ{m} over {epoch_gap}")

        # Last-epoch overlays (multi if ≥2, else single)
        if all(a is not None for a in last_arrays):
            if len(last_arrays) == 1:
                arr = last_arrays[0]
                _plot_hist(arr, os.path.join(layer_dir, "hist_last.png"),
                           title=f"{layer} – Histogram (last {labels[0]}={last_epochs[0]})", bins=bins)
                _plot_values_scatter(arr, os.path.join(layer_dir, "scatter_last.png"),
                                     title=f"{layer} – Value per neuron (last)")
                _plot_maxgap(arr, os.path.join(layer_dir, "maxgap_last.png"),
                             title=f"{layer} – Gap from max (last)")
                s = _basic_stats(arr, cv_mode=cv_mode)
                last_log.append(
                    f"[{layer}] LAST  {labels[0]}@{last_epochs[0]}\n"
                    f" mean={s['mean']:.6f}  median={s['median']:.6f}  var={s['variance']:.6f}  "
                    f"IQR={s['IQR']:.6f}  CV={s['CV']:.6f}  p95={s['p95']:.6f}  "
                    f"skew={s['skewness']:.6f}  gini={s['gini']:.6f}\n"
                )
            else:
                _overlay_hist_multi(last_arrays, labels,
                                    os.path.join(layer_dir, "overlay_hist_last.png"),
                                    title=f"{layer} – Histogram (last epochs " +
                                          ", ".join(f"{lab}={ep}" for lab,ep in zip(labels,last_epochs)) + ")",
                                    bins=bins)
                _overlay_scatter_multi(last_arrays, labels,
                                       os.path.join(layer_dir, "overlay_scatter_last.png"),
                                       title=f"{layer} – Value per neuron (last)")
                _overlay_maxgap_multi(last_arrays, labels,
                                      os.path.join(layer_dir, "overlay_maxgap_last.png"),
                                      title=f"{layer} – Gap from max (last)")

                # per-run stats
                for lab, arr in zip(labels, last_arrays):
                    s = _basic_stats(arr, cv_mode=cv_mode)
                    last_log.append(
                        f"[{layer}] LAST  {lab}@{last_epochs[labels.index(lab)]}\n"
                        f" mean={s['mean']:.6f}  median={s['median']:.6f}  var={s['variance']:.6f}  "
                        f"IQR={s['IQR']:.6f}  CV={s['CV']:.6f}  p95={s['p95']:.6f}  "
                        f"skew={s['skewness']:.6f}  gini={s['gini']:.6f}\n"
                    )
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

        # Best-epoch overlays (only if all best provided & present)
        have_all_best = all((be is not None and arr is not None)
                            for be, arr in zip(best_epochs, best_arrays))
        if have_all_best:
            if len(best_arrays) == 1:
                arr = best_arrays[0]
                _plot_hist(arr, os.path.join(layer_dir, "hist_best.png"),
                           title=f"{layer} – Histogram (best {labels[0]}={best_epochs[0]})", bins=bins)
                _plot_values_scatter(arr, os.path.join(layer_dir, "scatter_best.png"),
                                     title=f"{layer} – Value per neuron (best)")
                _plot_maxgap(arr, os.path.join(layer_dir, "maxgap_best.png"),
                             title=f"{layer} – Gap from max (best)")
                s = _basic_stats(arr, cv_mode=cv_mode)
                best_log.append(
                    f"[{layer}] BEST  {labels[0]}@{best_epochs[0]}\n"
                    f" mean={s['mean']:.6f}  median={s['median']:.6f}  var={s['variance']:.6f}  "
                    f"IQR={s['IQR']:.6f}  CV={s['CV']:.6f}  p95={s['p95']:.6f}  "
                    f"skew={s['skewness']:.6f}  gini={s['gini']:.6f}\n"
                )
            else:
                _overlay_hist_multi(best_arrays, labels,
                                    os.path.join(layer_dir, "overlay_hist_best.png"),
                                    title=f"{layer} – Histogram (best " +
                                          ", ".join(f"{lab}={be}" for lab,be in zip(labels,best_epochs)) + ")",
                                    bins=bins)
                _overlay_scatter_multi(best_arrays, labels,
                                       os.path.join(layer_dir, "overlay_scatter_best.png"),
                                       title=f"{layer} – Value per neuron (best)")
                _overlay_maxgap_multi(best_arrays, labels,
                                      os.path.join(layer_dir, "overlay_maxgap_best.png"),
                                      title=f"{layer} – Gap from max (best)")

                # per-run stats + pairwise
                n = len(labels)
                for lab, arr in zip(labels, best_arrays):
                    s = _basic_stats(arr, cv_mode=cv_mode)
                    best_log.append(
                        f"[{layer}] BEST  {lab}@{best_epochs[labels.index(lab)]}\n"
                        f" mean={s['mean']:.6f}  median={s['median']:.6f}  var={s['variance']:.6f}  "
                        f"IQR={s['IQR']:.6f}  CV={s['CV']:.6f}  p95={s['p95']:.6f}  "
                        f"skew={s['skewness']:.6f}  gini={s['gini']:.6f}\n"
                    )
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

    # Write logs (keep original filenames)
    with open(os.path.join(out_dir, "comparison_last_epochs.txt"), "w") as f:
        f.write("=== BETWEEN-RUN COMPARISON @ LAST EPOCHS ===\n")
        f.writelines(last_log)
    if best_log:
        with open(os.path.join(out_dir, "comparison_best_epochs.txt"), "w") as f:
            f.write("=== BETWEEN-RUN COMPARISON @ BEST EPOCHS ===\n")
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
                           title: str):
    plt.figure()
    for epochs, values, label in series:
        if epochs:
            plt.plot(epochs, values, marker="o", linestyle="-", label=label)
    plt.xlabel("Epoch"); plt.ylabel(metric_name); plt.title(title)
    if len(series) > 1: plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

def _overlay_hist_multi(arrays: List[np.ndarray],
                        labels: List[str],
                        out_png: str,
                        title: str,
                        bins: int = 50):
    # common binning across all arrays
    arrays = [a for a in arrays if a is not None and a.size > 0]
    if not arrays: return
    lo = float(min(a.min() for a in arrays))
    hi = float(max(a.max() for a in arrays))
    edges = np.linspace(lo, hi, bins+1)
    centers = (edges[:-1] + edges[1:]) / 2
    wbin = np.diff(edges)  # per-bin widths

    n = len(arrays)
    # Reserve 90% of each bin for bars; split evenly per run.
    per_w = 0.9 * wbin / max(1, n)
    offsets = [((k - (n-1)/2.0) * per_w) for k in range(n)]

    plt.figure()
    for k, (a, lab) in enumerate(zip(arrays, labels)):
        cnts, _ = np.histogram(a, bins=edges, density=False)
        plt.bar(centers + offsets[k], cnts, width=per_w, align="center", alpha=0.6, label=lab)
    plt.xlabel("Conductance (signed)"); plt.ylabel("Count"); plt.title(title); 
    if n > 1: plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

def _overlay_scatter_multi(arrays: List[np.ndarray],
                           labels: List[str],
                           out_png: str,
                           title: str):
    arrays = [a for a in arrays if a is not None and a.size > 0]
    if not arrays: return
    nmin = min(a.size for a in arrays)
    idx = np.arange(nmin)
    markers = ['o', 'x', '^', 's', 'd', '+', '*', 'v', '<', '>']

    plt.figure()
    for k, (a, lab) in enumerate(zip(arrays, labels)):
        v = a.ravel()[:nmin]
        m = markers[k % len(markers)]
        # Match original 2-run look: first '.', second 'x'
        if k == 0: plt.scatter(idx, v, s=8, label=lab)            # default 'o'
        elif k == 1: plt.scatter(idx, v, s=8, marker='x', label=lab)
        else: plt.scatter(idx, v, s=8, marker=m, label=lab)
    plt.xlabel("Neuron index"); plt.ylabel("Conductance"); plt.title(title)
    if len(arrays) > 1: plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

def _overlay_maxgap_multi(arrays: List[np.ndarray],
                          labels: List[str],
                          out_png: str,
                          title: str):
    # Keep arrays & labels in sync
    pairs = [(a, lab) for a, lab in zip(arrays, labels) if a is not None and a.size > 0]
    if not pairs:
        return
    arrays_f, labels_f = zip(*pairs)

    # Align on common length and plot ranked gaps (non-negative)
    nmin = min(a.size for a in arrays_f)
    idx = np.arange(nmin)

    plt.figure()
    for a, lab in zip(arrays_f, labels_f):
        v = np.sort(a.ravel())[::-1][:nmin]   # rank by value (0 = max)
        gap = v -v[0]                     # non-negative “gap from max”
        plt.plot(idx, gap, label=lab)

    plt.xlabel("Ranked neuron (0=max)")
    plt.ylabel("Gap from max")
    plt.title(title)
    if len(arrays_f) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

# ---- original helper names kept, now delegate to multi-run ----

def _plot_timeseries(metric_name: str,
                     epochs_A: List[int], values_A: List[float], label_A: str,
                     epochs_B: List[int], values_B: List[float], label_B: str,
                     out_png: str, title: str):
    _plot_timeseries_multi(metric_name,
                           [(epochs_A, values_A, label_A),
                            (epochs_B, values_B, label_B)],
                           out_png, title)

def _plot_timeseries_single(metric_name: str,
                            epochs: List[int],
                            values: List[float],
                            out_png: str,
                            title: str):
    _plot_timeseries_multi(metric_name,
                           [(epochs, values, "")],
                           out_png, title)

def _overlay_hist(a: np.ndarray, b: np.ndarray, out_png: str, title: str, bins: int = 50):
    _overlay_hist_multi([a, b], ["A", "B"], out_png, title, bins=bins)

def _overlay_scatter(a: np.ndarray, b: np.ndarray, out_png: str, title: str):
    _overlay_scatter_multi([a, b], ["A", "B"], out_png, title)

def _overlay_maxgap(a: np.ndarray, b: np.ndarray, out_png: str, title: str):
    _overlay_maxgap_multi([a, b], ["A", "B"], out_png, title)
    
def _plot_hist(x: np.ndarray, out_png: str, title: str, bins: int = 50):
    # unchanged single-run histogram
    plt.figure()
    counts, edges = np.histogram(x, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    plt.bar(centers, counts, width=np.diff(edges), align="center")
    plt.xlabel("Conductance (signed)"); plt.ylabel("Count"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

def _plot_values_scatter(x: np.ndarray, out_png: str, title: str):
    # unchanged single-run scatter
    plt.figure()
    plt.scatter(np.arange(x.size), x, s=8)
    plt.xlabel("Neuron index"); plt.ylabel("Conductance"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

def _plot_maxgap(x: np.ndarray, out_png: str, title: str):
    if x.size == 0: return
    xs = np.sort(x)[::-1]
    gaps =xs[0] -xs
    plt.figure()
    plt.plot(np.arange(xs.size), gaps)
    plt.xlabel("Ranked neuron (0=max)"); plt.ylabel("Gap from max"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

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
    )

    print(f"[OK] comparison complete. outputs in: {args.out}")

if __name__ == "__main__":
    main()