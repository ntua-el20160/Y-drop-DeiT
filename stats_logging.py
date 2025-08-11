# streaming_conductance_tracker.py
from __future__ import annotations
import os, json, math
from typing import Dict, Optional, Tuple, List
import numpy as np
import torch
import matplotlib.pyplot as plt

# ============== small math helpers ==============

def _to_1d_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().reshape(-1).cpu().numpy().astype(np.float32)

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

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

def _basic_stats(x: np.ndarray, cv_mode: str, eps: Optional[float], tau: Optional[float],
                 eps_q: float, tau_q: float) -> Dict[str, float]:
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
            stats = _basic_stats(mean, cv_mode=cv_mode, eps=eps, tau=tau,
                                 eps_q=eps_q, tau_q=tau_q)
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
