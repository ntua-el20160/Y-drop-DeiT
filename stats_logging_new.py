# streaming_conductance_tracker_last_only.py
from __future__ import annotations
import os, json, sys
from typing import Dict, Optional, Tuple, List
import numpy as np
import torch
import matplotlib.pyplot as plt

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

# ============== streaming epoch tracker (unchanged; provided for completeness) ==============

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

# ============== single-epoch plots & distances ==============

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
    if x.size == 0: return
    xs = np.sort(x)[::-1]
    gaps = xs[0] - xs  # non-negative gap from max
    plt.figure()
    plt.plot(np.arange(xs.size), gaps)
    plt.xlabel("Ranked neuron (0=max)"); plt.ylabel("Gap from max"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

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

    plt.figure()
    for k, (a, lab) in enumerate(zip(arrays, labels)):
        cnts, _ = np.histogram(a, bins=edges, density=False)
        plt.bar(centers + offsets[k], cnts, width=per_w, align="center", alpha=0.6, label=lab)
    plt.xlabel("Conductance (signed)"); plt.ylabel("Count"); plt.title(title)
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
        if k == 0: plt.scatter(idx, v, s=8, label=lab)
        elif k == 1: plt.scatter(idx, v, s=8, marker='x', label=lab)
        else: plt.scatter(idx, v, s=8, marker=m, label=lab)
    plt.xlabel("Neuron index"); plt.ylabel("Conductance"); plt.title(title)
    if len(arrays) > 1: plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

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

    plt.figure()
    for a, lab in zip(arrays_f, labels_f):
        v = np.sort(a.ravel())[::-1][:nmin]
        gap = v - v[0]
        plt.plot(idx, gap, label=lab)
    plt.xlabel("Ranked neuron (0=max)")
    plt.ylabel("Gap from max")
    plt.title(title)
    if len(arrays_f) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
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
    # pick the numerically last epoch
    ep_idx = max(int(d.split("_")[1]) for d in epoch_dirs)
    ep_name = f"epoch_{ep_idx:04d}"
    ep_dir = os.path.join(run_dir, ep_name)

    # prefer manifest for layer listing
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

    # Per-layer overlays & stats @ last epoch only
    for layer in common_layers:
        layer_dir = os.path.join(out_dir, layer)
        _ensure_dir(layer_dir)

        arrays = [d[layer] for d in per_run_layers]
        # single-run case: plot basic views
        if len(arrays) == 1:
            arr = arrays[0]
            _plot_hist(arr, os.path.join(layer_dir, "hist_last.png"),
                       title=f"{layer} – Histogram (last {labels[0]}={per_run_epoch[0]})", bins=bins)
            _plot_values_scatter(arr, os.path.join(layer_dir, "scatter_last.png"),
                                 title=f"{layer} – Value per neuron (last)")
            _plot_maxgap(arr, os.path.join(layer_dir, "maxgap_last.png"),
                         title=f"{layer} – Gap from max (last)")
            s = _basic_stats(arr, cv_mode=cv_mode)
            last_log.append(
                f"[{layer}] LAST  {labels[0]}@{per_run_epoch[0]}\n"
                f" mean={s['mean']:.6f}  median={s['median']:.6f}  var={s['variance']:.6f}  "
                f"IQR={s['IQR']:.6f}  CV={s['CV']:.6f}  p95={s['p95']:.6f}  "
                f"skew={s['skewness']:.6f}  gini={s['gini']:.6f}\n"
            )
        else:
            _overlay_hist_multi(arrays, labels,
                                os.path.join(layer_dir, "overlay_hist_last.png"),
                                title=f"{layer} – Histogram (last: " +
                                      ", ".join(f"{lab}={ep}" for lab,ep in zip(labels, per_run_epoch)) + ")",
                                bins=bins)
            _overlay_scatter_multi(arrays, labels,
                                   os.path.join(layer_dir, "overlay_scatter_last.png"),
                                   title=f"{layer} – Value per neuron (last)")
            _overlay_maxgap_multi(arrays, labels,
                                  os.path.join(layer_dir, "overlay_maxgap_last.png"),
                                  title=f"{layer} – Gap from max (last)")

            # per-run basic stats
            for lab, arr, ep in zip(labels, arrays, per_run_epoch):
                s = _basic_stats(arr, cv_mode=cv_mode)
                last_log.append(
                    f"[{layer}] LAST  {lab}@{ep}\n"
                    f" mean={s['mean']:.6f}  median={s['median']:.6f}  var={s['variance']:.6f}  "
                    f"IQR={s['IQR']:.6f}  CV={s['CV']:.6f}  p95={s['p95']:.6f}  "
                    f"skew={s['skewness']:.6f}  gini={s['gini']:.6f}\n"
                )

            # pairwise distances @ last epoch
            n = len(labels)
            for i in range(n):
                for j in range(i+1, n):
                    ai, bj = arrays[i], arrays[j]
                    ks = _ks_distance(ai, bj)
                    w1 = _wasserstein_1d(ai, bj)
                    nmin = min(ai.size, bj.size)
                    rho = spearman_r(ai[:nmin], bj[:nmin])
                    last_log.append(
                        f"[{layer}] LAST  {labels[i]} vs {labels[j]}  "
                        f"KS={ks:.4f}  W1={w1:.6f}  Spearman ρ_s={rho:.4f}\n"
                    )

    with open(os.path.join(out_dir, "comparison_last_epochs.txt"), "w") as f:
        f.write("=== BETWEEN-RUN COMPARISON @ LAST EPOCHS ONLY ===\n")
        f.writelines(last_log)

# ============== CLI ==============

import argparse

def _die(msg: str, code: int = 2):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Compare conductance runs (LAST EPOCH ONLY)")
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

    print(f"[OK] last-epoch comparison complete. outputs in: {args.out}")

if __name__ == "__main__":
    main()
