import torch
from typing import Optional, Tuple


# -------------------------------
# Helpers
# -------------------------------

def _as_tensor_scalar(val, like: torch.Tensor) -> torch.Tensor:
    t = torch.as_tensor(val, device=like.device, dtype=like.dtype)
    if t.numel() != 1:
        raise ValueError("Expected a scalar-like value for a/b/mu.")
    return t


def _validate_params(x: torch.Tensor, a, b, mu):
    if x.numel() == 0:
        raise ValueError("x must be non-empty")
    if not x.is_floating_point():
        raise TypeError("x must be a floating point tensor")
    a = _as_tensor_scalar(a, x)
    b = _as_tensor_scalar(b, x)
    mu = _as_tensor_scalar(mu, x)
    if torch.any(b <= a):
        raise ValueError("Require b > a")
    if not (a <= mu).all() or not (mu <= b).all():
        raise ValueError("Target mean mu must lie within [a, b]")
    return a, b, mu


def _enforce_mean_and_bounds(y0: torch.Tensor, a, b, mu, cap_scale_at_one: bool = True) -> torch.Tensor:
    """
    Given a preliminary y0, return y = mu + t * (y0 - y0.mean())
    choosing the largest nonnegative t that keeps all y within [a,b].
    If cap_scale_at_one=True, additionally clamp t <= 1 (keeps variation similar to y0).
    Order is preserved for t >= 0. Mean is exactly mu for any t.
    """
    a = _as_tensor_scalar(a, y0)
    b = _as_tensor_scalar(b, y0)
    mu = _as_tensor_scalar(mu, y0)
    d = y0 - y0.mean()
    # Compute per-sign constraints on t
    eps = torch.finfo(y0.dtype).eps
    pos = d > eps
    neg = d < -eps
    t_candidates = []
    if pos.any():
        t_candidates.append(((b - mu) / d[pos]).min())
    if neg.any():
        t_candidates.append(((mu - a) / (-d[neg])).min())
    if len(t_candidates) == 0:
        # All d ~ 0: return constant mu
        return mu.expand_as(y0)
    t_max = torch.min(torch.stack(t_candidates))
    if cap_scale_at_one:
        t_max = torch.minimum(t_max, torch.tensor(1.0, device=y0.device, dtype=y0.dtype))
    y = mu + t_max * d
    # Numerical safety clamp
    return torch.clamp(y, min=a.item(), max=b.item())


def _linear_minmax_scale(x: torch.Tensor, a, b) -> torch.Tensor:
    a = _as_tensor_scalar(a, x)
    b = _as_tensor_scalar(b, x)
    xmin = x.min()
    xmax = x.max()
    if (xmax - xmin) <= torch.finfo(x.dtype).eps:
        return (a + b) / 2 * torch.ones_like(x)
    return a + (b - a) * (x - xmin) / (xmax - xmin)


def _median_and_mad(x: torch.Tensor, normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    med = x.median()
    mad = (x - med).abs().median()
    if normalize:
        mad = mad * 1.4826  # consistency with std for Gaussian
    # avoid zero MAD
    mad = torch.clamp(mad, min=torch.finfo(x.dtype).eps)
    return med, mad


def _ranks_average(x: torch.Tensor) -> torch.Tensor:
    """Average ranks (1..n). Ties get the average of their positions.
    Returns a float tensor of shape x, with ranks in [1, n].
    """
    x_flat = x.flatten()
    n = x_flat.numel()
    # Sort
    sorted_vals, sorted_idx = torch.sort(x_flat, stable=True)
    # Group equal neighbors
    uniq, counts = torch.unique_consecutive(sorted_vals, return_counts=True)
    starts = torch.cumsum(torch.cat([torch.zeros(1, device=x.device, dtype=torch.long), counts[:-1]]), dim=0)
    ends = starts + counts - 1
    avg_ranks = (starts.to(x.dtype) + ends.to(x.dtype)) / 2 + 1.0
    ranks_sorted = torch.repeat_interleave(avg_ranks, counts)
    ranks = torch.empty_like(ranks_sorted)
    ranks[sorted_idx] = ranks_sorted
    return ranks.view_as(x)


# -------------------------------
# 1) Winsorized min–max
# -------------------------------

def winsorized_minmax_map(
    x: torch.Tensor,
    a: float,
    b: float,
    mu: float,
    lower_q: float = 0.02,
    upper_q: float = 0.98,
    cap_scale_at_one: bool = True,
) -> torch.Tensor:
    """Linear map after clipping x to [q_lower, q_upper]. Then enforce mean μ and [a,b]."""
    a, b, mu = _validate_params(x, a, b, mu)
    ql = torch.quantile(x, lower_q)
    qu = torch.quantile(x, upper_q)
    xw = torch.clamp(x, min=ql.item(), max=qu.item())
    y0 = _linear_minmax_scale(xw, a, b)
    return _enforce_mean_and_bounds(y0, a, b, mu, cap_scale_at_one)


# -------------------------------
# 2) Robust z-score + logistic squash
# -------------------------------

def robust_logistic_map(
    x: torch.Tensor,
    a: float,
    b: float,
    mu: float,
    lam: float = 1.0,
    calibrate_theta: bool = True,
    max_iter: int = 50,
    tol: float = 1e-6,
    cap_scale_at_one: bool = True,
) -> torch.Tensor:
    """
    Robustly standardize x via median/MAD, apply p = sigmoid(lam * z + theta),
    scale to [a,b], then enforce exact mean μ and bounds.

    If calibrate_theta=True, choose theta by bisection so mean(p) ≈ (mu-a)/(b-a).
    Otherwise theta=0.
    """
    a, b, mu = _validate_params(x, a, b, mu)
    med, mad = _median_and_mad(x)
    z = (x - med) / mad

    target = ((mu - a) / (b - a)).clamp(0.0, 1.0)

    if calibrate_theta:
        lo = torch.tensor(-20.0, device=x.device, dtype=x.dtype)
        hi = torch.tensor(20.0, device=x.device, dtype=x.dtype)
        for _ in range(max_iter):
            mid = (lo + hi) / 2
            p = torch.sigmoid(lam * z + mid)
            m = p.mean()
            hi = torch.where(m > target, mid, hi)
            lo = torch.where(m < target, mid, lo)
            if (hi - lo).abs().max() < tol:
                break
        theta = (lo + hi) / 2
    else:
        theta = torch.tensor(0.0, device=x.device, dtype=x.dtype)

    p = torch.sigmoid(lam * z + theta)
    y0 = a + (b - a) * p
    return _enforce_mean_and_bounds(y0, a, b, mu, cap_scale_at_one)


# -------------------------------
# 3) Rank → power curve (robust, tunable spacing)
# -------------------------------

def rank_power_map(
    x: torch.Tensor,
    a: float,
    b: float,
    mu: float,
    gamma: float = 1.0,
    cap_scale_at_one: bool = True,
) -> torch.Tensor:
    """
    Map ranks u=(r-0.5)/n through q = u**gamma (gamma<1 spreads low end; >1 compresses it),
    scale to [a,b], then enforce mean μ and bounds.
    Completely order-based => robust to outliers.
    """
    a, b, mu = _validate_params(x, a, b, mu)
    r = _ranks_average(x)
    n = x.numel()
    u = (r - 0.5) / float(n)
    q = torch.clamp(u, 0.0, 1.0) ** gamma
    y0 = a + (b - a) * q
    return _enforce_mean_and_bounds(y0, a, b, mu, cap_scale_at_one)


# -------------------------------
# 4) Isotonic regression toward a linear target (PAVA)
# -------------------------------

def _pava(values: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Pool-Adjacent-Violators Algorithm (increasing). Returns fitted values.
    values: 1D tensor already sorted by the predictor.
    weights: optional 1D positive weights (defaults to 1).
    """
    y = values.detach().clone()
    n = y.numel()
    w = torch.ones_like(y) if weights is None else weights.detach().clone()
    # Work in Python lists for simplicity
    v_stack = []  # block values
    w_stack = []  # block weights
    len_stack = []  # block lengths
    for i in range(n):
        v_stack.append(y[i].item())
        w_stack.append(w[i].item())
        len_stack.append(1)
        # Merge while violation exists
        while len(v_stack) >= 2 and v_stack[-2] > v_stack[-1]:
            v2, v1 = v_stack[-2], v_stack[-1]
            w2, w1 = w_stack[-2], w_stack[-1]
            L2, L1 = len_stack[-2], len_stack[-1]
            new_w = w1 + w2
            new_v = (v1 * w1 + v2 * w2) / new_w
            v_stack[-2] = new_v
            w_stack[-2] = new_w
            len_stack[-2] = L1 + L2
            # pop last
            v_stack.pop(); w_stack.pop(); len_stack.pop()
    # Expand back to length n
    fitted = torch.empty(n, dtype=values.dtype, device=values.device)
    idx = 0
    for v, L in zip(v_stack, len_stack):
        fitted[idx:idx+L] = v
        idx += L
    return fitted


def isotonic_regression_map(
    x: torch.Tensor,
    a: float,
    b: float,
    mu: float,
    cap_scale_at_one: bool = True,
) -> torch.Tensor:
    """
    Build a simple linear target over [a,b] from x, then project it to a monotone sequence via isotonic regression (PAVA).
    Finally enforce mean μ and bounds.
    """
    a, b, mu = _validate_params(x, a, b, mu)
    # Linear target in [a,b]
    t = _linear_minmax_scale(x, a, b)
    # Sort by x and fit isotonic
    x_flat = x.flatten()
    order = torch.argsort(x_flat, stable=True)
    t_sorted = t.flatten()[order]
    z_sorted = _pava(t_sorted)
    # Unsort back
    z = torch.empty_like(z_sorted)
    z[order] = z_sorted
    y0 = z.view_as(x)
    return _enforce_mean_and_bounds(y0, a, b, mu, cap_scale_at_one)


# -------------------------------
# 5) Simple z-score → linear map → clamp
# -------------------------------

def zscore_clamp_map(
    x: torch.Tensor,
    a: float,
    b: float,
    mu: float,
    c: Optional[float] = None,
    cap_scale_at_one: bool = True,
) -> torch.Tensor:
    """
    Standardize, map linearly around mu via y' = mu + c*z, clamp to [a,b], then enforce mean μ and bounds.
    If c is None, choose c so that ±2 std dev roughly span half the range.
    """
    a, b, mu = _validate_params(x, a, b, mu)
    mean = x.mean()
    std = x.std(unbiased=False)
    std = torch.clamp(std, min=torch.finfo(x.dtype).eps)
    z = (x - mean) / std
    if c is None:
        c = 0.25 * (b - a)  # so ~±2σ -> ~±0.5*(b-a)
    y_prime = mu + _as_tensor_scalar(c, x) * z
    y0 = torch.clamp(y_prime, min=a.item(), max=b.item())
    return _enforce_mean_and_bounds(y0, a, b, mu, cap_scale_at_one)


# -------------------------------
# 6) Rank-even blend around μ (zero tuning)
# -------------------------------

def rank_even_blend_map(
    x: torch.Tensor,
    a: float,
    b: float,
    mu: float,
) -> torch.Tensor:
    """
    Evenly spaced ranks in [a,b], then blend with mu: y = mu + t*(y_rank - mu) with the
    maximum t that keeps y within [a,b]. Mean is exactly mu by construction.
    """
    a, b, mu = _validate_params(x, a, b, mu)
    r = _ranks_average(x)
    n = x.numel()
    u = (r - 0.5) / float(n)
    y_rank = a + (b - a) * u
    d = y_rank - mu
    pos = d > 0
    neg = d < 0
    t_candidates = []
    if pos.any():
        t_candidates.append(((b - mu) / d[pos]).min())
    if neg.any():
        t_candidates.append(((mu - a) / (-d[neg])).min())
    if len(t_candidates) == 0:
        return mu.expand_as(x)
    t_max = torch.min(torch.stack(t_candidates))
    y = mu + t_max * d
    return torch.clamp(y, min=a.item(), max=b.item())


# -------------------------------
# 7) Yeo–Johnson transform → min–max
# -------------------------------

def _yeo_johnson(x: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    x_pos = x >= 0
    out = torch.empty_like(x)
    # x >= 0
    if abs(lam) > 1e-12:
        out[x_pos] = ((x[x_pos] + 1.0) ** lam - 1.0) / lam
    else:
        out[x_pos] = torch.log(x[x_pos] + 1.0)
    # x < 0
    lam2 = 2.0 - lam
    if abs(lam2) > 1e-12:
        out[~x_pos] = -(((1.0 - x[~x_pos]) ** lam2 - 1.0) / lam2)
    else:
        out[~x_pos] = -torch.log(1.0 - x[~x_pos])
    return out


def yeo_johnson_map(
    x: torch.Tensor,
    a: float,
    b: float,
    mu: float,
    lam: float = 1.0,
    cap_scale_at_one: bool = True,
) -> torch.Tensor:
    """
    Apply Yeo–Johnson to reduce skew while allowing negatives, then min–max to [a,b],
    then enforce mean μ and bounds.
    """
    a, b, mu = _validate_params(x, a, b, mu)
    xt = _yeo_johnson(x, lam)
    y0 = _linear_minmax_scale(xt, a, b)
    return _enforce_mean_and_bounds(y0, a, b, mu, cap_scale_at_one)


# -------------------------------
# Convenience: quick dispatcher
# -------------------------------

METHODS = {
    "winsor": winsorized_minmax_map,
    "robust_logistic": robust_logistic_map,
    "rank_power": rank_power_map,
    "isotonic": isotonic_regression_map,
    "zscore_clamp": zscore_clamp_map,
    "rank_blend": rank_even_blend_map,
    "yeo_johnson": yeo_johnson_map,
}


if __name__ == "__main__":
    # Minimal smoke test (run if you paste into a Python file)
    torch.manual_seed(0)
    x = torch.randn(20) * 2 - 1  # can be negative/positive
    a, b, mu = 10.0, 20.0, 15.0
    for name, fn in METHODS.items():
        y = fn(x, a, b, mu)
        assert (y >= a - 1e-5).all() and (y <= b + 1e-5).all(), name
        print(name, float(y.mean()))
