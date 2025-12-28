import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

def estimate_id_twonn(X_np, fraction_to_keep=1.0, eps=1e-12):
    """
    TWO-NN (Two Nearest Neighbors) intrinsic dimension estimator.
    
    This method estimates the intrinsic dimension of data by analyzing the 
    distribution of ratios between distances to the 2nd and 1st nearest neighbors.
    
    Theory: If data lies on a d-dimensional manifold, the ratio mu = r2/r1 
    follows a distribution where F(mu) = 1 - mu^(-d), which gives a linear 
    relationship: -log(1-F) = d * log(mu)
    
    Args:
        X_np: numpy array of shape [N, D] with data points.
        fraction_to_keep: float in (0,1], optional.
            If <1, only the central fraction of points (w.r.t. CDF)
            is used for slope fitting, for robustness against outliers.
        eps: small constant for numerical stability
            
    Returns:
        d_hat: float, estimated intrinsic dimension.
    """
    N, D = X_np.shape
    # Early exit for degenerate cases
    if N < 3:
        raise ValueError(f"Need at least 3 points for TWO-NN, got {N}")
    # ---- 1. L2-normalize each sample (optional but common) ----
    norms = np.linalg.norm(X_np, axis=1, keepdims=True) + eps
    X_norm = X_np / norms


    # ---- 2. Build k-NN structure and query 3 neighbors ----
    # We ask for 3 neighbors because the closest one is the point itself.
    nn = NearestNeighbors(n_neighbors=3, algorithm="auto", metric='euclidean').fit(X_norm)
    distances, _ = nn.kneighbors(X_norm)  # shape: [N, 3]
    
    # distances[:, 0] is 0 (distance to itself), so we use [:,1] and [:,2]
    r1 = distances[:, 1]
    r2 = distances[:, 2]

    # ---- 3. Remove degenerate points ----
    # Filter out points where 1st neighbor is too close (numerical issues)
    # or where r2 <= r1 (shouldn't happen but check anyway)
    mask = (r1 > eps) & (r2 > r1)

    r1 = r1[mask]
    r2 = r2[mask]
    if len(r1) < 10:
        raise ValueError(f"Too few valid points after filtering: {len(r1)}")

    # ---- 4. Compute ratios mu = r2 / r1 ----
    mu = r2 / r1

    # ---- 5. Sort ratios and build empirical CDF ----
    mu_sorted = np.sort(mu)
    M = len(mu_sorted)
    F = (np.arange(1, M + 1) - 0.5) / M  # CDF values in (0,1)

    # ---- 6. Optionally keep only central fraction of data ----
    if fraction_to_keep < 1.0:
        low = (1.0 - fraction_to_keep) / 2.0
        high = 1.0 - low
        keep = (F >= low) & (F <= high)
        mu_sorted = mu_sorted[keep]
        F = F[keep]
        M = len(mu_sorted)
        if M < 10:
            raise ValueError(f"Too few points after keeping central fraction: {M}")

    # ---- 7. Map to (x, y) = (log mu, -log(1 - F)) ----
    x = np.log(mu_sorted + eps)
    y = -np.log(1.0 - F + eps)

    # ---- 8. Fit line through origin: y ≈ d * x ----
    d_hat = (x @ y) / (x @ x + eps)  # slope via least squares

    return float(d_hat)


def estimate_vit_layer_id_twonn(layer_out, max_points=20000, fraction_to_keep=0.9, apply_relu=False,only_cls=False):
    """
    Estimate intrinsic dimension of a ViT layer given its outputs for a batch.
    
    Args:
        layer_out: torch.Tensor of shape [B, T, D]
            B = batch_size
            T = token_amount
            D = hidden_dimension
        max_points: int, maximum number of points to use for ID estimation.
            If B*T > max_points, we will randomly subsample points.
        fraction_to_keep: passed to TWO-NN; central fraction of points
            to use for regression (helps robustness).
            
    Returns:
        d_hat: float, estimated intrinsic dimension.
    """
    # ---- 1. layer_out should be detached from graph and on CPU ----
    # If it's already detached/CPU, this will be cheap.
    with torch.no_grad():
        if apply_relu:
            acts = torch.nn.functional.gelu(layer_out)
        else:
            acts = layer_out
        x = acts.detach().cpu()  # [B, T, D]
    B, T, D = x.shape

    # ---- 2. Flatten [B, T, D] -> [N_points, D] ----
    # Each (image, token) pair = a point in D-dimensional space.
    if only_cls:
        x = x[:,0:1,:]  # Keep only CLS token
        T = 1
    x_flat = x.reshape(B * T, D)  # [N_points, D]
    N_points = x_flat.shape[0]

    # ---- 3. Optionally subsample to avoid quadratic cost when N is huge ----
    # TWO-NN is relatively cheap but k-NN on too many points can be slow.
    if N_points > max_points:
        # Randomly choose max_points indices
        idx = torch.randperm(N_points)[:max_points]
        x_flat = x_flat[idx]
        N_points = max_points

    # ---- 4. Convert to NumPy for scikit-learn ----
    X_np = x_flat.numpy().astype(np.float32)

    # ---- 5. Call TWO-NN estimator ----
    try:
        d_hat = estimate_id_twonn(X_np, fraction_to_keep=fraction_to_keep)
    except ValueError as e:
        # Handle edge cases gracefully
        print(f"Warning: TWO-NN estimation failed: {e}")
        return float(D)  # Return ambient dimension as fallback

    return d_hat

def estimate_attention_pattern_id_per_head(attn,
                                           max_points_per_head=20000,
                                           fraction_to_keep=0.9,
                                           use_log=False):
    """
    Estimate intrinsic dimension of attention *patterns* separately per head.

    Args:
        attn: torch.Tensor of shape [B, H, T, T]
              Attention weights after softmax (probabilities).
        max_points_per_head: int, maximum number of rows (query patterns)
              to use per head. If B*T > this, we randomly subsample rows.
        fraction_to_keep: float in (0,1], central fraction of points
              used in the TWO-NN regression (for robustness).
        use_log: bool, if True apply log-transform to probabilities
              before computing ID (helps leave the simplex geometry).

    Returns:
        id_per_head: torch.Tensor of shape [H]
              Intrinsic dimension estimate for each head.
    """
    with torch.no_grad():
        a = attn.detach().cpu()   # ensure on CPU and detached

    B, H, T, _ = a.shape
    ids = []

    for h in range(H):
        # ---- 1. Select one head: [B, T, T] ----
        head_attn = a[:, h, :, :]  # [B, T, T]

        # ---- 2. Flatten rows: each query token's attention dist = one point ----
        # Shape becomes [B*T, T]
        rows = head_attn.reshape(B * T, T)
        N_points = rows.shape[0]

        # ---- 3. Optional subsampling for this head ----
        if N_points > max_points_per_head:
            idx = torch.randperm(N_points)[:max_points_per_head]
            rows = rows[idx]
            N_points = max_points_per_head

        # ---- 4. Optional log-transform to leave probability simplex ----
        if use_log:
            eps = 1e-8
            rows = torch.log(rows + eps)

        # ---- 5. Convert to NumPy and call TWO-NN estimator ----
        X_np = rows.numpy().astype(np.float32)

        d_hat = estimate_id_twonn(X_np, fraction_to_keep=fraction_to_keep)
        ids.append(d_hat)

    # Convert list of floats to a torch tensor for convenience
    id_per_head = torch.tensor(ids) # [H]

    return id_per_head