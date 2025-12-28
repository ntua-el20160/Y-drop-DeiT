import torch

def effective_rank_from_acts(pre_acts: torch.Tensor,
                             center: bool = True,
                             apply_relu: bool = True,
                             eps: float = 1e-12,
                             only_cls: bool = False) -> torch.Tensor:
    """
    Compute the effective rank of a ViT layer's activations.

    Args:
        pre_acts: Tensor of shape [batch_size, num_tokens, hidden_dim]
                  (pre-ReLU activations, or post-ReLU if you prefer).
        center:   If True, subtract the mean from each feature (recommended).
        eps:      Small constant for numerical stability.

    Returns:
        A 0-D tensor (scalar) with the effective rank.
    """
    if apply_relu:
        acts = torch.nn.functional.gelu(pre_acts)
    else:
        acts = pre_acts
    # pre_acts: [B, T, D]
    B, T, D = acts.shape


    # 1) Flatten batch and tokens into a single "sample" dimension: [N, D]
    if only_cls:
        acts = acts[:,0:1,:]  # Keep only CLS token
        T = 1
    Z = acts.reshape(B * T, D)   # N = B * T

    # 2) Optional: center each feature (column)
    if center:
        Z = Z - Z.mean(dim=0, keepdim=True)

    # 3) Build Gram / covariance-like matrix: [D, D]
    #    G = (1/N) * Z^T Z
    N = Z.shape[0]
    G = (Z.T @ Z) / max(N, 1)

    # 4) Eigenvalues of G (G is symmetric PSD, so use eigvalsh)
    evals = torch.linalg.eigvalsh(G)      # [D]

    # Numerical clean-up: clamp to non-negative
    evals = torch.clamp(evals, min=0.0)
    participation_ratio = (evals.sum() ** 2) / ( (evals ** 2).sum() + eps)

    # 5) Turn eigenvalues into a probability distribution
    trace = evals.sum()
    if trace <= eps:
        # All-zero activations (or extremely tiny) → no meaningful rank
        return torch.tensor(0.0, device=acts.device)

    p = evals / trace   # each p_i >= 0, sum p_i = 1

    # 6) Shannon entropy H = -sum p_i log p_i
    entropy = -(p * (p + eps).log()).sum()

    # 7) Effective rank = exp(H)
    eff_rank = torch.exp(entropy)

    return eff_rank, participation_ratio
import torch

def effective_rank_from_attn(attn: torch.Tensor,
                             center: bool = True,
                             eps: float = 1e-12) -> torch.Tensor:
    """
    Compute effective rank per head from multi-head attention weights.

    Args:
        attn:  Attention weights of shape [B, H, T, T],
               typically softmax(QK^T / sqrt(d_head)).
        center: If True, subtract mean from each "feature" (column) before
                building the Gram matrix. This can help stabilize the estimate.
        eps:   Small constant for numerical stability.

    Returns:
        eranks: Tensor of shape [H], effective rank for each head.
    """
    if attn.dim() != 4:
        raise ValueError(f"Expected attn of shape [B, H, T, T], got {attn.shape}")

    B, H, T_q, T_k = attn.shape
    if T_q != T_k:
        raise ValueError("Only self-attention supported here; got T_q != T_k")

    device = attn.device
    eranks = []
    participation_ratios = []

    for h in range(H):
        # 1) Extract one head: [B, T, T]
        A = attn[:, h, :, :]   # [B, T, T]

        # 2) Flatten (sample, query_token) into one dimension: [N, T]
        #    N = B * T  (each row is one attention pattern over tokens)
        A_flat = A.reshape(B * T_q, T_k)  # [N, T]

        # 3) Optional centering across samples
        if center:
            A_flat = A_flat - A_flat.mean(dim=0, keepdim=True)

        N = A_flat.shape[0]
        if N == 0:
            eranks.append(torch.tensor(0.0, device=device))
            continue

        # 4) Gram matrix over tokens: G = (1/N) * A_flat^T A_flat  → [T, T]
        G = (A_flat.T @ A_flat) / float(N)

        # 5) Eigenvalues of symmetric PSD Gram matrix
        evals = torch.linalg.eigvalsh(G)          # [T]
        evals = torch.clamp(evals, min=0.0)       # avoid tiny negatives from numerics
        
        participation_ratio = (evals.sum() ** 2) / ( (evals ** 2).sum() + eps)
        participation_ratios.append(participation_ratio)
        trace = evals.sum()
        if trace <= eps:
            # Almost no variance; attention patterns are effectively constant
            eranks.append(torch.tensor(0.0, device=device))
            continue

        # 6) Turn eigenvalues into probability distribution
        p = evals / trace

        # 7) Shannon entropy H = -sum p_i log p_i
        entropy = -(p * (p + eps).log()).sum()

        # 8) Effective rank = exp(H)
        eff_rank = torch.exp(entropy)

        eranks.append(eff_rank)

    # Stack per-head ranks: [H]
    return torch.stack(eranks), torch.stack(participation_ratios)
