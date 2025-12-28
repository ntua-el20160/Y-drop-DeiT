import torch

def count_post_relu_nonzeros(tensor,only_cls=False) -> int:
    """
    Count the number of non-zero elements in the tensor after applying ReLU.
    
    Args:
        tensor: torch.Tensor of any shape.
    Returns:
        int: Number of non-zero elements after ReLU.
    """
    if only_cls:
        tensor = tensor[:,0:1,:]  # Keep only CLS token
    relu_tensor = torch.relu(tensor)
    nonzero_count = torch.count_nonzero(relu_tensor).item()
    return nonzero_count

def per_head_entropy_from_attn(attn: torch.Tensor) -> torch.Tensor:
    """
    Compute the entropy of attention distributions per head.

    Args:
        attn: torch.Tensor of shape [B, H, T, T]
              Attention weights after softmax (probabilities).

    Returns:
        entropies: torch.Tensor of shape [H]
              Entropy of attention distributions for each head.
    """
    B, H, T, _ = attn.shape
    entropies = []

    for h in range(H):
        # Select one head: [B, T, T]
        head_attn = attn[:, h, :, :]  # [B, T, T]

        # Flatten batch and tokens: [B*T, T]
        rows = head_attn.reshape(B * T, T)

        # Compute entropy for each row
        eps = 1e-12
        log_rows = torch.log(rows + eps)
        entropy_per_row = -torch.sum(rows * log_rows, dim=1)  # [B*T]

        # Average entropy over all rows for this head
        avg_entropy = entropy_per_row.mean().item()
        entropies.append(avg_entropy)

    return torch.tensor(entropies)