# utils.py (ensure this exact version is used – includes float32 casts for all SVD ops)
import torch
from scipy.optimize import linear_sum_assignment
from geoopt.manifolds import Stiefel
from .config import device, DIM, K_MODES
from typing import Tuple
from .config import IDEAL_HARMONICS

manifold = Stiefel()

def stiefel_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Cast to float32 to ensure batched SVD compatibility (no Half support)
    x_fp32 = x.to(torch.float32)
    y_fp32 = y.to(torch.float32)
    cos_angles = torch.linalg.svdvals(x_fp32.transpose(-2, -1) @ y_fp32)
    cos_angles = torch.clamp(cos_angles, -1.0 + 1e-6, 1.0 - 1e-6)
    angles = torch.acos(cos_angles)
    return angles.norm(p=2, dim=-1).to(x.dtype)

def safe_proj(tensor: torch.Tensor) -> torch.Tensor:
    tensor_fp32 = tensor.to(torch.float32)
    u, s, vh = torch.linalg.svd(tensor_fp32, full_matrices=False)
    return (u @ vh).to(tensor.dtype)

def find_best_perm_sign(overlap: torch.Tensor):
    overlap_detached = overlap.detach()
    overlap_abs = torch.abs(overlap_detached).cpu().numpy()
    cost = -overlap_abs
    row_ind, col_ind = linear_sum_assignment(cost)
    perm = torch.tensor(col_ind, device=overlap.device)

    diag = overlap[torch.arange(K_MODES), perm].detach()
    signs = torch.sign(diag)
    signs[signs == 0] = 1
    return perm, signs

def align_and_compute_freq(
    true_vel_dir: torch.Tensor,
    learned_vel_dir: torch.Tensor,
    damping_rates: torch.Tensor,
    speed_scalars: torch.Tensor,
    inharm_b: torch.Tensor,
    full_freq: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    overlap = true_vel_dir.T @ learned_vel_dir  # shape (K_MODES, K_MODES)

    perm, signs = find_best_perm_sign(overlap)

    aligned_damping_rates = damping_rates[perm]
    aligned_speed_scalars = speed_scalars[perm]
    aligned_inharm_b = inharm_b[perm]
    aligned_learned_freq = full_freq[perm]

    return aligned_damping_rates, aligned_speed_scalars, aligned_learned_freq, aligned_inharm_b

def get_pca_initial_basis(data_points: torch.Tensor, k_modes: int) -> torch.Tensor:
    reshaped = data_points.reshape(-1, DIM)
    reshaped_fp32 = reshaped.to(torch.float32)  # Full float32 path

    n_samples = reshaped.shape[0]

    if n_samples <= 1:
        init = safe_proj(torch.randn(DIM, k_modes, device=device))
        print("Stable PCA init | Degenerate → random orthonormal")
        return init

    mean_col = reshaped_fp32.mean(dim=0)
    total_var = (reshaped_fp32 - mean_col).pow(2).sum() / (n_samples - 1)

    try:
        U, S, V = torch.pca_lowrank(reshaped_fp32, q=k_modes, center=True, niter=6)
        initial_basis = V
        captured_var = S.pow(2).sum() / (n_samples - 1)
        ratio = captured_var / total_var if total_var > 1e-12 else 1.0
    except Exception:
        print("pca_lowrank failed, falling back to SVD")
        centered = reshaped_fp32 - mean_col.unsqueeze(0)
        _, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        initial_basis = Vh.t()[:, :k_modes]
        captured_var = S[:k_modes].pow(2).sum() / (n_samples - 1)
        ratio = captured_var / total_var if total_var > 1e-12 else 1.0

    initial_basis = safe_proj(initial_basis)
    print(f"Stable PCA init | Captured variance (top {k_modes}): {ratio:.4f}")
    return initial_basis