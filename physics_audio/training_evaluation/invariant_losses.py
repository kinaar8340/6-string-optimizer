"""Double-coset invariant losses for physics_audio (Nielsen & Okamura 2026)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
_FISHER_RAO_HOME = Path.home() / "Projects" / "Fisher_Rao"

for candidate in (_ROOT, _FISHER_RAO_HOME):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    from fisher_rao.invariants import location_scale_pair_invariant
except ImportError:

    def location_scale_pair_invariant(mu1, V1, mu2, V2):
        mu1 = mu1.float().flatten()
        mu2 = mu2.float().flatten()
        V1 = V1.float()
        V2 = V2.float()
        nu = torch.linalg.solve(V2, mu1 - mu2)
        S = torch.linalg.solve(V2, V1)
        singular_values = torch.linalg.svdvals(S)
        block_norms = nu.norm().unsqueeze(0)
        from collections import namedtuple

        Inv = namedtuple(
            "LocationScaleMaximalInvariant",
            ["singular_values", "multiplicities", "block_norms", "relative_scale", "whitened_location"],
        )
        return Inv(singular_values, [singular_values.numel()], block_norms, S, nu)


def harmonic_design_matrix(
    k_modes: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Columns [1, k-1] for log-linear damping envelope d_k = base + slope * (k-1)."""
    harmonics = torch.arange(1, k_modes + 1, device=device, dtype=dtype)
    return torch.stack([torch.ones_like(harmonics), harmonics - 1.0], dim=1)


def target_damping_coefficients(
    target_damping: torch.Tensor,
    design: torch.Tensor | None = None,
) -> torch.Tensor:
    """Least-squares [base, slope] coefficients for a K-mode damping target."""
    target = target_damping.float().flatten()
    if design is None:
        design = harmonic_design_matrix(target.numel(), target.device, target.dtype)
    solution = torch.linalg.lstsq(design, target.unsqueeze(-1)).solution
    return solution.squeeze(-1)


def learned_damping_coefficients(
    log_base_rate: torch.Tensor,
    log_slope: torch.Tensor,
) -> torch.Tensor:
    """Model damping coefficients in the same [base, slope] basis."""
    base = torch.exp(log_base_rate.float().reshape(()))
    slope = torch.exp(log_slope.float().reshape(()))
    return torch.stack([base, slope])


def damping_invariant_loss(
    log_base_rate: torch.Tensor,
    log_slope: torch.Tensor,
    target_damping: torch.Tensor,
    singular_weight: float = 0.0,
) -> torch.Tensor:
    """
    Prop 3.3 invariant loss on the 2D damping sufficient statistic.

    Compares learned (base, slope) to the LS fit of ``target_damping`` in
    coefficient space with V1 = V2 = I_2.  Block norms reduce to the
    whitened coefficient residual; invariant to global reparameterization
    of the shared harmonic design subspace.
    """
    target = target_damping.float().flatten()
    k_modes = target.numel()
    design = harmonic_design_matrix(k_modes, target.device, target.dtype)

    mu1 = learned_damping_coefficients(log_base_rate, log_slope)
    mu2 = target_damping_coefficients(target, design)
    eye = torch.eye(2, device=target.device, dtype=target.dtype)

    inv = location_scale_pair_invariant(mu1, eye, mu2, eye)

    loss = inv.block_norms.pow(2).sum()
    if singular_weight > 0.0:
        loss = loss + singular_weight * (inv.singular_values - 1.0).pow(2).sum()
    return loss