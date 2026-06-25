"""
Group-invariant reductions for transformation models.

Inspired by: "Group invariance of f-divergences and the Fisher–Rao distance"
— distances reduce to maximal invariants under group actions on parameter space.

Practical quotients implemented here:
- Scale invariance (positive scaling symmetry)
- Rotation invariance (orthogonal action → singular values)
- Location-scale (multivariate normal family → relative scale singular values + location norm)
"""

from __future__ import annotations

import torch


def scale_quotient(x: torch.Tensor, mode: str = "l2") -> torch.Tensor:
    """
    Quotient out global positive scaling: x ↦ x / ||x||.

    mode: "l2" (unit norm) or "sum" (probability normalization).
    """
    if mode == "sum":
        return x / x.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def rotation_singular_invariants(matrix: torch.Tensor) -> torch.Tensor:
    """
    Rotation-invariant features: singular values of a (..., m, n) matrix.

    Under O(m) × O(n) actions, singular values are maximal invariants.
    """
    return torch.linalg.svdvals(matrix.float())


def location_scale_invariants(
    loc: torch.Tensor,
    scale: torch.Tensor,
    loc_ref: torch.Tensor | None = None,
    scale_ref: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """
    Maximal invariants for location-scale families (multivariate normal analogue).

    Given location μ and positive-definite scale Σ, compare relative geometry:
    - singular values of Σ_ref^{-1/2} Σ Σ_ref^{-1/2}  (scale shape)
    - ||Σ^{-1/2}(μ - μ_ref)||                          (normalized location offset)

    If references are None, uses identity / zero.
    """
    loc = loc.float()
    scale = scale.float()

    if scale.dim() == 2:
        scale = scale.unsqueeze(0)
    if loc.dim() == 1:
        loc = loc.unsqueeze(0)

    if loc_ref is None:
        loc_ref = torch.zeros_like(loc)
    if scale_ref is None:
        scale_ref = torch.eye(scale.shape[-1], device=scale.device, dtype=scale.dtype)
        if scale.shape[0] > 1:
            scale_ref = scale_ref.unsqueeze(0).expand(scale.shape[0], -1, -1)

    scale_ref = scale_ref.float()
    loc_ref = loc_ref.float()

    # Relative scale: S = L_ref^{-1} Σ L_ref^{-T} via Cholesky
    L_ref = torch.linalg.cholesky(scale_ref + 1e-6 * torch.eye(scale_ref.shape[-1], device=scale.device))
    L = torch.linalg.cholesky(scale + 1e-6 * torch.eye(scale.shape[-1], device=scale.device))
    relative = torch.linalg.solve(L_ref, L)
    rel_singular = torch.linalg.svdvals(relative)

    # Whitened location offset
    delta = loc - loc_ref
    whitened = torch.linalg.solve_triangular(L, delta.unsqueeze(-1), upper=False).squeeze(-1)
    loc_norm = whitened.norm(dim=-1)

    return {
        "relative_scale_singular": rel_singular,
        "whitened_location_norm": loc_norm,
    }


def project_to_invariants(
    x: torch.Tensor,
    symmetries: list[str] | None = None,
) -> torch.Tensor:
    """
    Apply a sequence of symmetry quotients and return a concatenated invariant vector.

    symmetries: subset of ["scale", "rotation"].
    """
    if symmetries is None:
        symmetries = ["scale"]

    features: list[torch.Tensor] = []
    current = x

    for sym in symmetries:
        if sym == "scale":
            current = scale_quotient(current, mode="l2")
            features.append(current.flatten(start_dim=-2))
        elif sym == "rotation" and current.dim() >= 2:
            sv = rotation_singular_invariants(current)
            features.append(sv)
            current = sv
        else:
            features.append(current.flatten(start_dim=-1))

    return torch.cat(features, dim=-1)