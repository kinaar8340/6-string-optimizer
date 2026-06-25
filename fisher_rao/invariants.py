"""
Group-invariant reductions for transformation models.

Derivation: Nielsen & Okamura, "Group invariance of f-divergences and the
Fisher-Rao distance" (see ~/Projects/Fisher_Rao/proof.pdf).

Key results used here:
- Theorem 2.1: f-divergences invariant under transformation model (2.2)
- Theorem 2.3: transitive actions → maximal invariant is double coset H g1^{-1} g2 H
- Proposition 3.3: location-scale pair invariant = singular values of S = V2^{-1}V1
  plus block norms of U^T nu where nu = V2^{-1}(mu1 - mu2)
- Proposition 4.2: Fisher-Rao distance has the same invariant reduction;
  d_FR((mu1,[V1]),(mu2,[V2])) = d_FR((nu,[S]),(0,[Id]))
"""

from __future__ import annotations

import torch
from typing import NamedTuple


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


class LocationScaleMaximalInvariant(NamedTuple):
    """Maximal invariant coordinates from Proposition 3.3 / 4.2 (proof.pdf)."""

    singular_values: torch.Tensor  # (k,) strictly decreasing tau_1 > ... > tau_k
    multiplicities: list[int]       # block sizes m_1, ..., m_k
    block_norms: torch.Tensor       # (k,) norms r_j = ||z^{(j)}||
    relative_scale: torch.Tensor    # S = V2^{-1} V1
    whitened_location: torch.Tensor # nu = V2^{-1}(mu1 - mu2)


def _svd_block_norms(singular_values: torch.Tensor, z: torch.Tensor, tol: float = 1e-6) -> tuple[list[int], torch.Tensor]:
    """Group z by equal singular values; return multiplicities and block norms."""
    sv = singular_values.flatten()
    z = z.flatten()
    d = sv.numel()
    if d == 0:
        return [], torch.tensor([], dtype=z.dtype, device=z.device)

    multiplicities: list[int] = []
    block_norms: list[torch.Tensor] = []
    start = 0
    for i in range(1, d + 1):
        if i == d or (sv[start] - sv[i]).abs() > tol:
            m = i - start
            multiplicities.append(m)
            block_norms.append(z[start:i].norm())
            start = i

    return multiplicities, torch.stack(block_norms)


def location_scale_pair_invariant(
    mu1: torch.Tensor,
    V1: torch.Tensor,
    mu2: torch.Tensor,
    V2: torch.Tensor,
) -> LocationScaleMaximalInvariant:
    """
    Explicit maximal invariant for Aff(d)-equivariant location-scale pairs.

    Implements Proposition 3.3: for theta_i = (mu_i, V_i) in R^d x GL(d,R),
        S = V2^{-1} V1,  nu = V2^{-1}(mu1 - mu2)
    take SVD S = U diag(tau) W^T, z = U^T nu, and block norms r_j = ||z^{(j)}||
    determined by multiplicities of tau.

    Every invariant f-divergence and (Prop 4.2) the Fisher-Rao distance depend
    only on (tau, multiplicities, block_norms).
    """
    mu1 = mu1.float().flatten()
    mu2 = mu2.float().flatten()
    V1 = V1.float()
    V2 = V2.float()

    S = torch.linalg.solve(V2, V1)
    nu = torch.linalg.solve(V2, mu1 - mu2)

    U, singular_values, _ = torch.linalg.svd(S, full_matrices=False)
    z = U.T @ nu
    multiplicities, block_norms = _svd_block_norms(singular_values, z)

    return LocationScaleMaximalInvariant(
        singular_values=singular_values,
        multiplicities=multiplicities,
        block_norms=block_norms,
        relative_scale=S,
        whitened_location=nu,
    )


def fisher_rao_canonical_pair(
    mu1: torch.Tensor,
    V1: torch.Tensor,
    mu2: torch.Tensor,
    V2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Proposition 4.2 reduction: map pair ((mu1,[V1]),(mu2,[V2])) to canonical
    ((nu,[S]),(0,[Id])) with the same Fisher-Rao distance.

    Returns (nu, S) where nu = V2^{-1}(mu1 - mu2), S = V2^{-1} V1.
    """
    inv = location_scale_pair_invariant(mu1, V1, mu2, V2)
    return inv.whitened_location, inv.relative_scale


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