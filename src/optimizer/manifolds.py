# src/optimizer/manifolds.py
"""Manifold helpers: stereographic S^3 projection and Fisher-Rao simplex geometry."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import geoopt

def _ensure_fisher_rao_path() -> None:
    candidates = [
        Path(__file__).resolve().parents[2] / "fisher_rao",
        Path.home() / "Projects" / "Fisher_Rao",
    ]
    for root in candidates:
        if root.exists() and str(root.parent) not in sys.path:
            sys.path.insert(0, str(root.parent))
            return


_ensure_fisher_rao_path()

try:
    from fisher_rao.metrics import (
        sqrt_embed,
        simplex_to_sphere,
        sphere_to_simplex,
        softmax_to_fr_sphere,
        fisher_rao_distance,
        egrad_to_rgrad_fr,
    )
except ImportError:
    def _normalize_probs(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        p = p.clamp(min=eps)
        return p / p.sum(dim=-1, keepdim=True)

    def sqrt_embed(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        p = _normalize_probs(p, eps=eps)
        s = p.sqrt()
        return s / s.norm(dim=-1, keepdim=True).clamp(min=eps)

    simplex_to_sphere = sqrt_embed

    def sphere_to_simplex(s: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return _normalize_probs(s.pow(2).clamp(min=0.0), eps=eps)

    def softmax_to_fr_sphere(logits: torch.Tensor) -> torch.Tensor:
        return sqrt_embed(F.softmax(logits, dim=-1))

    def fisher_rao_distance(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        p = _normalize_probs(p, eps=eps)
        q = _normalize_probs(q, eps=eps)
        bc = (p.sqrt() * q.sqrt()).sum(dim=-1).clamp(0.0, 1.0)
        return 2.0 * torch.acos(bc.clamp(-1.0 + 1e-7, 1.0 - 1e-7))

    def egrad_to_rgrad_fr(s: torch.Tensor, egrad: torch.Tensor) -> torch.Tensor:
        inner = (s * egrad).sum(dim=-1, keepdim=True)
        return egrad - inner * s


class FisherRaoSphere(geoopt.manifolds.Sphere):
    """
    Unit sphere with Fisher-Rao interpretation via sqrt-embedding of simplices.

    Geodesics and inner products are the standard round metric on S^{n-1},
    which corresponds to the Fisher-Rao metric on Δ^{n-1} under φ(p) = √p/||√p||.
    """

    def simplex_to_manifold(self, p: torch.Tensor) -> torch.Tensor:
        return simplex_to_sphere(p)

    def manifold_to_simplex(self, s: torch.Tensor) -> torch.Tensor:
        return sphere_to_simplex(s)

    def fr_distance_on_simplex(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return fisher_rao_distance(p, q)

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return egrad_to_rgrad_fr(x, u)


def logits_to_fr_manifold_param(
    logits: torch.Tensor,
    manifold: FisherRaoSphere | None = None,
) -> geoopt.ManifoldParameter:
    """Create a ManifoldParameter on the Fisher-Rao sphere from unconstrained logits."""
    if manifold is None:
        manifold = FisherRaoSphere()
    s = softmax_to_fr_sphere(logits)
    return geoopt.ManifoldParameter(s, manifold=manifold)