"""Fisher-Rao metric on probability simplices via sqrt-sphere embedding."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalize_probs(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp(min=eps)
    return p / p.sum(dim=-1, keepdim=True)


def sqrt_embed(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Map a probability vector to the positive orthant of the unit sphere."""
    p = _normalize_probs(p, eps=eps)
    s = p.sqrt()
    return s / s.norm(dim=-1, keepdim=True).clamp(min=eps)


def simplex_to_sphere(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Alias for sqrt_embed."""
    return sqrt_embed(p, eps=eps)


def sphere_to_simplex(s: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Inverse map: square sphere coordinates and renormalize to the simplex."""
    p = s.pow(2).clamp(min=0.0)
    return _normalize_probs(p, eps=eps)


def softmax_to_fr_sphere(logits: torch.Tensor) -> torch.Tensor:
    """Softmax to simplex to Fisher-Rao sphere embedding."""
    return sqrt_embed(F.softmax(logits, dim=-1))


def bhattacharyya_coefficient(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """BC(p, q) = sum sqrt(p_i q_i), in [0, 1] for normalized distributions."""
    p = _normalize_probs(p, eps=eps)
    q = _normalize_probs(q, eps=eps)
    return (p.sqrt() * q.sqrt()).sum(dim=-1).clamp(0.0, 1.0)


def fisher_rao_distance(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Fisher-Rao geodesic distance between distributions."""
    bc = bhattacharyya_coefficient(p, q, eps=eps)
    d = 2.0 * torch.acos(bc.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
    return torch.where(bc >= 1.0 - 1e-6, torch.zeros_like(d), d)


def fisher_rao_inner(s: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Round-sphere inner product at point s (Fisher-Rao metric in sqrt coords)."""
    return (u * v).sum(dim=-1)


def egrad_to_rgrad_fr(s: torch.Tensor, egrad: torch.Tensor) -> torch.Tensor:
    """Convert Euclidean gradient w.r.t. sphere coords to Riemannian gradient."""
    inner = (s * egrad).sum(dim=-1, keepdim=True)
    return egrad - inner * s


def probs_to_tangent(p: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Project an ambient vector v to the Fisher-Rao tangent space at p."""
    s = sqrt_embed(p, eps=eps)
    t = v / (2.0 * s.clamp(min=eps))
    inner = (s * t).sum(dim=-1, keepdim=True)
    return t - inner * s