"""
Fisher-Rao and f-divergence losses for distribution comparison.

Use when comparing spectral envelopes, modal amplitude distributions,
or any normalized non-negative feature vectors interpreted as probabilities.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .metrics import fisher_rao_distance, bhattacharyya_coefficient, _normalize_probs


def fisher_rao_loss(
    p: torch.Tensor,
    q: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Squared Fisher-Rao distance loss."""
    d = fisher_rao_distance(p, q)
    loss = d.pow(2)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def hellinger_loss(p: torch.Tensor, q: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Hellinger distance squared: 2(1 - BC(p,q))."""
    bc = bhattacharyya_coefficient(p, q)
    loss = 2.0 * (1.0 - bc)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def kl_from_probs(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KL(p || q) with numerical stabilization."""
    p = _normalize_probs(p, eps=eps)
    q = _normalize_probs(q, eps=eps)
    return (p * (p.log() - q.log())).sum(dim=-1)


def spectral_fr_loss(
    pred_spectrum: torch.Tensor,
    target_spectrum: torch.Tensor,
    temperature: float = 1.0,
    use_log_domain: bool = True,
) -> torch.Tensor:
    """
    Fisher-Rao loss on normalized spectral magnitude distributions.

    Args:
        pred_spectrum:  (..., F) non-negative magnitudes
        target_spectrum: same shape
        temperature: softmax temperature (lower = peakier distribution)
        use_log_domain: if True, apply log1p before softmax (audio spectra)
    """
    if use_log_domain:
        pred = torch.log1p(pred_spectrum.clamp(min=0.0))
        target = torch.log1p(target_spectrum.clamp(min=0.0))
    else:
        pred = pred_spectrum
        target = target_spectrum

    pred_p = F.softmax(pred / temperature, dim=-1)
    target_p = F.softmax(target / temperature, dim=-1)
    return fisher_rao_loss(pred_p, target_p)


def modal_distribution_fr_loss(
    mode_amps: torch.Tensor,
    target_amps: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fisher-Rao distance between normalized modal amplitude distributions.

    If target_amps is None, compares to uniform over modes (maximum entropy reference).
    """
    mode_amps = mode_amps.clamp(min=0.0)
    p = mode_amps / mode_amps.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    if target_amps is None:
        k = mode_amps.shape[-1]
        q = torch.full_like(p, 1.0 / k)
    else:
        target_amps = target_amps.clamp(min=0.0)
        q = target_amps / target_amps.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    return fisher_rao_loss(p, q)