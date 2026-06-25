"""Fisher-Rao spectral and modal losses for physics_audio training."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_FISHER_RAO_ROOT = Path.home() / "Projects" / "Fisher_Rao"
if _FISHER_RAO_ROOT.exists() and str(_FISHER_RAO_ROOT) not in sys.path:
    sys.path.insert(0, str(_FISHER_RAO_ROOT))

try:
    from fisher_rao.losses import spectral_fr_loss, modal_distribution_fr_loss, fisher_rao_loss
    from fisher_rao.invariants import scale_quotient
except ImportError:
    import torch.nn.functional as F

    def fisher_rao_loss(p, q, reduction="mean"):
        p = p.clamp(min=1e-8) / p.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        q = q.clamp(min=1e-8) / q.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        bc = (p.sqrt() * q.sqrt()).sum(dim=-1).clamp(0, 1)
        d = 2.0 * torch.acos(bc.clamp(-1 + 1e-7, 1 - 1e-7))
        loss = d.pow(2)
        return loss.mean() if reduction == "mean" else loss

    def spectral_fr_loss(pred_spectrum, target_spectrum, temperature=1.0, use_log_domain=True):
        if use_log_domain:
            pred = torch.log1p(pred_spectrum.clamp(min=0.0))
            target = torch.log1p(target_spectrum.clamp(min=0.0))
        else:
            pred, target = pred_spectrum, target_spectrum
        return fisher_rao_loss(F.softmax(pred / temperature, dim=-1), F.softmax(target / temperature, dim=-1))

    def modal_distribution_fr_loss(mode_amps, target_amps=None):
        p = mode_amps.clamp(min=0) / mode_amps.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        if target_amps is None:
            q = torch.full_like(p, 1.0 / p.shape[-1])
        else:
            q = target_amps.clamp(min=0) / target_amps.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return fisher_rao_loss(p, q)

    def scale_quotient(x, mode="l2"):
        if mode == "sum":
            return x / x.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def stft_bin_fr_loss(
    spec_pred: torch.Tensor,
    spec_true: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Fisher-Rao loss on per-frame STFT magnitude distributions.

    spec_pred, spec_true: (frames, bins) or (bins,) magnitude tensors.
    """
    if spec_pred.dim() == 1:
        return spectral_fr_loss(spec_pred, spec_true, temperature=temperature)
    return spectral_fr_loss(spec_pred, spec_true, temperature=temperature)


def mode_amplitude_fr_loss(
    predicted_amps: torch.Tensor,
    observed_amps: torch.Tensor | None = None,
    scale_invariant: bool = True,
) -> torch.Tensor:
    """
    Fisher-Rao loss on normalized modal amplitude distributions.

    scale_invariant: quotient global scale before comparing (group invariance).
    """
    p = predicted_amps
    q = observed_amps
    if scale_invariant:
        p = scale_quotient(p, mode="sum")
        if q is not None:
            q = scale_quotient(q, mode="sum")
    return modal_distribution_fr_loss(p, q)