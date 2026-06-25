"""Fisher-Rao feature extraction for physics_audio losses."""

from __future__ import annotations

import torch


def mode_amplitudes_from_stiefel(points: torch.Tensor) -> torch.Tensor:
    """
    Normalized modal energy distribution from Stiefel observations.

    points: (T, dim, K) → amplitudes (K,) proportional to per-mode RMS energy.
    """
    amps = points.float().pow(2).mean(dim=(0, 1))
    return amps.clamp(min=1e-12)


def modal_spectral_envelope(points: torch.Tensor) -> torch.Tensor:
    """
    Per-time-step mode energy vectors for Fisher-Rao spectral comparison.

    points: (T, dim, K) → (T, K)
    """
    return points.float().pow(2).mean(dim=1).clamp(min=1e-12)


def fr_loss_kwargs_from_batch(
    preds: torch.Tensor,
    data_points: torch.Tensor,
    *,
    fr_mode_weight: float,
    fr_spectral_weight: float,
    target_mode_amps: torch.Tensor | None = None,
    target_spectrum: torch.Tensor | None = None,
) -> dict:
    """Build optional kwargs for total_loss() from a forward batch."""
    kwargs: dict = {}
    if fr_mode_weight > 0.0:
        kwargs["fr_mode_weight"] = fr_mode_weight
        kwargs["mode_amps"] = mode_amplitudes_from_stiefel(preds)
        kwargs["target_mode_amps"] = (
            mode_amplitudes_from_stiefel(data_points) if target_mode_amps is None else target_mode_amps
        )
    if fr_spectral_weight > 0.0:
        kwargs["fr_spectral_weight"] = fr_spectral_weight
        kwargs["pred_spectrum"] = modal_spectral_envelope(preds)
        kwargs["target_spectrum"] = (
            modal_spectral_envelope(data_points) if target_spectrum is None else target_spectrum
        )
    return kwargs