"""Fisher-Rao feature extraction for physics_audio losses."""

from __future__ import annotations

import torch


def extract_coupling_skew(model) -> torch.Tensor:
    """Skew-symmetric coupling matrix from model.coupling_raw (no librosa dep)."""
    raw = model.coupling_raw
    return raw.tril(diagonal=-1) - raw.triu(diagonal=1)


def resolve_target_mode_amps(
    data_points: torch.Tensor,
    prior_targets: dict | None = None,
    override: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Target modal amplitudes for FR losses.

    Prefers piptrack time-average (``partial_amps_mean`` in prior_targets),
    then explicit override, then Stiefel RMS energy from observations.
    """
    if override is not None:
        return override
    if prior_targets is not None and "partial_amps_mean" in prior_targets:
        return prior_targets["partial_amps_mean"]
    return mode_amplitudes_from_stiefel(data_points)


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
    fr_invariant_weight: float = 0.0,
    fr_invariant_modal: float = 0.0,
    target_mode_amps: torch.Tensor | None = None,
    target_spectrum: torch.Tensor | None = None,
) -> dict:
    """Build optional kwargs for total_loss() from a forward batch."""
    kwargs: dict = {}
    need_modal = fr_mode_weight > 0.0 or (fr_invariant_weight > 0.0 and fr_invariant_modal > 0.0)
    if need_modal:
        kwargs["mode_amps"] = mode_amplitudes_from_stiefel(preds)
        kwargs["target_mode_amps"] = (
            mode_amplitudes_from_stiefel(data_points) if target_mode_amps is None else target_mode_amps
        )
    if fr_mode_weight > 0.0:
        kwargs["fr_mode_weight"] = fr_mode_weight
    if fr_spectral_weight > 0.0:
        kwargs["fr_spectral_weight"] = fr_spectral_weight
        kwargs["pred_spectrum"] = modal_spectral_envelope(preds)
        kwargs["target_spectrum"] = (
            modal_spectral_envelope(data_points) if target_spectrum is None else target_spectrum
        )
    return kwargs