# losses.py
# Updated: Fixed missing stiefel_dist import + restored full-precision autocast wrapper
# - Added from .utils import stiefel_dist
# - Wrapped geo_loss in autocast(enabled=False) for SVD safety
# - Kept gentle mean speed prior (with tunable lambda)
# - Added multi-resolution STFT loss for real-audio fitting

import torch
import torch.nn.functional as F
from torch.amp import autocast

from .config import (
    speed_uniform_lambda, coupling_prior_lambda, TRUE_COUPLING_STRENGTH,
    damping_prior_lambda, TRUE_DAMPING_RATES,
    inharm_l2_lambda, inharm_ceiling_lambda, inharm_ceiling_threshold,
    VELOCITY_SCALE_BASE, REAL_AUDIO_STFT_FFT_SIZES, REAL_AUDIO_STFT_HOP_RATIO,
    fr_invariant_weight, fr_invariant_damping, fr_invariant_coupling,
    fr_invariant_coupling_strength_weight, fr_invariant_speed, fr_invariant_inharm,
    fr_invariant_modal,
    fr_replace_mse_priors,
    TRUE_INHARM_B,
)
from .utils import stiefel_dist

speed_mean_prior_lambda = 1.0


def geo_loss(preds, data_points):
    with autocast('cuda', enabled=False):
        return stiefel_dist(preds.float(), data_points.float()).pow(2).mean()


def _mse_prior_skip_terms(
    inv_weight: float,
    replace_mse: bool,
    *,
    inv_damping: float,
    inv_coupling: float,
    inv_speed: float,
    inv_inharm: float,
    log_base_rate: torch.Tensor | None,
    log_slope: torch.Tensor | None,
    coupling_skew: torch.Tensor | None,
) -> frozenset[str]:
    """Terms in prior_loss superseded by active Fisher-Rao invariants (Phase 4)."""
    if inv_weight <= 0.0 or not replace_mse:
        return frozenset()
    skip: set[str] = set()
    if inv_damping > 0.0 and log_base_rate is not None and log_slope is not None:
        skip.add("damping")
    if inv_coupling > 0.0 and coupling_skew is not None:
        skip.add("coupling")
    if inv_speed > 0.0:
        skip.add("speed")
    if inv_inharm > 0.0:
        skip.add("inharm")
    return frozenset(skip)


def prior_loss(
    damping_rates,
    coupling_strength,
    inharm_b,
    speed_scalars,
    prior_targets: dict | None = None,
    skip_mse_terms: frozenset[str] | None = None,
):
    """Physics priors. When prior_targets is set, use estimated real-audio values."""
    skip = skip_mse_terms or frozenset()
    zero = damping_rates.new_tensor(0.0)
    if prior_targets is not None:
        target_damping = prior_targets['damping_rates']
        target_coupling = prior_targets.get('coupling_strength', TRUE_COUPLING_STRENGTH)
        target_speed_mean = prior_targets.get('speed_mean', VELOCITY_SCALE_BASE)
        target_inharm = prior_targets.get('inharm_b', None)
    else:
        target_damping = TRUE_DAMPING_RATES
        target_coupling = TRUE_COUPLING_STRENGTH
        target_speed_mean = VELOCITY_SCALE_BASE
        target_inharm = None

    speed_uniform_loss = speed_uniform_lambda * (speed_scalars.std() / (speed_scalars.mean() + 1e-8))
    if "speed" in skip:
        speed_mean_prior = zero
    else:
        speed_mean_prior = speed_mean_prior_lambda * (speed_scalars.mean() - target_speed_mean).pow(2)
    if "coupling" in skip:
        coupling_prior_loss = zero
    else:
        coupling_prior_loss = coupling_prior_lambda * (coupling_strength - target_coupling).pow(2)
    if "damping" in skip:
        damping_prior_loss = zero
    else:
        damping_prior_loss = damping_prior_lambda * F.mse_loss(damping_rates, target_damping)

    if "inharm" in skip:
        inharm_l2_loss = zero
    elif target_inharm is not None:
        inharm_l2_loss = inharm_l2_lambda * F.mse_loss(inharm_b, target_inharm)
    else:
        inharm_l2_loss = inharm_l2_lambda * inharm_b.pow(2).mean()

    inharm_ceiling_loss = inharm_ceiling_lambda * F.relu(inharm_b - inharm_ceiling_threshold).pow(2).mean()
    return (
        speed_uniform_loss
        + speed_mean_prior
        + coupling_prior_loss
        + damping_prior_loss
        + inharm_l2_loss
        + inharm_ceiling_loss
    )


def multi_resolution_stft_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    fft_sizes: list | None = None,
    hop_ratio: float = REAL_AUDIO_STFT_HOP_RATIO,
) -> torch.Tensor:
    """L1 magnitude loss across multiple STFT resolutions."""
    if fft_sizes is None:
        fft_sizes = REAL_AUDIO_STFT_FFT_SIZES

    y_pred = y_pred.float().flatten()
    y_true = y_true.float().flatten()
    min_len = min(y_pred.shape[0], y_true.shape[0])
    y_pred = y_pred[:min_len]
    y_true = y_true[:min_len]

    total = torch.tensor(0.0, device=y_pred.device)
    for n_fft in fft_sizes:
        hop = max(1, int(n_fft * hop_ratio))
        window = torch.hann_window(n_fft, device=y_pred.device)
        spec_pred = torch.stft(y_pred, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
        spec_true = torch.stft(y_true, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
        total = total + F.l1_loss(spec_pred.abs(), spec_true.abs())

    return total / len(fft_sizes)


def total_loss(
    preds,
    data_points,
    damping_rates,
    coupling_strength,
    inharm_b,
    speed_scalars,
    prior_targets: dict | None = None,
    synth_waveform: torch.Tensor | None = None,
    target_waveform: torch.Tensor | None = None,
    stft_weight: float = 0.0,
    fr_spectral_weight: float = 0.0,
    pred_spectrum: torch.Tensor | None = None,
    target_spectrum: torch.Tensor | None = None,
    mode_amps: torch.Tensor | None = None,
    target_mode_amps: torch.Tensor | None = None,
    fr_mode_weight: float = 0.0,
    log_base_rate: torch.Tensor | None = None,
    log_slope: torch.Tensor | None = None,
    coupling_skew: torch.Tensor | None = None,
    target_coupling_skew: torch.Tensor | None = None,
    fr_invariant_weight_override: float | None = None,
    fr_invariant_damping_override: float | None = None,
    fr_invariant_coupling_override: float | None = None,
    fr_invariant_speed_override: float | None = None,
    fr_invariant_inharm_override: float | None = None,
    fr_invariant_modal_override: float | None = None,
    fr_replace_mse_priors_override: bool | None = None,
    target_speed_scalars: torch.Tensor | None = None,
    target_inharm_b: torch.Tensor | None = None,
):
    inv_weight = fr_invariant_weight if fr_invariant_weight_override is None else fr_invariant_weight_override
    inv_damping = fr_invariant_damping if fr_invariant_damping_override is None else fr_invariant_damping_override
    inv_coupling = fr_invariant_coupling if fr_invariant_coupling_override is None else fr_invariant_coupling_override
    inv_speed = fr_invariant_speed if fr_invariant_speed_override is None else fr_invariant_speed_override
    inv_inharm = fr_invariant_inharm if fr_invariant_inharm_override is None else fr_invariant_inharm_override
    inv_modal = fr_invariant_modal if fr_invariant_modal_override is None else fr_invariant_modal_override
    replace_mse = (
        fr_replace_mse_priors if fr_replace_mse_priors_override is None else fr_replace_mse_priors_override
    )

    skip_mse = _mse_prior_skip_terms(
        inv_weight,
        replace_mse,
        inv_damping=inv_damping,
        inv_coupling=inv_coupling,
        inv_speed=inv_speed,
        inv_inharm=inv_inharm,
        log_base_rate=log_base_rate,
        log_slope=log_slope,
        coupling_skew=coupling_skew,
    )

    loss = geo_loss(preds, data_points) + prior_loss(
        damping_rates, coupling_strength, inharm_b, speed_scalars,
        prior_targets=prior_targets,
        skip_mse_terms=skip_mse,
    )

    if inv_weight > 0.0:
        from .invariant_losses import (
            damping_invariant_loss,
            coupling_invariant_loss,
            speed_profile_invariant_loss,
            inharm_profile_invariant_loss,
            modal_amp_pair_invariant_loss,
        )

        if log_base_rate is not None and log_slope is not None:
            if prior_targets is not None:
                target_damping = prior_targets["damping_rates"]
            else:
                target_damping = TRUE_DAMPING_RATES
            loss = loss + inv_weight * inv_damping * damping_invariant_loss(
                log_base_rate, log_slope, target_damping,
            )

        if inv_coupling > 0.0 and coupling_skew is not None:
            if prior_targets is not None:
                target_strength = prior_targets.get("coupling_strength", TRUE_COUPLING_STRENGTH)
                ref_skew = target_coupling_skew
                if ref_skew is None:
                    ref_skew = prior_targets.get("coupling_skew")
            else:
                target_strength = TRUE_COUPLING_STRENGTH
                ref_skew = target_coupling_skew
            loss = loss + inv_weight * inv_coupling * coupling_invariant_loss(
                coupling_skew,
                coupling_strength,
                ref_skew,
                target_strength,
                strength_weight=fr_invariant_coupling_strength_weight,
            )

        if inv_speed > 0.0:
            ref_speed = target_speed_scalars
            if ref_speed is None and prior_targets is not None:
                ref_speed = prior_targets.get("speed_scalars")
            loss = loss + inv_weight * inv_speed * speed_profile_invariant_loss(speed_scalars, ref_speed)

        if inv_inharm > 0.0:
            ref_inharm = target_inharm_b
            if ref_inharm is None:
                if prior_targets is not None and prior_targets.get("inharm_b") is not None:
                    ref_inharm = prior_targets["inharm_b"]
                else:
                    ref_inharm = TRUE_INHARM_B
            loss = loss + inv_weight * inv_inharm * inharm_profile_invariant_loss(inharm_b, ref_inharm)

        if inv_modal > 0.0 and mode_amps is not None:
            modal_obs = target_mode_amps
            if prior_targets is not None:
                if "partial_amps_temporal" in prior_targets:
                    modal_obs = prior_targets["partial_amps_temporal"]
                elif "partial_amps_mean" in prior_targets:
                    modal_obs = prior_targets["partial_amps_mean"]
            if modal_obs is not None:
                loss = loss + inv_weight * inv_modal * modal_amp_pair_invariant_loss(
                    mode_amps, modal_obs,
                )
    if stft_weight > 0.0 and synth_waveform is not None and target_waveform is not None:
        if synth_waveform.is_cuda or target_waveform.is_cuda:
            from .streaming_stft import multi_resolution_stft_loss_gpu
            loss = loss + stft_weight * multi_resolution_stft_loss_gpu(synth_waveform, target_waveform)
        else:
            loss = loss + stft_weight * multi_resolution_stft_loss(synth_waveform, target_waveform)
    if fr_spectral_weight > 0.0 and pred_spectrum is not None and target_spectrum is not None:
        from .fisher_rao_losses import stft_bin_fr_loss
        loss = loss + fr_spectral_weight * stft_bin_fr_loss(pred_spectrum, target_spectrum)
    if fr_mode_weight > 0.0 and mode_amps is not None:
        from .fisher_rao_losses import mode_amplitude_fr_loss
        loss = loss + fr_mode_weight * mode_amplitude_fr_loss(mode_amps, target_mode_amps)
    return loss