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
)
from .utils import stiefel_dist

speed_mean_prior_lambda = 1.0


def geo_loss(preds, data_points):
    with autocast('cuda', enabled=False):
        return stiefel_dist(preds.float(), data_points.float()).pow(2).mean()


def prior_loss(
    damping_rates,
    coupling_strength,
    inharm_b,
    speed_scalars,
    prior_targets: dict | None = None,
):
    """Physics priors. When prior_targets is set, use estimated real-audio values."""
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
    speed_mean_prior = speed_mean_prior_lambda * (speed_scalars.mean() - target_speed_mean).pow(2)
    coupling_prior_loss = coupling_prior_lambda * (coupling_strength - target_coupling).pow(2)
    damping_prior_loss = damping_prior_lambda * F.mse_loss(damping_rates, target_damping)

    if target_inharm is not None:
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
):
    loss = geo_loss(preds, data_points) + prior_loss(
        damping_rates, coupling_strength, inharm_b, speed_scalars, prior_targets=prior_targets,
    )
    if stft_weight > 0.0 and synth_waveform is not None and target_waveform is not None:
        if synth_waveform.is_cuda or target_waveform.is_cuda:
            from .streaming_stft import multi_resolution_stft_loss_gpu
            loss = loss + stft_weight * multi_resolution_stft_loss_gpu(synth_waveform, target_waveform)
        else:
            loss = loss + stft_weight * multi_resolution_stft_loss(synth_waveform, target_waveform)
    return loss