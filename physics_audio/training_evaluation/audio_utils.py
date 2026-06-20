"""
Audio-domain utilities for the real-audio extension:
  - piptrack + continuity partial tracking
  - Physical parameter estimation
  - Manifold observation construction from partial trajectories
  - Model initialization from estimated physics
  - Modal synthesis (NumPy + differentiable PyTorch)
"""

from __future__ import annotations

import numpy as np
import torch
import librosa
from scipy.optimize import linear_sum_assignment
from typing import Optional, Tuple

from .config import (
    DIM, K_MODES, N_POINTS, device, IDEAL_HARMONICS, VELOCITY_SCALE_BASE,
    STREAM_HOP_LENGTH, STREAM_N_FFT, STREAM_CHUNK_SECONDS,
)
from .model import StiefelDampedCoupledInharmGR
from .utils import safe_proj, get_pca_initial_basis, manifold


# ---------------------------------------------------------------------------
# Partial tracking (piptrack + continuity linking)
# ---------------------------------------------------------------------------

def _link_frame_partials(
    peak_freqs: np.ndarray,
    peak_mags: np.ndarray,
    f0_t: float,
    n_partials: int,
    prev_freqs: Optional[np.ndarray],
    b_guess: float = 0.0005,
    continuity_weight: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign peaks to harmonic partial indices with continuity bias."""
    if len(peak_freqs) == 0 or f0_t <= 0:
        if prev_freqs is not None:
            return prev_freqs.copy(), np.zeros(n_partials)
        return np.zeros(n_partials), np.zeros(n_partials)

    expected = np.array([
        (k + 1) * f0_t * np.sqrt(1.0 + b_guess * (k + 1) ** 2)
        for k in range(n_partials)
    ])

    n_peaks = len(peak_freqs)
    cost = np.zeros((n_partials, n_peaks))
    for k in range(n_partials):
        for p in range(n_peaks):
            harmonic_cost = abs(peak_freqs[p] - expected[k]) / (expected[k] + 1e-8)
            if prev_freqs is not None and prev_freqs[k] > 0:
                continuity_cost = abs(peak_freqs[p] - prev_freqs[k]) / (prev_freqs[k] + 1e-8)
            else:
                continuity_cost = 0.0
            cost[k, p] = harmonic_cost + continuity_weight * continuity_cost

    # Pad cost matrix if fewer peaks than partials
    if n_peaks < n_partials:
        padded = np.full((n_partials, n_partials), 1e6)
        padded[:, :n_peaks] = cost
        cost = padded
        n_peaks = n_partials

    row_ind, col_ind = linear_sum_assignment(cost)

    freqs_out = np.zeros(n_partials)
    amps_out = np.zeros(n_partials)
    for k, p in zip(row_ind, col_ind):
        if p < len(peak_freqs):
            freqs_out[k] = peak_freqs[p]
            amps_out[k] = peak_mags[p]
        elif prev_freqs is not None:
            freqs_out[k] = prev_freqs[k]

    return freqs_out, amps_out


def _clean_trajectories(partial_freqs: np.ndarray, partial_amps: np.ndarray) -> None:
    """Median-filter gaps and interpolate missing frames in-place."""
    from scipy.ndimage import median_filter

    n_partials, n_frames = partial_freqs.shape
    for k in range(n_partials):
        valid = partial_freqs[k] > 0
        if np.sum(valid) < 5:
            continue
        idx = np.where(valid)[0]
        partial_freqs[k] = np.interp(np.arange(n_frames), idx, partial_freqs[k][valid])
        partial_amps[k] = np.interp(np.arange(n_frames), idx, partial_amps[k][valid])
        partial_freqs[k] = median_filter(partial_freqs[k], size=5, mode='nearest')
        partial_amps[k] = median_filter(partial_amps[k], size=5, mode='nearest')


def extract_partials_piptrack(
    y: np.ndarray,
    sr: int,
    n_partials: int = K_MODES,
    hop_length: int = 512,
    n_fft: int = 2048,
    fmin: float = 50.0,
    fmax: float = 2000.0,
    b_guess: float = 0.0005,
    mag_threshold_ratio: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract f0 + inharmonic partials using pYIN + librosa.piptrack with
    per-frame Hungarian assignment and temporal continuity linking.
    """
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr,
        frame_length=n_fft, hop_length=hop_length,
    )

    pitches, magnitudes = librosa.piptrack(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        fmin=fmin, fmax=fmax,
    )

    n_frames = pitches.shape[1]
    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)

    partial_freqs = np.zeros((n_partials, n_frames))
    partial_amps = np.zeros((n_partials, n_frames))
    prev_freqs = None

    for t_idx in range(n_frames):
        frame_pitches = pitches[:, t_idx]
        frame_mags = magnitudes[:, t_idx]
        frame_max = frame_mags.max() + 1e-12
        valid = (frame_mags > mag_threshold_ratio * frame_max) & (frame_pitches > fmin)

        if not np.any(valid):
            if prev_freqs is not None:
                partial_freqs[:, t_idx] = prev_freqs
            continue

        peak_bins = np.where(valid)[0]
        peak_freqs = frame_pitches[peak_bins]
        peak_mags = frame_mags[peak_bins]
        order = np.argsort(peak_freqs)
        peak_freqs = peak_freqs[order]
        peak_mags = peak_mags[order]

        if voiced_flag[t_idx] and not np.isnan(f0[t_idx]):
            f0_t = float(f0[t_idx])
        elif prev_freqs is not None and prev_freqs[0] > 0:
            f0_t = float(prev_freqs[0])
        else:
            f0_t = float(peak_freqs[0])

        freqs_t, amps_t = _link_frame_partials(
            peak_freqs, peak_mags, f0_t, n_partials, prev_freqs, b_guess=b_guess,
        )
        partial_freqs[:, t_idx] = freqs_t
        partial_amps[:, t_idx] = amps_t
        if freqs_t[0] > 0:
            prev_freqs = freqs_t.copy()

    _clean_trajectories(partial_freqs, partial_amps)
    return partial_freqs, partial_amps, times, f0


def estimate_physical_params(
    partial_freqs: np.ndarray,
    partial_amps: np.ndarray,
    times: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """Estimate per-mode damping, f0, and inharmonicity B from tracked partials."""
    n_modes = partial_freqs.shape[0]

    damping_rates = np.zeros(n_modes)
    for k in range(n_modes):
        amp = partial_amps[k]
        valid = amp > 0.01 * (amp.max() + 1e-12)
        if np.sum(valid) > 10:
            log_amp = np.log(amp[valid] + 1e-8)
            t_valid = times[valid]
            slope, _ = np.polyfit(t_valid, log_amp, 1)
            damping_rates[k] = max(-slope, 1e-4)

    f0_estimates, b_estimates = [], []
    step = max(1, len(times) // 20)
    for t in range(0, len(times), step):
        freqs_t = partial_freqs[:, t]
        valid = freqs_t > 10
        if np.sum(valid) < 3:
            continue
        n = np.arange(1, n_modes + 1)[valid]
        f_obs = freqs_t[valid]
        y = (f_obs / n) ** 2
        x = n ** 2
        A = np.vstack([np.ones_like(x), x]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        f0_sq, slope = coeffs
        if f0_sq > 0 and slope >= 0:
            f0_estimates.append(np.sqrt(f0_sq))
            b_estimates.append(slope / f0_sq)

    f0_est = float(np.median(f0_estimates)) if f0_estimates else 440.0
    b_est = float(np.median(b_estimates)) if b_estimates else 0.0002
    return damping_rates, f0_est, b_est


# ---------------------------------------------------------------------------
# Manifold observation construction
# ---------------------------------------------------------------------------

def _features_for_mode(
    freq: float,
    amp: float,
    mode_idx: int,
    f0_ref: float,
    dim: int,
) -> np.ndarray:
    feat = np.zeros(dim, dtype=np.float64)
    feat[0] = np.log(freq + 1e-8) / 8.0
    feat[1] = np.log(amp + 1e-8)
    feat[2] = (mode_idx + 1) / K_MODES
    feat[3] = freq / (f0_ref + 1e-8)
    for j in range(4, dim):
        feat[j] = np.sin((j - 3) * (mode_idx + 1) * np.pi / K_MODES) * feat[1]
    return feat


def partials_to_manifold_data(
    partial_freqs: np.ndarray,
    partial_amps: np.ndarray,
    audio_times: np.ndarray,
    f0_est: float,
    n_points: int = N_POINTS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert partial trajectories into Stiefel manifold observations (N_POINTS, DIM, K_MODES)
    and corresponding model time coordinates in [-2, 2].
    """
    k_modes = partial_freqs.shape[0]
    t_min, t_max = audio_times[0], audio_times[-1]
    model_times_np = np.linspace(t_min, t_max, n_points)
    model_times = torch.linspace(-2.0, 2.0, n_points, device=device)

    freqs_interp = np.zeros((k_modes, n_points))
    amps_interp = np.zeros((k_modes, n_points))
    for k in range(k_modes):
        freqs_interp[k] = np.interp(model_times_np, audio_times, partial_freqs[k])
        amps_interp[k] = np.interp(model_times_np, audio_times, partial_amps[k])

    data = np.zeros((n_points, DIM, k_modes), dtype=np.float32)
    for i in range(n_points):
        mat = np.zeros((DIM, k_modes), dtype=np.float64)
        for k in range(k_modes):
            mat[:, k] = _features_for_mode(
                freqs_interp[k, i], amps_interp[k, i], k, f0_est, DIM,
            )
        q, _ = np.linalg.qr(mat)
        data[i] = q[:, :k_modes].astype(np.float32)

    data_points = torch.tensor(data, device=device, dtype=torch.float32)
    return data_points, model_times


# ---------------------------------------------------------------------------
# Model initialization from physics estimates
# ---------------------------------------------------------------------------

def initialize_model_from_physics(
    model: StiefelDampedCoupledInharmGR,
    damping_rates: np.ndarray,
    f0_est: float,
    b_est: float,
    data_points: torch.Tensor,
) -> StiefelDampedCoupledInharmGR:
    """Wire estimated physical parameters into model learnable params."""
    harmonics = np.arange(1, K_MODES + 1, dtype=np.float64)
    A = np.vstack([np.ones(K_MODES), harmonics - 1.0]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, damping_rates, rcond=None)
    base_rate = max(float(coeffs[0]), 1e-4)
    slope = max(float(coeffs[1]), 1e-6)

    with torch.no_grad():
        model.log_base_rate.copy_(torch.log(torch.tensor(base_rate, device=device)))
        model.log_slope.copy_(torch.log(torch.tensor(slope, device=device)))
        model.raw_lin_b.copy_(torch.log(torch.tensor(max(b_est, 1e-8), device=device)))
        model.raw_quad_b.copy_(torch.tensor(-18.0, device=device))

        speed = f0_est / VELOCITY_SCALE_BASE
        model.log_speed.copy_(torch.log(torch.ones(K_MODES, device=device) * max(speed, 0.1)))

        initial_basis = get_pca_initial_basis(data_points, K_MODES)
        model.base.copy_(initial_basis)

        vel_dir = manifold.proju(model.base, torch.randn(DIM, K_MODES, device=device))
        vel_dir = vel_dir / (vel_dir.norm(dim=0, keepdim=True) + 1e-8)
        model.vel_dir_raw.copy_(vel_dir)

    return model


def build_prior_targets(
    damping_rates: np.ndarray,
    f0_est: float,
    b_est: float,
    coupling_strength: float = 0.30,
) -> dict:
    """Build prior target dict for real-audio loss (replaces synthetic TRUE_* constants)."""
    harmonics = torch.arange(1, K_MODES + 1, device=device, dtype=torch.float32)
    inharm_b = torch.full((K_MODES,), b_est, device=device) * harmonics
    return {
        'damping_rates': torch.tensor(damping_rates, device=device, dtype=torch.float32),
        'coupling_strength': coupling_strength,
        'inharm_b': inharm_b,
        'speed_mean': f0_est / VELOCITY_SCALE_BASE,
    }


# ---------------------------------------------------------------------------
# Coupling helpers
# ---------------------------------------------------------------------------

def extract_coupling_skew(model: StiefelDampedCoupledInharmGR) -> torch.Tensor:
    """Return skew-symmetric coupling matrix from model parameters."""
    raw = model.coupling_raw
    return raw.tril(diagonal=-1) - raw.triu(diagonal=1)


# ---------------------------------------------------------------------------
# Modal synthesis (with optional skew-symmetric coupling)
# ---------------------------------------------------------------------------

def _coupled_modal_sum(
    t: np.ndarray,
    freqs: np.ndarray,
    damping: np.ndarray,
    amps: np.ndarray,
    coupling_strength: float = 0.0,
    coupling_skew: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Synthesize coupled modal sum (NumPy)."""
    n_modes = len(freqs)
    t_row = t[np.newaxis, :]
    sin_modes = np.sin(2.0 * np.pi * freqs[:, np.newaxis] * t_row)
    cos_modes = np.cos(2.0 * np.pi * freqs[:, np.newaxis] * t_row)
    envelopes = np.exp(-damping[:, np.newaxis] * t_row)

    if coupling_strength > 0.0 and coupling_skew is not None:
        cross = coupling_skew @ (envelopes * sin_modes)
        phase_mod = coupling_strength * cross
        contrib = amps[:, np.newaxis] * envelopes * (
            sin_modes * np.cos(phase_mod) + cos_modes * np.sin(phase_mod)
        )
    else:
        contrib = amps[:, np.newaxis] * envelopes * sin_modes

    return contrib.sum(axis=0)


def modal_synthesis(
    freqs: np.ndarray,
    damping: np.ndarray,
    sr: int,
    duration: float,
    amps: Optional[np.ndarray] = None,
    phases: Optional[np.ndarray] = None,
    coupling_strength: float = 0.0,
    coupling_skew: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Generate audio waveform from modal frequencies, damping, and coupling."""
    n_samples = int(sr * duration)
    t = np.arange(n_samples, dtype=np.float64) / sr
    n_modes = len(freqs)
    if amps is None:
        amps = 1.0 / (np.arange(1, n_modes + 1))
    if phases is not None and np.any(phases != 0):
        y = np.zeros(n_samples, dtype=np.float64)
        for k in range(n_modes):
            y += amps[k] * np.sin(2.0 * np.pi * freqs[k] * t + phases[k]) * np.exp(-damping[k] * t)
    else:
        y = _coupled_modal_sum(t, freqs, damping, amps, coupling_strength, coupling_skew)

    peak = np.max(np.abs(y))
    if peak > 1e-8:
        y /= peak
    return y.astype(np.float32)


def modal_synthesis_torch(
    freqs: torch.Tensor,
    damping: torch.Tensor,
    duration: float,
    sr: int,
    amps: Optional[torch.Tensor] = None,
    coupling_strength: Optional[torch.Tensor] = None,
    coupling_skew: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Differentiable coupled modal synthesis for STFT loss during optimization."""
    n_samples = int(sr * duration)
    t = torch.linspace(0.0, duration, n_samples, device=freqs.device, dtype=freqs.dtype)
    n_modes = freqs.shape[0]
    if amps is None:
        amps = 1.0 / torch.arange(1, n_modes + 1, device=freqs.device, dtype=freqs.dtype)

    sin_modes = torch.sin(2.0 * np.pi * freqs.unsqueeze(1) * t.unsqueeze(0))
    cos_modes = torch.cos(2.0 * np.pi * freqs.unsqueeze(1) * t.unsqueeze(0))
    envelopes = torch.exp(-damping.unsqueeze(1) * t.unsqueeze(0))

    c_strength = float(coupling_strength.item()) if coupling_strength is not None else 0.0
    if c_strength > 0.0 and coupling_skew is not None:
        cross = coupling_skew @ (envelopes * sin_modes)
        phase_mod = coupling_strength * cross
        contrib = amps.unsqueeze(1) * envelopes * (
            sin_modes * torch.cos(phase_mod) + cos_modes * torch.sin(phase_mod)
        )
        y = contrib.sum(dim=0)
    else:
        y = (amps.unsqueeze(1) * envelopes * sin_modes).sum(dim=0)

    peak = y.abs().max()
    if peak > 1e-8:
        y = y / peak
    return y


# ---------------------------------------------------------------------------
# Streaming partial tracker (real-time / chunk-based)
# ---------------------------------------------------------------------------

class StreamingPartialTracker:
    """
    Incremental partial tracker for real-time or chunked audio.

    Feed audio in small chunks via process_chunk(); retrieve rolling
    trajectories via get_trajectories() or finalize() at end of stream.
    """

    AUDIO_EXTENSIONS = {'.wav', '.flac', '.ogg', '.mp3', '.aiff', '.aif'}

    def __init__(
        self,
        sr: int,
        n_partials: int = K_MODES,
        hop_length: int = STREAM_HOP_LENGTH,
        n_fft: int = STREAM_N_FFT,
        fmin: float = 50.0,
        fmax: float = 2000.0,
        b_guess: float = 0.0005,
        mag_threshold_ratio: float = 0.01,
    ):
        self.sr = sr
        self.n_partials = n_partials
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.fmin = fmin
        self.fmax = fmax
        self.b_guess = b_guess
        self.mag_threshold_ratio = mag_threshold_ratio

        self._buffer = np.zeros(0, dtype=np.float32)
        self._prev_freqs: Optional[np.ndarray] = None
        self._frame_freqs: list = []
        self._frame_amps: list = []
        self._frame_times: list = []
        self._total_samples = 0
        self._frames_processed = 0

    @property
    def chunk_size(self) -> int:
        return max(self.hop_length, int(self.sr * STREAM_CHUNK_SECONDS))

    def process_chunk(self, chunk: np.ndarray) -> Optional[dict]:
        """
        Process a new audio chunk. Returns latest partial state dict or None
        if not enough samples accumulated yet.
        """
        chunk = np.asarray(chunk, dtype=np.float32).flatten()
        self._buffer = np.concatenate([self._buffer, chunk])
        self._total_samples += len(chunk)

        latest = None
        while len(self._buffer) >= self.n_fft:
            frame = self._buffer[:self.n_fft]
            self._buffer = self._buffer[self.hop_length:]

            pitches, magnitudes = librosa.piptrack(
                y=frame, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length,
                fmin=self.fmin, fmax=self.fmax,
            )
            frame_pitches = pitches[:, 0]
            frame_mags = magnitudes[:, 0]
            frame_max = frame_mags.max() + 1e-12
            valid = (frame_mags > self.mag_threshold_ratio * frame_max) & (frame_pitches > self.fmin)

            t_sec = (self._frames_processed * self.hop_length) / self.sr

            if not np.any(valid):
                if self._prev_freqs is not None:
                    freqs_t = self._prev_freqs.copy()
                    amps_t = np.zeros(self.n_partials)
                else:
                    self._frames_processed += 1
                    continue
            else:
                peak_bins = np.where(valid)[0]
                peak_freqs = frame_pitches[peak_bins]
                peak_mags = frame_mags[peak_bins]
                order = np.argsort(peak_freqs)
                peak_freqs = peak_freqs[order]
                peak_mags = peak_mags[order]
                f0_t = float(peak_freqs[0])
                freqs_t, amps_t = _link_frame_partials(
                    peak_freqs, peak_mags, f0_t, self.n_partials,
                    self._prev_freqs, b_guess=self.b_guess,
                )

            self._frame_freqs.append(freqs_t.copy())
            self._frame_amps.append(amps_t.copy())
            self._frame_times.append(t_sec)
            if freqs_t[0] > 0:
                self._prev_freqs = freqs_t.copy()
            self._frames_processed += 1

            latest = {
                'time': t_sec,
                'freqs': freqs_t.copy(),
                'amps': amps_t.copy(),
                'frame_idx': self._frames_processed - 1,
            }

        return latest

    def get_trajectories(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (partial_freqs, partial_amps, times) accumulated so far."""
        if not self._frame_times:
            empty = np.zeros((self.n_partials, 0))
            return empty, empty.copy(), np.zeros(0)

        times = np.array(self._frame_times)
        freqs = np.column_stack(self._frame_freqs)
        amps = np.column_stack(self._frame_amps)
        _clean_trajectories(freqs, amps)
        return freqs, amps, times

    def finalize(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Flush buffer and return cleaned trajectories + placeholder f0 track."""
        partial_freqs, partial_amps, times = self.get_trajectories()
        f0_track = partial_freqs[0].copy() if partial_freqs.shape[1] > 0 else np.zeros(0)
        return partial_freqs, partial_amps, times, f0_track

    def track_file_streaming(
        self,
        y: np.ndarray,
        chunk_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Track an entire signal by feeding it through process_chunk in blocks."""
        chunk_size = chunk_size or self.chunk_size
        for start in range(0, len(y), chunk_size):
            self.process_chunk(y[start:start + chunk_size])
        return self.finalize()


def extract_partials_streaming(
    y: np.ndarray,
    sr: int,
    n_partials: int = K_MODES,
    hop_length: int = STREAM_HOP_LENGTH,
    n_fft: int = STREAM_N_FFT,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convenience wrapper: streaming tracker over a complete signal."""
    tracker = StreamingPartialTracker(
        sr=sr, n_partials=n_partials, hop_length=hop_length, n_fft=n_fft, **kwargs,
    )
    return tracker.track_file_streaming(y)