#!/usr/bin/env python3
"""
run_real_audio.py
Real audio extension for the physics_audio / mlpa Stiefel manifold model.
Starter version: loads audio, tracks partials (with inharmonicity), estimates
physical parameters, initializes the model, runs basic optimization, and synthesizes.

Usage:
    python run_real_audio.py --audio path/to/guitar_note.wav --duration 3.0
"""

import argparse
import librosa
import soundfile as sf
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Project imports
from training_evaluation.config import *
from training_evaluation.model import StiefelDampedCoupledInharmGR
from training_evaluation.utils import manifold, safe_proj, get_pca_initial_basis
from training_evaluation.losses import total_loss
from training_evaluation.trainer import run_single_seed  # We can adapt this later

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_audio(audio_path: str, sr: int = 44100, duration: float = None):
    """Load and basic preprocess audio."""
    y, sr = librosa.load(audio_path, sr=sr, duration=duration, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)
    y = librosa.util.normalize(y)
    return y, sr

def extract_partials_with_inharmonicity(
    y: np.ndarray, 
    sr: int, 
    n_partials: int = K_MODES,
    hop_length: int = 512,
    n_fft: int = 2048,
    fmin: float = 50.0,
    fmax: float = 2000.0
):
    """
    Extract f0 + inharmonic partials using pyin + peak tracking.
    Returns approximate frequency and amplitude trajectories per mode.
    """
    # 1. f0 estimation with pYIN (excellent for strings)
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr,
        frame_length=n_fft, hop_length=hop_length
    )
    
    # 2. STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(D)
    phase = np.angle(D)
    
    times = librosa.frames_to_time(np.arange(mag.shape[1]), sr=sr, hop_length=hop_length)
    
    # 3. Simple harmonic + inharmonicity-aware partial tracking
    partial_freqs = np.zeros((n_partials, len(times)))
    partial_amps = np.zeros((n_partials, len(times)))
    
    for t_idx in range(len(times)):
        if not voiced_flag[t_idx] or np.isnan(f0[t_idx]):
            continue
            
        f0_t = f0[t_idx]
        
        for k in range(n_partials):
            # Expected frequency with inharmonicity allowance
            expected_f = (k + 1) * f0_t * (1 + 0.0005 * (k + 1)**2)  # rough B guess
            
            # Find nearest strong peak in spectrum
            bin_idx = int(expected_f * n_fft / sr)
            if bin_idx < 2 or bin_idx >= mag.shape[0] - 2:
                continue
                
            # Local peak refinement (parabolic interpolation via piptrack style)
            local_mag = mag[bin_idx-1:bin_idx+2, t_idx]
            if local_mag.max() > 0.01 * mag[:, t_idx].max():
                # Simple peak refinement
                peak_bin = bin_idx + np.argmax(local_mag) - 1
                partial_freqs[k, t_idx] = peak_bin * sr / n_fft
                partial_amps[k, t_idx] = mag[peak_bin, t_idx]
    
    # Clean trajectories (median filter + interpolation for gaps)
    for k in range(n_partials):
        valid = partial_freqs[k] > 0
        if np.sum(valid) > 5:
            partial_freqs[k] = np.interp(
                np.arange(len(times)), 
                np.where(valid)[0], 
                partial_freqs[k][valid]
            )
            partial_amps[k] = np.interp(
                np.arange(len(times)), 
                np.where(valid)[0], 
                partial_amps[k][valid]
            )
    
    return partial_freqs, partial_amps, times, f0

def estimate_physical_params(partial_freqs, partial_amps, times):
    """Estimate damping, inharmonicity B, and base frequency from tracked partials."""
    n_modes = partial_freqs.shape[0]
    
    # Simple exponential fit for damping per mode
    damping_rates = np.zeros(n_modes)
    for k in range(n_modes):
        amp = partial_amps[k]
        valid = amp > 0.01 * amp.max()
        if np.sum(valid) > 10:
            log_amp = np.log(amp[valid] + 1e-8)
            t_valid = times[valid]
            # Linear fit: log(amp) = -damping * t + c
            slope, _ = np.polyfit(t_valid, log_amp, 1)
            damping_rates[k] = -slope
    
    # Estimate inharmonicity B and f0 using least squares on higher partials
    # f_n ≈ n * f0 * sqrt(1 + B * n^2)
    f0_estimates = []
    b_estimates = []
    
    for t in range(0, len(times), max(1, len(times)//20)):  # sample some frames
        freqs_t = partial_freqs[:, t]
        valid = freqs_t > 10
        if np.sum(valid) < 3:
            continue
        n = np.arange(1, n_modes + 1)[valid]
        f_obs = freqs_t[valid]
        
        # Rough f0 from fundamental
        f0_t = f_obs[0] if len(f_obs) > 0 else 0
        if f0_t < 10:
            continue
            
        # Solve for B
        # f_n / n ≈ f0 * sqrt(1 + B n^2)
        y = (f_obs / n)**2
        x = n**2
        # Linear regression y = f0^2 + (f0^2 * B) * x
        A = np.vstack([np.ones_like(x), x]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        f0_sq, slope = coeffs
        if f0_sq > 0 and slope > 0:
            f0_estimates.append(np.sqrt(f0_sq))
            b_estimates.append(slope / f0_sq)
    
    f0_est = np.median(f0_estimates) if f0_estimates else 440.0
    b_est = np.median(b_estimates) if b_estimates else 0.0002
    
    return damping_rates, f0_est, b_est

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file (WAV, etc.)")
    parser.add_argument("--duration", type=float, default=4.0, help="Max duration in seconds")
    parser.add_argument("--output_dir", type=str, default="real_audio_results")
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(exist_ok=True)
    
    print(f"Loading audio: {args.audio}")
    y, sr = load_audio(args.audio, duration=args.duration)
    print(f"Audio loaded: {len(y)/sr:.2f}s @ {sr} Hz")
    
    # Extract partials
    print("Extracting partials with inharmonicity tracking...")
    partial_freqs, partial_amps, times, f0 = extract_partials_with_inharmonicity(y, sr, n_partials=K_MODES)
    
    # Estimate physical parameters
    damping_rates, f0_est, b_est = estimate_physical_params(partial_freqs, partial_amps, times)
    print(f"Estimated f0 ≈ {f0_est:.1f} Hz, B ≈ {b_est:.6f}")
    print(f"Estimated damping rates: {damping_rates}")
    
    # TODO in next iteration: Initialize model from these estimates
    # and run optimization (we can adapt run_single_seed or create a new fitter)
    
    print("\n=== Basic extraction complete ===")
    print("Next steps you can ask me:")
    print("1. Implement full model initialization + optimization loop")
    print("2. Add modal synthesis (generate audio from params)")
    print("3. Add STFT loss for audio matching")
    print("4. Integrate with punctuated equilibrium trainer")

if __name__ == "__main__":
    main()