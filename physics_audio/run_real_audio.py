#!/usr/bin/env python3
"""
run_real_audio.py
Real audio extension for the physics_audio / mlpa Stiefel manifold model.

Pipeline:
  1. Load audio
  2. Track partials (piptrack + continuity linking)
  3. Estimate physical parameters (damping, B, f0)
  4. Build manifold observations + initialize model
  5. Optimize (geometric + physics priors + multi-res STFT loss)
  6. Synthesize + compare spectrograms

Usage:
    python run_real_audio.py --audio path/to/guitar_note.wav --duration 3.0
    python run_real_audio.py --audio note.wav --max_steps 5000 --no_stft
"""

import argparse
import librosa
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from training_evaluation.config import (
    K_MODES, REAL_AUDIO_MAX_STEPS, REAL_AUDIO_STFT_WEIGHT, REAL_AUDIO_SR,
)
from training_evaluation.model import StiefelDampedCoupledInharmGR
from training_evaluation.audio_utils import (
    extract_partials_piptrack,
    estimate_physical_params,
    partials_to_manifold_data,
    initialize_model_from_physics,
    build_prior_targets,
    modal_synthesis,
)
from training_evaluation.trainer import run_single_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_audio(audio_path: str, sr: int = REAL_AUDIO_SR, duration: float = None):
    """Load and basic preprocess audio."""
    y, sr = librosa.load(audio_path, sr=sr, duration=duration, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)
    y = librosa.util.normalize(y)
    return y, sr


def save_spectrogram_comparison(
    y_target: np.ndarray,
    y_synth: np.ndarray,
    sr: int,
    output_path: str,
    title: str = "Real vs Synthesized",
):
    """Save side-by-side log-magnitude spectrograms."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, y, label in zip(axes[:2], [y_target, y_synth], ["Target", "Synthesized"]):
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        ax.set_title(label)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')

    min_len = min(len(y_target), len(y_synth))
    residual = y_target[:min_len] - y_synth[:min_len]
    S_res = librosa.feature.melspectrogram(y=residual, sr=sr, n_mels=128)
    S_res_db = librosa.power_to_db(S_res, ref=np.max)
    img = librosa.display.specshow(S_res_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[2])
    axes[2].set_title("Residual")
    fig.colorbar(img, ax=axes[2], format='%+2.0f dB')

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_partial_plot(partial_freqs, partial_amps, times, output_path: str):
    """Save tracked partial frequency and amplitude trajectories."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for k in range(partial_freqs.shape[0]):
        axes[0].plot(times, partial_freqs[k], label=f"mode {k+1}")
        axes[1].plot(times, partial_amps[k], alpha=0.8)
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].legend(ncol=4, fontsize=8)
    axes[0].set_title("Tracked partial frequencies (piptrack + continuity)")
    axes[1].set_ylabel("Magnitude")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Tracked partial amplitudes")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file (WAV, etc.)")
    parser.add_argument("--duration", type=float, default=4.0, help="Max duration in seconds")
    parser.add_argument("--output_dir", type=str, default="real_audio_results")
    parser.add_argument("--max_steps", type=int, default=REAL_AUDIO_MAX_STEPS)
    parser.add_argument("--stft_weight", type=float, default=REAL_AUDIO_STFT_WEIGHT)
    parser.add_argument("--no_stft", action="store_true", help="Disable STFT audio loss")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    stft_weight = 0.0 if args.no_stft else args.stft_weight

    print(f"Loading audio: {args.audio}")
    y, sr = load_audio(args.audio, duration=args.duration)
    audio_duration = len(y) / sr
    print(f"Audio loaded: {audio_duration:.2f}s @ {sr} Hz")

    print("Extracting partials (piptrack + continuity linking)...")
    partial_freqs, partial_amps, times, f0 = extract_partials_piptrack(
        y, sr, n_partials=K_MODES,
    )
    save_partial_plot(
        partial_freqs, partial_amps, times,
        str(output_dir / "partials_tracked.png"),
    )

    damping_rates, f0_est, b_est = estimate_physical_params(partial_freqs, partial_amps, times)
    print(f"Estimated f0 ≈ {f0_est:.1f} Hz, B ≈ {b_est:.6f}")
    print(f"Estimated damping rates: {np.round(damping_rates, 4)}")

    print("Building manifold observations from partial trajectories...")
    data_points, model_times = partials_to_manifold_data(
        partial_freqs, partial_amps, times, f0_est,
    )

    print("Initializing model from physical estimates...")
    from training_evaluation.config import DIM
    model = StiefelDampedCoupledInharmGR(DIM, K_MODES, data_points[0])
    model = initialize_model_from_physics(model, damping_rates, f0_est, b_est, data_points)

    prior_targets = build_prior_targets(damping_rates, f0_est, b_est)
    target_waveform = torch.tensor(y, device=device, dtype=torch.float32)

    print(f"Running optimization ({args.max_steps} steps, STFT weight={stft_weight})...")
    result = run_single_seed(
        seed=args.seed,
        real_audio_data=data_points,
        real_audio_times=model_times,
        target_waveform=target_waveform,
        audio_sr=sr,
        audio_duration=audio_duration,
        prior_targets=prior_targets,
        max_steps=args.max_steps,
        stft_weight=stft_weight,
        preinitialized_model=model,
    )

    print(f"\n=== Optimization complete ===")
    print(f"  Recon MSE:     {result['total_recon_mse_pred']:.6f}")
    print(f"  Damping RMSE:  {result['damping_rmse']:.6f}")
    print(f"  Damping corr:  {result['damping_corr']:.4f}")
    print(f"  Jumps:         {result['jumps']}")
    print(f"  Wall time:     {result['wall_time']:.1f}s")

    learned_freqs = result['full_freq'].cpu().numpy()
    learned_damps = result['damping_rates'].cpu().numpy()
    print(f"  Learned freqs: {np.round(learned_freqs, 1)}")
    print(f"  Learned damps: {np.round(learned_damps, 4)}")

    print("Synthesizing audio from learned parameters...")
    y_synth = modal_synthesis(learned_freqs, learned_damps, sr, audio_duration)
    synth_path = output_dir / "synthesized.wav"
    import soundfile as sf
    sf.write(str(synth_path), y_synth, sr)
    print(f"  Saved synthesis: {synth_path}")

    spec_path = output_dir / "spectrogram_comparison.png"
    save_spectrogram_comparison(y, y_synth, sr, str(spec_path))
    print(f"  Saved spectrogram comparison: {spec_path}")

    print("\nDone. Next iterations:")
    print("  - Differentiable coupling in modal synthesis")
    print("  - Batch processing of multiple notes")
    print("  - Real-time capable fitting")


if __name__ == "__main__":
    main()