#!/usr/bin/env python3
"""
run_real_audio.py
Real audio extension for the physics_audio / mlpa Stiefel manifold model.

Pipeline:
  1. Load audio (file, folder batch, or streaming chunks)
  2. Track partials (piptrack + continuity linking)
  3. Estimate physical parameters (damping, B, f0)
  4. Build manifold observations + initialize model
  5. Optimize (geometric + physics priors + multi-res STFT + punctuated jumps)
  6. Synthesize (with coupling) + compare spectrograms

Usage:
    python run_real_audio.py --audio path/to/guitar_note.wav --duration 3.0
    python run_real_audio.py --audio_dir path/to/notes/ --max_steps 5000
    python run_real_audio.py --audio note.wav --streaming
"""

import argparse
import csv
import json
import librosa
import numpy as np
import soundfile as sf
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from training_evaluation.config import (
    K_MODES, DIM, REAL_AUDIO_MAX_STEPS, REAL_AUDIO_STFT_WEIGHT, REAL_AUDIO_SR,
)
from training_evaluation.model import StiefelDampedCoupledInharmGR
from training_evaluation.audio_utils import (
    extract_partials_piptrack,
    extract_partials_streaming,
    StreamingPartialTracker,
    estimate_physical_params,
    partials_to_manifold_data,
    initialize_model_from_physics,
    build_prior_targets,
    modal_synthesis,
)
from training_evaluation.trainer import run_single_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AUDIO_EXTENSIONS = {'.wav', '.flac', '.ogg', '.mp3', '.aiff', '.aif'}


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


def save_partial_plot(partial_freqs, partial_amps, times, output_path: str, title_suffix: str = ""):
    """Save tracked partial frequency and amplitude trajectories."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for k in range(partial_freqs.shape[0]):
        axes[0].plot(times, partial_freqs[k], label=f"mode {k+1}")
        axes[1].plot(times, partial_amps[k], alpha=0.8)
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].legend(ncol=4, fontsize=8)
    axes[0].set_title(f"Tracked partial frequencies{title_suffix}")
    axes[1].set_ylabel("Magnitude")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Tracked partial amplitudes")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def discover_audio_files(path: Path) -> list[Path]:
    """Recursively find audio files in a directory."""
    if path.is_file():
        return [path]
    files = sorted(
        p for p in path.rglob('*')
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )
    return files


def process_single_note(
    audio_path: Path,
    output_dir: Path,
    duration: float,
    max_steps: int,
    stft_weight: float,
    seed: int,
    streaming: bool = False,
) -> dict:
    """Run full pipeline on one audio file. Returns result summary dict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    note_name = audio_path.stem

    print(f"\n{'='*60}")
    print(f"Processing: {audio_path}")
    print(f"Output dir: {output_dir}")

    y, sr = load_audio(str(audio_path), duration=duration)
    audio_duration = len(y) / sr
    print(f"Audio loaded: {audio_duration:.2f}s @ {sr} Hz")

    tracker_label = "streaming" if streaming else "piptrack + continuity"
    print(f"Extracting partials ({tracker_label})...")
    if streaming:
        partial_freqs, partial_amps, times, f0 = extract_partials_streaming(y, sr, n_partials=K_MODES)
    else:
        partial_freqs, partial_amps, times, f0 = extract_partials_piptrack(y, sr, n_partials=K_MODES)

    save_partial_plot(
        partial_freqs, partial_amps, times,
        str(output_dir / "partials_tracked.png"),
        title_suffix=f" ({tracker_label})",
    )

    damping_rates, f0_est, b_est = estimate_physical_params(partial_freqs, partial_amps, times)
    print(f"Estimated f0 ≈ {f0_est:.1f} Hz, B ≈ {b_est:.6f}")
    print(f"Estimated damping rates: {np.round(damping_rates, 4)}")

    print("Building manifold observations...")
    data_points, model_times = partials_to_manifold_data(
        partial_freqs, partial_amps, times, f0_est,
    )

    print("Initializing model from physical estimates...")
    model = StiefelDampedCoupledInharmGR(DIM, K_MODES, data_points[0])
    model = initialize_model_from_physics(model, damping_rates, f0_est, b_est, data_points)

    prior_targets = build_prior_targets(damping_rates, f0_est, b_est)
    target_waveform = torch.tensor(y, device=device, dtype=torch.float32)

    print(f"Running optimization ({max_steps} steps, STFT={stft_weight}, jumps enabled)...")
    result = run_single_seed(
        seed=seed,
        real_audio_data=data_points,
        real_audio_times=model_times,
        target_waveform=target_waveform,
        audio_sr=sr,
        audio_duration=audio_duration,
        prior_targets=prior_targets,
        max_steps=max_steps,
        stft_weight=stft_weight,
        preinitialized_model=model,
    )

    learned_freqs = result['full_freq'].cpu().numpy()
    learned_damps = result['damping_rates'].cpu().numpy()
    coupling_strength = result['coupling_strength']
    coupling_skew = result['coupling_skew'].cpu().numpy()

    print(f"  Recon MSE:     {result['total_recon_mse_pred']:.6f}")
    print(f"  Damping RMSE:  {result['damping_rmse']:.6f}")
    print(f"  Damping corr:  {result['damping_corr']:.4f}")
    print(f"  Coupling:      {coupling_strength:.4f}")
    print(f"  Jumps:         {result['jumps']}")
    print(f"  Wall time:     {result['wall_time']:.1f}s")

    print("Synthesizing with coupled modal model...")
    y_synth = modal_synthesis(
        learned_freqs, learned_damps, sr, audio_duration,
        coupling_strength=coupling_strength,
        coupling_skew=coupling_skew,
    )
    sf.write(str(output_dir / "synthesized.wav"), y_synth, sr)
    save_spectrogram_comparison(
        y, y_synth, sr,
        str(output_dir / "spectrogram_comparison.png"),
        title=f"{note_name}: Real vs Synthesized",
    )

    summary = {
        'file': str(audio_path),
        'name': note_name,
        'duration_s': audio_duration,
        'f0_est_hz': f0_est,
        'b_est': b_est,
        'recon_mse': result['total_recon_mse_pred'],
        'damping_rmse': result['damping_rmse'],
        'damping_corr': result['damping_corr'],
        'coupling_strength': coupling_strength,
        'jumps': result['jumps'],
        'wall_time_s': result['wall_time'],
        'learned_freqs': learned_freqs.tolist(),
        'learned_damps': learned_damps.tolist(),
        'tracker': tracker_label,
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def run_batch(
    audio_dir: Path,
    output_dir: Path,
    duration: float,
    max_steps: int,
    stft_weight: float,
    seed: int,
    streaming: bool,
) -> list[dict]:
    """Process all audio files in a folder."""
    files = discover_audio_files(audio_dir)
    if not files:
        raise FileNotFoundError(f"No audio files found in {audio_dir}")

    print(f"Batch mode: {len(files)} file(s) in {audio_dir}")
    results = []
    for i, audio_path in enumerate(files):
        note_out = output_dir / audio_path.stem
        summary = process_single_note(
            audio_path, note_out, duration, max_steps, stft_weight,
            seed=seed + i, streaming=streaming,
        )
        results.append(summary)

    # Write batch summary CSV
    csv_path = output_dir / "batch_summary.csv"
    if results:
        fieldnames = [
            'name', 'file', 'duration_s', 'f0_est_hz', 'b_est',
            'recon_mse', 'damping_rmse', 'damping_corr', 'coupling_strength',
            'jumps', 'wall_time_s', 'tracker',
        ]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({k: r[k] for k in fieldnames})

    print(f"\n{'='*60}")
    print(f"Batch complete: {len(results)} notes processed")
    print(f"Summary CSV: {csv_path}")
    return results


def demo_streaming_live(audio_path: str, duration: float = 3.0):
    """Demonstrate streaming tracker chunk-by-chunk (prints live partial state)."""
    y, sr = load_audio(audio_path, duration=duration)
    tracker = StreamingPartialTracker(sr=sr, n_partials=K_MODES)
    chunk_size = tracker.chunk_size
    print(f"Streaming demo: {len(y)/sr:.2f}s @ {sr}Hz, chunk={chunk_size} samples ({chunk_size/sr*1000:.0f}ms)")

    for start in range(0, len(y), chunk_size):
        state = tracker.process_chunk(y[start:start + chunk_size])
        if state is not None:
            f0_live = state['freqs'][0]
            print(f"  t={state['time']:.3f}s  f0={f0_live:.1f}Hz  frame={state['frame_idx']}")

    freqs, amps, times, _ = tracker.finalize()
    print(f"Streaming finalize: {times.shape[0]} frames, f0 range [{freqs[0].min():.1f}, {freqs[0].max():.1f}] Hz")
    return freqs, amps, times


def main():
    parser = argparse.ArgumentParser(description="Real-audio fitting pipeline")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--audio", type=str, help="Path to a single audio file")
    src.add_argument("--audio_dir", type=str, help="Folder of audio files for batch processing")
    parser.add_argument("--duration", type=float, default=4.0, help="Max duration per file (seconds)")
    parser.add_argument("--output_dir", type=str, default="real_audio_results")
    parser.add_argument("--max_steps", type=int, default=REAL_AUDIO_MAX_STEPS)
    parser.add_argument("--stft_weight", type=float, default=REAL_AUDIO_STFT_WEIGHT)
    parser.add_argument("--no_stft", action="store_true", help="Disable STFT audio loss")
    parser.add_argument("--streaming", action="store_true", help="Use streaming partial tracker")
    parser.add_argument("--stream_demo", action="store_true", help="Run streaming demo only (no fitting)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    stft_weight = 0.0 if args.no_stft else args.stft_weight
    output_dir = Path(args.output_dir)

    if args.stream_demo:
        if not args.audio:
            parser.error("--stream_demo requires --audio")
        demo_streaming_live(args.audio, args.duration)
        return

    if args.audio_dir:
        run_batch(
            Path(args.audio_dir), output_dir,
            args.duration, args.max_steps, stft_weight, args.seed, args.streaming,
        )
    else:
        process_single_note(
            Path(args.audio), output_dir,
            args.duration, args.max_steps, stft_weight, args.seed, args.streaming,
        )


if __name__ == "__main__":
    main()