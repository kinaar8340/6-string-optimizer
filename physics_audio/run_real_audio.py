#!/usr/bin/env python3
"""
run_real_audio.py
Real audio extension for the physics_audio / mlpa Stiefel manifold model.

Usage:
    python run_real_audio.py --audio path/to/guitar_note.wav --duration 3.0
    python run_real_audio.py --audio_dir path/to/notes/ --analyze_batch
    python run_real_audio.py --audio note.wav --jump_test --low_patience 300
    python run_real_audio.py --live_mic --duration 5 --fit_after
    python run_real_audio.py --stft_benchmark
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
    JUMP_TEST_LOW_PATIENCE, JUMP_TEST_MIN_STEP, JUMP_TEST_FORCE_EVERY, JUMP_TEST_PLATEAU_AT,
    LIVE_MIC_BLOCKSIZE, LIVE_MIC_DEFAULT_SECONDS, GPU_STFT_N_FFT, GPU_STFT_HOP,
    fr_invariant_weight,
    fr_invariant_coupling,
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
from training_evaluation.streaming_stft import StreamingGPUSTFT
from training_evaluation.batch_analysis import analyze_batch_results
from training_evaluation.trainer import run_single_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AUDIO_EXTENSIONS = {'.wav', '.flac', '.ogg', '.mp3', '.aiff', '.aif'}


def load_audio(audio_path: str, sr: int = REAL_AUDIO_SR, duration: float = None):
    y, sr = librosa.load(audio_path, sr=sr, duration=duration, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)
    y = librosa.util.normalize(y)
    return y, sr


def build_jump_test_config(args) -> Optional[dict]:
    if not getattr(args, 'jump_test', False):
        return None
    cfg = {'verbose': True}
    if args.low_patience is not None:
        cfg['low_patience'] = args.low_patience
    else:
        cfg['low_patience'] = JUMP_TEST_LOW_PATIENCE
    if args.min_step_for_jump is not None:
        cfg['min_step_for_jump'] = args.min_step_for_jump
    else:
        cfg['min_step_for_jump'] = JUMP_TEST_MIN_STEP
    if args.force_jump_every is not None:
        cfg['force_jump_every'] = args.force_jump_every
    elif getattr(args, 'force_jumps', False):
        cfg['force_jump_every'] = JUMP_TEST_FORCE_EVERY
    if args.artificial_plateau_at is not None:
        cfg['artificial_plateau_at'] = args.artificial_plateau_at
    elif getattr(args, 'artificial_plateau', False):
        cfg['artificial_plateau_at'] = JUMP_TEST_PLATEAU_AT
    cfg['pop_size'] = getattr(args, 'jump_pop_size', None) or 6
    cfg['rollout_horizon'] = getattr(args, 'jump_rollout_horizon', None) or 800
    return cfg


def save_spectrogram_comparison(y_target, y_synth, sr, output_path, title="Real vs Synthesized"):
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


def save_partial_plot(partial_freqs, partial_amps, times, output_path, title_suffix=""):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for k in range(partial_freqs.shape[0]):
        axes[0].plot(times, partial_freqs[k], label=f"mode {k+1}")
        axes[1].plot(times, partial_amps[k], alpha=0.8)
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].legend(ncol=4, fontsize=8)
    axes[0].set_title(f"Tracked partial frequencies{title_suffix}")
    axes[1].set_ylabel("Magnitude")
    axes[1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def discover_audio_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob('*') if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)


def process_single_note(
    audio_path: Path,
    output_dir: Path,
    duration: float,
    max_steps: int,
    stft_weight: float,
    seed: int,
    streaming: bool = False,
    jump_test: Optional[dict] = None,
    use_gpu_stft: bool = False,
    fr_invariant_weight_override: float | None = None,
    fr_invariant_coupling_override: float | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    note_name = audio_path.stem

    print(f"\n{'='*60}")
    print(f"Processing: {audio_path}")

    y, sr = load_audio(str(audio_path), duration=duration)
    audio_duration = len(y) / sr
    print(f"Audio loaded: {audio_duration:.2f}s @ {sr} Hz")

    tracker_label = "streaming" if streaming else "piptrack + continuity"
    print(f"Extracting partials ({tracker_label})...")
    if streaming:
        partial_freqs, partial_amps, times, _ = extract_partials_streaming(y, sr, n_partials=K_MODES)
    else:
        partial_freqs, partial_amps, times, _ = extract_partials_piptrack(y, sr, n_partials=K_MODES)

    if use_gpu_stft:
        gpu_stft = StreamingGPUSTFT(n_fft=GPU_STFT_N_FFT, hop_length=GPU_STFT_HOP, sr=sr)
        chunk_t = torch.tensor(y, device=device, dtype=torch.float32)
        for i in range(0, len(chunk_t), GPU_STFT_HOP):
            gpu_stft.push(chunk_t[i:i + GPU_STFT_HOP])
        bench = gpu_stft.benchmark(n_chunks=50, chunk_samples=GPU_STFT_HOP)
        print(f"  GPU STFT: hop_latency={bench['hop_latency_ms']:.2f}ms mean_push={bench['mean_push_ms']:.3f}ms")

    save_partial_plot(partial_freqs, partial_amps, times, str(output_dir / "partials_tracked.png"),
                      title_suffix=f" ({tracker_label})")

    damping_rates, f0_est, b_est = estimate_physical_params(partial_freqs, partial_amps, times)
    print(f"Estimated f0 ≈ {f0_est:.1f} Hz, B ≈ {b_est:.6f}")

    data_points, model_times = partials_to_manifold_data(partial_freqs, partial_amps, times, f0_est)
    model = StiefelDampedCoupledInharmGR(DIM, K_MODES, data_points[0])
    model = initialize_model_from_physics(model, damping_rates, f0_est, b_est, data_points)

    prior_targets = build_prior_targets(damping_rates, f0_est, b_est)
    target_waveform = torch.tensor(y, device=device, dtype=torch.float32)

    active_fr_inv = fr_invariant_weight if fr_invariant_weight_override is None else fr_invariant_weight_override
    active_fr_coup = (
        fr_invariant_coupling if fr_invariant_coupling_override is None else fr_invariant_coupling_override
    )
    jump_label = " + jump_test" if jump_test else ""
    print(
        f"Running optimization ({max_steps} steps, STFT={stft_weight}, "
        f"fr_invariant={active_fr_inv}, fr_coupling={active_fr_coup}{jump_label})..."
    )
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
        fr_invariant_weight=active_fr_inv,
        fr_invariant_coupling=active_fr_coup,
        preinitialized_model=model,
        jump_test=jump_test,
    )

    learned_freqs = result['full_freq'].cpu().numpy()
    learned_damps = result['damping_rates'].cpu().numpy()
    coupling_strength = result['coupling_strength']
    coupling_skew = result['coupling_skew'].cpu().numpy()

    print(f"  Recon MSE: {result['total_recon_mse_pred']:.6f} | Jumps: {result['jumps']} | "
          f"Coupling: {coupling_strength:.4f}")

    y_synth = modal_synthesis(learned_freqs, learned_damps, sr, audio_duration,
                              coupling_strength=coupling_strength, coupling_skew=coupling_skew)
    sf.write(str(output_dir / "synthesized.wav"), y_synth, sr)
    save_spectrogram_comparison(y, y_synth, sr, str(output_dir / "spectrogram_comparison.png"),
                                title=f"{note_name}: Real vs Synthesized")

    summary = {
        'file': str(audio_path), 'name': note_name, 'duration_s': audio_duration,
        'f0_est_hz': f0_est, 'b_est': b_est,
        'recon_mse': result['total_recon_mse_pred'],
        'damping_rmse': result['damping_rmse'], 'damping_corr': result['damping_corr'],
        'coupling_strength': coupling_strength, 'jumps': result['jumps'],
        'wall_time_s': result['wall_time'],
        'learned_freqs': learned_freqs.tolist(), 'learned_damps': learned_damps.tolist(),
        'tracker': tracker_label,
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def run_batch(audio_dir, output_dir, duration, max_steps, stft_weight, seed, streaming,
              jump_test, use_gpu_stft, analyze_batch, n_clusters,
              fr_invariant_weight_override=None, fr_invariant_coupling_override=None):
    files = discover_audio_files(audio_dir)
    if not files:
        raise FileNotFoundError(f"No audio files in {audio_dir}")
    print(f"Batch: {len(files)} file(s)")
    results = []
    for i, p in enumerate(files):
        results.append(process_single_note(
            p, output_dir / p.stem, duration, max_steps, stft_weight,
            seed=seed + i, streaming=streaming, jump_test=jump_test, use_gpu_stft=use_gpu_stft,
            fr_invariant_weight_override=fr_invariant_weight_override,
            fr_invariant_coupling_override=fr_invariant_coupling_override,
        ))
    csv_path = output_dir / "batch_summary.csv"
    fields = ['name', 'file', 'duration_s', 'f0_est_hz', 'b_est', 'recon_mse',
              'damping_rmse', 'damping_corr', 'coupling_strength', 'jumps', 'wall_time_s', 'tracker']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fields})
    print(f"Batch CSV: {csv_path}")
    if analyze_batch:
        analyze_batch_results(results, output_dir, n_clusters=n_clusters)
    return results


def run_stft_benchmark():
    print("GPU streaming STFT benchmark")
    for dev_name in (['cuda'] if torch.cuda.is_available() else []) + ['cpu']:
        dev = torch.device(dev_name)
        stft = StreamingGPUSTFT(n_fft=GPU_STFT_N_FFT, hop_length=GPU_STFT_HOP, dev=dev)
        bench = stft.benchmark(n_chunks=500, chunk_samples=GPU_STFT_HOP)
        print(f"  [{dev_name}] hop={bench['hop_latency_ms']:.2f}ms "
              f"mean={bench['mean_push_ms']:.3f}ms max={bench['max_push_ms']:.3f}ms "
              f"sub10ms={bench['under_10ms']}")


def run_live_mic(duration, sr, output_dir, fit_after, max_steps, stft_weight, streaming):
    try:
        import sounddevice as sd
    except ImportError as e:
        raise ImportError("Install sounddevice: pip install sounddevice") from e

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker = StreamingPartialTracker(sr=sr, n_partials=K_MODES)
    gpu_stft = StreamingGPUSTFT(n_fft=GPU_STFT_N_FFT, hop_length=GPU_STFT_HOP, sr=sr)
    recorded = []

    print(f"Live mic: {duration}s @ {sr}Hz (blocksize={LIVE_MIC_BLOCKSIZE})")
    print("Speak or play now...")

    def callback(indata, frames, time_info, status):
        if status:
            print(f"  [sd] {status}")
        chunk = indata[:, 0].astype(np.float32)
        recorded.append(chunk.copy())
        state = tracker.process_chunk(chunk)
        gpu_stft.push(torch.tensor(chunk, device=device, dtype=torch.float32))
        if state is not None:
            print(f"  t={state['time']:.2f}s f0={state['freqs'][0]:.1f}Hz "
                  f"gpu_stft={gpu_stft._last_frame_ms:.2f}ms", flush=True)

    with sd.InputStream(samplerate=sr, channels=1, blocksize=LIVE_MIC_BLOCKSIZE, callback=callback):
        sd.sleep(int(duration * 1000))

    if not recorded:
        print("No audio captured.")
        return

    y = np.concatenate(recorded)
    y, _ = librosa.effects.trim(y, top_db=30)
    if y.size == 0:
        print("Captured silence only.")
        return
    y = librosa.util.normalize(y)

    wav_path = output_dir / "live_recording.wav"
    sf.write(str(wav_path), y, sr)
    print(f"Saved recording: {wav_path} ({len(y)/sr:.2f}s)")

    partial_freqs, partial_amps, times, _ = tracker.finalize()
    save_partial_plot(partial_freqs, partial_amps, times,
                      str(output_dir / "live_partials.png"), title_suffix=" (live)")

    bench = gpu_stft.benchmark(n_chunks=100, chunk_samples=GPU_STFT_HOP)
    print(f"GPU STFT benchmark: mean={bench['mean_push_ms']:.3f}ms sub10ms={bench['under_10ms']}")

    if fit_after:
        print("Fitting model on live recording...")
        process_single_note(
            wav_path, output_dir / "live_fit", len(y) / sr,
            max_steps, stft_weight, seed=0, streaming=streaming,
        )


def demo_streaming_live(audio_path, duration=3.0):
    y, sr = load_audio(audio_path, duration=duration)
    tracker = StreamingPartialTracker(sr=sr, n_partials=K_MODES)
    gpu_stft = StreamingGPUSTFT(n_fft=GPU_STFT_N_FFT, hop_length=GPU_STFT_HOP, sr=sr)
    chunk_size = tracker.chunk_size
    for start in range(0, len(y), chunk_size):
        chunk = y[start:start + chunk_size]
        state = tracker.process_chunk(chunk)
        gpu_stft.push(torch.tensor(chunk, device=device, dtype=torch.float32))
        if state:
            print(f"  t={state['time']:.3f}s f0={state['freqs'][0]:.1f}Hz "
                  f"gpu_stft={gpu_stft._last_frame_ms:.2f}ms")
    freqs, _, times = tracker.get_trajectories()
    print(f"Done: {times.shape[0]} frames, f0 [{freqs[0].min():.1f}, {freqs[0].max():.1f}] Hz")


def main():
    parser = argparse.ArgumentParser(description="Real-audio fitting pipeline (Phase 4)")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--audio", type=str, help="Single audio file")
    src.add_argument("--audio_dir", type=str, help="Folder for batch processing")
    src.add_argument("--live_mic", action="store_true", help="Record from microphone")
    src.add_argument("--stft_benchmark", action="store_true", help="Benchmark GPU streaming STFT")

    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--output_dir", type=str, default="real_audio_results")
    parser.add_argument("--max_steps", type=int, default=REAL_AUDIO_MAX_STEPS)
    parser.add_argument("--stft_weight", type=float, default=REAL_AUDIO_STFT_WEIGHT)
    parser.add_argument(
        "--fr_invariant_weight",
        type=float,
        default=None,
        help=f"Damping double-coset prior weight (default config: {fr_invariant_weight})",
    )
    parser.add_argument(
        "--fr_invariant_coupling",
        type=float,
        default=None,
        help=f"Coupling skew singular-value invariant weight (default config: {fr_invariant_coupling})",
    )
    parser.add_argument("--no_stft", action="store_true")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--gpu_stft", action="store_true", help="Enable GPU streaming STFT path")
    parser.add_argument("--stream_demo", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    # Jump testing
    parser.add_argument("--jump_test", action="store_true", help="Enable jump-test mode")
    parser.add_argument("--force_jumps", action="store_true", help="Force periodic jumps")
    parser.add_argument("--low_patience", type=int, default=None, help="Override stagnation patience")
    parser.add_argument("--min_step_for_jump", type=int, default=None)
    parser.add_argument("--force_jump_every", type=int, default=None, help="Force jump every N steps")
    parser.add_argument("--artificial_plateau", action="store_true", help="Freeze LR to simulate plateau")
    parser.add_argument("--artificial_plateau_at", type=int, default=None)
    parser.add_argument("--jump_pop_size", type=int, default=None, help="Smaller pop for jump tests")
    parser.add_argument("--jump_rollout_horizon", type=int, default=None)

    # Batch analysis
    parser.add_argument("--analyze_batch", action="store_true", help="Cluster notes + key detection")
    parser.add_argument("--n_clusters", type=int, default=None)

    # Live mic
    parser.add_argument("--fit_after", action="store_true", help="Fit model after live recording")

    args = parser.parse_args()
    stft_weight = 0.0 if args.no_stft else args.stft_weight
    jump_test = build_jump_test_config(args)
    output_dir = Path(args.output_dir)

    if args.stft_benchmark:
        run_stft_benchmark()
        return

    if args.live_mic:
        run_live_mic(args.duration or LIVE_MIC_DEFAULT_SECONDS, REAL_AUDIO_SR, output_dir,
                     args.fit_after, args.max_steps, stft_weight, args.streaming)
        return

    if args.stream_demo:
        if not args.audio:
            parser.error("--stream_demo requires --audio")
        demo_streaming_live(args.audio, args.duration)
        return

    if args.audio_dir:
        run_batch(Path(args.audio_dir), output_dir, args.duration, args.max_steps,
                  stft_weight, args.seed, args.streaming, jump_test, args.gpu_stft,
                  args.analyze_batch, args.n_clusters, args.fr_invariant_weight,
                  args.fr_invariant_coupling)
    else:
        process_single_note(Path(args.audio), output_dir, args.duration, args.max_steps,
                            stft_weight, args.seed, args.streaming, jump_test, args.gpu_stft,
                            fr_invariant_weight_override=args.fr_invariant_weight,
                            fr_invariant_coupling_override=args.fr_invariant_coupling)


if __name__ == "__main__":
    main()