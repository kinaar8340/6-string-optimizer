"""
GPU-accelerated streaming STFT for sub-10ms frame latency.

Uses a ring buffer + torch.stft on CUDA (or CPU fallback) for incremental
magnitude-spectrum frames suitable for real-time monitoring and fast STFT loss.
"""

from __future__ import annotations

import time
import torch
import torch.nn.functional as F
from typing import Optional

from .config import device, REAL_AUDIO_STFT_HOP_RATIO


class StreamingGPUSTFT:
    """
    Incremental STFT on GPU with ring-buffered audio.

    Default n_fft=512, hop=128 → ~2.9ms hop latency @ 44.1kHz (well under 10ms).
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: Optional[int] = None,
        sr: int = 44100,
        dev: Optional[torch.device] = None,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length or max(64, int(n_fft * REAL_AUDIO_STFT_HOP_RATIO))
        self.sr = sr
        self.dev = dev or device

        self._window = torch.hann_window(n_fft, device=self.dev)
        self._buffer = torch.zeros(n_fft, device=self.dev, dtype=torch.float32)
        self._pending = torch.zeros(0, device=self.dev, dtype=torch.float32)
        self._frames: list[torch.Tensor] = []
        self._total_samples = 0
        self._last_frame_ms: float = 0.0

    @property
    def hop_latency_ms(self) -> float:
        return 1000.0 * self.hop_length / self.sr

    @property
    def window_latency_ms(self) -> float:
        return 1000.0 * self.n_fft / self.sr

    def reset(self) -> None:
        self._buffer.zero_()
        self._pending = torch.zeros(0, device=self.dev, dtype=torch.float32)
        self._frames.clear()
        self._total_samples = 0

    def push(self, chunk: torch.Tensor) -> list[torch.Tensor]:
        """
        Push audio chunk (1D). Returns list of new magnitude frames produced.
        Each frame shape: (n_fft // 2 + 1,).
        """
        t0 = time.perf_counter()
        chunk = chunk.to(self.dev, dtype=torch.float32).flatten()
        self._pending = torch.cat([self._pending, chunk])
        new_frames: list[torch.Tensor] = []

        while self._pending.numel() >= self.hop_length:
            step = self._pending[:self.hop_length]
            self._pending = self._pending[self.hop_length:]
            self._buffer = torch.cat([self._buffer[self.hop_length:], step])
            self._total_samples += self.hop_length

            spec = torch.stft(
                self._buffer, n_fft=self.n_fft, hop_length=self.n_fft,
                window=self._window, return_complex=True, center=False,
            )
            mag = spec.abs().squeeze(-1)
            new_frames.append(mag)
            self._frames.append(mag)

        self._last_frame_ms = (time.perf_counter() - t0) * 1000.0
        return new_frames

    def get_accumulated_magnitude(self) -> Optional[torch.Tensor]:
        """Return (n_bins, n_frames) magnitude spectrogram accumulated so far."""
        if not self._frames:
            return None
        return torch.stack(self._frames, dim=1)

    def streaming_stft_loss(
        self,
        target_frames: list[torch.Tensor],
        pred_frames: list[torch.Tensor],
    ) -> torch.Tensor:
        """L1 loss between matched streaming magnitude frames."""
        n = min(len(target_frames), len(pred_frames))
        if n == 0:
            return torch.tensor(0.0, device=self.dev)
        loss = torch.tensor(0.0, device=self.dev)
        for i in range(n):
            loss = loss + F.l1_loss(pred_frames[i], target_frames[i])
        return loss / n

    def benchmark(self, n_chunks: int = 200, chunk_samples: int = 128) -> dict:
        """Benchmark push() latency — reports mean/max ms per chunk."""
        self.reset()
        latencies = []
        dummy = torch.randn(chunk_samples, device=self.dev)
        for _ in range(n_chunks):
            t0 = time.perf_counter()
            self.push(dummy)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        return {
            'device': str(self.dev),
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'hop_latency_ms': self.hop_latency_ms,
            'mean_push_ms': float(sum(latencies) / len(latencies)),
            'max_push_ms': float(max(latencies)),
            'under_10ms': max(latencies) < 10.0,
        }


def multi_resolution_stft_loss_gpu(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    fft_sizes: list | None = None,
    hop_ratio: float = REAL_AUDIO_STFT_HOP_RATIO,
    dev: Optional[torch.device] = None,
) -> torch.Tensor:
    """GPU-optimized multi-resolution STFT loss (all tensors on device)."""
    dev = dev or y_pred.device
    if fft_sizes is None:
        from .config import REAL_AUDIO_STFT_FFT_SIZES
        fft_sizes = REAL_AUDIO_STFT_FFT_SIZES

    y_pred = y_pred.float().flatten().to(dev)
    y_true = y_true.float().flatten().to(dev)
    min_len = min(y_pred.shape[0], y_true.shape[0])
    y_pred = y_pred[:min_len]
    y_true = y_true[:min_len]

    total = torch.tensor(0.0, device=dev)
    for n_fft in fft_sizes:
        hop = max(1, int(n_fft * hop_ratio))
        window = torch.hann_window(n_fft, device=dev)
        spec_pred = torch.stft(y_pred, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
        spec_true = torch.stft(y_true, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
        total = total + F.l1_loss(spec_pred.abs(), spec_true.abs())
    return total / len(fft_sizes)