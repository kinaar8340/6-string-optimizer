"""
Fisher-information-aware modulation for GeooptBurstOptimizer.

Augments loss-scaled damping with information-geometric sensitivity estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .fisher_info import fisher_modulation_factors, score_norm_from_grads, score_norm_proxy


@dataclass
class FisherInfoBurstModulator:
    """
    Stateful modulator that tracks Fisher information and produces
    per-step scaling factors for burst/damping/stagnation behavior.
    """

    info_scale: float = 100.0
    ema_alpha: float = 0.1
    use_hvp: bool = False
    hvp_probes: int = 1
    smoothed_info: float | None = field(default=None, init=False)

    def update(
        self,
        params: list[torch.Tensor],
        log_likelihood: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """
        Estimate Fisher information and return modulation factors.

        Prefers score_norm_from_grads (post-backward) when gradients are available;
        falls back to autograd score_norm_proxy when log_likelihood is provided.
        """
        has_grads = any(p.grad is not None for p in params)
        if has_grads:
            raw_info = score_norm_from_grads(params)
        elif log_likelihood is not None and self.use_hvp:
            from .fisher_info import fisher_info_trace_hvp
            raw_info = fisher_info_trace_hvp(params, log_likelihood, num_probes=self.hvp_probes)
        elif log_likelihood is not None:
            raw_info = score_norm_proxy(params, log_likelihood)
        else:
            raw_info = torch.tensor(0.0)

        raw_val = raw_info.detach().float().item()

        if self.smoothed_info is None:
            self.smoothed_info = raw_val
        else:
            a = self.ema_alpha
            self.smoothed_info = (1 - a) * self.smoothed_info + a * raw_val

        factors = fisher_modulation_factors(
            torch.tensor(self.smoothed_info),
            info_scale=self.info_scale,
        )
        factors["raw_fisher_info"] = raw_val
        factors["smoothed_fisher_info"] = self.smoothed_info
        return factors

    def apply_to_group(
        self,
        group: dict,
        factors: dict[str, float],
        base_damping: float,
        base_burst_factor: float,
        base_threshold: float,
    ) -> tuple[float, float, float]:
        """Apply modulation factors to optimizer hyperparameters for one param group."""
        effective_damping = min(
            0.999,
            base_damping + factors["damping_boost"],
        )
        effective_burst = base_burst_factor * factors["burst_factor_scale"]
        effective_threshold = base_threshold * factors["threshold_scale"]
        return effective_damping, effective_burst, effective_threshold