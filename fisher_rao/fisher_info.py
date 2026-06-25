"""Fisher information estimators for adaptive burst and damping modulation."""

from __future__ import annotations

import torch
from torch import Tensor


def score_norm_from_grads(params: list[Tensor]) -> Tensor:
    """Fisher info proxy from already-computed parameter gradients (post-backward)."""
    total = None
    for p in params:
        if p.grad is None:
            continue
        g2 = p.grad.detach().pow(2).sum()
        total = g2 if total is None else total + g2
    if total is None:
        return torch.tensor(0.0)
    return total


def score_norm_proxy(params: list[Tensor], log_likelihood: Tensor) -> Tensor:
    """Lightweight Fisher info proxy: squared gradient norm of log-likelihood."""
    grads = torch.autograd.grad(
        log_likelihood,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    total = torch.tensor(0.0, device=log_likelihood.device, dtype=log_likelihood.dtype)
    for g in grads:
        if g is not None:
            total = total + g.pow(2).sum()
    return total


def fisher_info_trace_hvp(
    params: list[Tensor],
    log_likelihood: Tensor,
    num_probes: int = 1,
) -> Tensor:
    """Hutchinson trace estimator for the Fisher information matrix."""
    device = log_likelihood.device
    dtype = log_likelihood.dtype
    trace_est = torch.tensor(0.0, device=device, dtype=dtype)

    for _ in range(num_probes):
        vs = [torch.randint(0, 2, p.shape, device=device, dtype=dtype) * 2 - 1 for p in params]
        grad = torch.autograd.grad(log_likelihood, params, create_graph=True, allow_unused=True)
        grad_flat = [g if g is not None else torch.zeros_like(p) for g, p in zip(grad, params)]
        hvp = torch.autograd.grad(grad_flat, params, grad_outputs=vs, retain_graph=True, allow_unused=True)
        for v, hv in zip(vs, hvp):
            if hv is not None:
                trace_est = trace_est + (v * hv).sum()

    return trace_est / max(num_probes, 1)


def fisher_modulation_factors(
    fisher_info: Tensor,
    info_scale: float = 1.0,
    min_damping_boost: float = 0.0,
    max_damping_boost: float = 0.08,
    min_burst_damp: float = 0.5,
    max_burst_damp: float = 1.0,
    min_threshold_scale: float = 0.85,
    max_threshold_scale: float = 1.25,
) -> dict[str, float]:
    """Map Fisher information to burst/damping modulation factors."""
    fi = fisher_info.detach().float().item() if torch.is_tensor(fisher_info) else float(fisher_info)
    sensitivity = 1.0 - torch.exp(torch.tensor(-fi / max(info_scale, 1e-8))).item()

    damping_boost = min_damping_boost + (max_damping_boost - min_damping_boost) * sensitivity
    burst_factor_scale = max_burst_damp - (max_burst_damp - min_burst_damp) * sensitivity
    threshold_scale = min_threshold_scale + (max_threshold_scale - min_threshold_scale) * sensitivity
    stagnation_boost = 1.0 + (1.0 - sensitivity) * 2.0

    return {
        "damping_boost": damping_boost,
        "burst_factor_scale": burst_factor_scale,
        "threshold_scale": threshold_scale,
        "stagnation_boost": stagnation_boost,
        "sensitivity": sensitivity,
    }