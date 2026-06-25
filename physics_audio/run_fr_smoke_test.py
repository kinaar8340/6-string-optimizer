#!/usr/bin/env python3
"""
Phase 1 smoke test: physics_audio Fisher-Rao losses on synthetic Stiefel data.

Compares three arms with identical init:
  1. baseline (geo + priors only)
  2. + damping invariant (Prop 3.3, fr_invariant_weight)
  3. + damping invariant + modal FR loss (fr_mode_weight)

Usage (from physics_audio/):
  python run_fr_smoke_test.py --steps 1200
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
from geoopt.optim import RiemannianAdam
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training_evaluation.config import (
    DIM,
    K_MODES,
    TRUE_DAMPING_RATES,
    TRUE_INHARM_B,
    TRUE_COUPLING_STRENGTH,
    VELOCITY_SCALE_BASE,
    IDEAL_HARMONICS,
    USE_AMP,
)
from training_evaluation.model import StiefelDampedCoupledInharmGR
from training_evaluation.utils import safe_proj, get_pca_initial_basis, manifold
from training_evaluation.losses import total_loss
from training_evaluation.fr_utils import mode_amplitudes_from_stiefel, modal_spectral_envelope, fr_loss_kwargs_from_batch
from training_evaluation.audio_utils import extract_coupling_skew

SMOKE_N = 100


def generate_synthetic(seed: int, noise_amp: float, device: torch.device):
    g = torch.Generator(device=device).manual_seed(seed)
    true_base = safe_proj(torch.randn(DIM, K_MODES, device=device, generator=g))
    true_vel_dir = manifold.proju(true_base, torch.randn(DIM, K_MODES, device=device, generator=g))
    true_vel_dir = true_vel_dir / true_vel_dir.norm(dim=0, keepdim=True).clamp(min=1e-8)

    true_freq = IDEAL_HARMONICS * torch.sqrt(
        1 + TRUE_INHARM_B * IDEAL_HARMONICS.pow(2)
    )
    true_vel = true_vel_dir * VELOCITY_SCALE_BASE * true_freq

    times = torch.linspace(-2.0, 2.0, SMOKE_N, device=device)
    envelope = torch.exp(-TRUE_DAMPING_RATES * torch.abs(times).view(-1, 1, 1))
    base_batch = true_base.unsqueeze(0).expand(SMOKE_N, -1, -1)
    vel_batch = times.view(-1, 1, 1) * true_vel.unsqueeze(0) * envelope
    exact = manifold.expmap(base_batch, vel_batch)
    data = exact + noise_amp * torch.randn_like(exact, generator=g)
    return data, times, get_pca_initial_basis(data, K_MODES)


def run_arm(
    label: str,
    data_points: torch.Tensor,
    times: torch.Tensor,
    initial_basis: torch.Tensor,
    state_dict: dict,
    steps: int,
    *,
    fr_invariant_weight: float = 0.0,
    fr_mode_weight: float = 0.0,
    fr_spectral_weight: float = 0.0,
) -> dict:
    device = data_points.device
    target_mode_amps = mode_amplitudes_from_stiefel(data_points)
    target_spectrum = modal_spectral_envelope(data_points)

    model = StiefelDampedCoupledInharmGR(DIM, K_MODES, initial_basis)
    model.load_state_dict(copy.deepcopy(state_dict))
    model.to(device)

    optimizer = RiemannianAdam([
        {"params": model.base, "lr": 0.08},
        {"params": [p for n, p in model.named_parameters() if p is not model.base], "lr": 0.0005},
    ], stabilize=10)
    scaler = GradScaler("cuda", enabled=USE_AMP and device.type == "cuda")

    best = float("inf")
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=USE_AMP and device.type == "cuda"):
            preds, damping_rates, coupling_strength, inharm_b, speed_scalars, _ = model(times)
            fr_kw = fr_loss_kwargs_from_batch(
                preds, data_points,
                fr_mode_weight=fr_mode_weight,
                fr_spectral_weight=fr_spectral_weight,
                target_mode_amps=target_mode_amps,
                target_spectrum=target_spectrum,
            )
            loss = total_loss(
                preds, data_points, damping_rates, coupling_strength, inharm_b, speed_scalars,
                log_base_rate=model.log_base_rate,
                log_slope=model.log_slope,
                coupling_skew=extract_coupling_skew(model),
                fr_invariant_weight_override=fr_invariant_weight,
                **fr_kw,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        best = min(best, loss.item())

    with torch.no_grad():
        preds, damping_rates, coupling_strength, inharm_b, speed_scalars, _ = model(times)
        fr_kw = fr_loss_kwargs_from_batch(
            preds, data_points,
            fr_mode_weight=fr_mode_weight,
            fr_spectral_weight=fr_spectral_weight,
            target_mode_amps=target_mode_amps,
            target_spectrum=target_spectrum,
        )
        final = total_loss(
            preds, data_points, damping_rates, coupling_strength, inharm_b, speed_scalars,
            log_base_rate=model.log_base_rate,
            log_slope=model.log_slope,
            coupling_skew=extract_coupling_skew(model),
            fr_invariant_weight_override=fr_invariant_weight,
            **fr_kw,
        ).item()
        geo = (preds.float() - data_points.float()).pow(2).mean().item()

    return {"label": label, "final_loss": final, "best_loss": best, "geo_mse": geo}


def main():
    p = argparse.ArgumentParser(description="Phase 1 Fisher-Rao physics_audio smoke test")
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise", type=float, default=0.04)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, times, basis = generate_synthetic(args.seed, args.noise, device)

    torch.manual_seed(args.seed)
    probe = StiefelDampedCoupledInharmGR(DIM, K_MODES, basis).to(device)
    init_state = copy.deepcopy(probe.state_dict())

    arms = [
        ("baseline", 0.0, 0.0, 0.0),
        ("+invariant", 0.3, 0.0, 0.0),
        ("+invariant+modal_fr", 0.3, 0.15, 0.0),
    ]

    print(f"Phase 1 smoke test | dim={DIM} k={K_MODES} steps={args.steps} seed={args.seed}\n")
    results = []
    for label, inv_w, mode_w, spec_w in arms:
        r = run_arm(
            label, data, times, basis, init_state, args.steps,
            fr_invariant_weight=inv_w,
            fr_mode_weight=mode_w,
            fr_spectral_weight=spec_w,
        )
        results.append(r)
        print(f"  {label:22s} | final={r['final_loss']:.4f} best={r['best_loss']:.4f} geo_mse={r['geo_mse']:.4f}")

    base_best = results[0]["best_loss"]
    full_best = results[-1]["best_loss"]
    print(f"\nΔ best_loss (full FR - baseline): {full_best - base_best:+.4f}")
    if full_best < base_best:
        print("Phase 1 smoke test: Fisher-Rao arm improved over baseline.")
    else:
        print("Phase 1 smoke test: no improvement at this step budget (try more steps).")


if __name__ == "__main__":
    main()