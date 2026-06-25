#!/usr/bin/env python3
"""
Fisher-Rao Phase 1–2 smoke test on synthetic Stiefel data.

Arms (identical init):
  1. baseline — geo + priors
  2. phase1 — + damping invariant (Prop 3.3)
  3. phase2 — + coupling SV invariant + speed/inharm FR profiles

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
    fr_invariant_coupling,
    fr_invariant_speed,
    fr_invariant_inharm,
)
from training_evaluation.model import StiefelDampedCoupledInharmGR
from training_evaluation.utils import safe_proj, get_pca_initial_basis, manifold
from training_evaluation.losses import total_loss
from training_evaluation.fr_utils import (
    extract_coupling_skew,
    fr_loss_kwargs_from_batch,
    mode_amplitudes_from_stiefel,
    modal_spectral_envelope,
)

SMOKE_N = 100


def generate_synthetic(seed: int, noise_amp: float, device: torch.device):
    """Match trainer synthetic physics including coupled modes."""
    g = torch.Generator(device=device).manual_seed(seed)
    true_base = safe_proj(torch.randn(DIM, K_MODES, device=device, generator=g))
    true_vel_dir = manifold.proju(true_base, torch.randn(DIM, K_MODES, device=device, generator=g))
    true_vel_dir = true_vel_dir / true_vel_dir.norm(dim=0, keepdim=True).clamp(min=1e-8)

    true_freq = IDEAL_HARMONICS * torch.sqrt(1 + TRUE_INHARM_B * IDEAL_HARMONICS.pow(2))
    true_vel = true_vel_dir * VELOCITY_SCALE_BASE * true_freq

    true_coupling_raw = torch.randn(K_MODES, K_MODES, device=device, generator=g) * 0.05
    true_coupling_skew = true_coupling_raw.tril(diagonal=-1) - true_coupling_raw.triu(diagonal=1)
    true_coupling_vel = manifold.proju(true_base, true_base @ true_coupling_skew)
    true_vel_total = true_vel + TRUE_COUPLING_STRENGTH * true_coupling_vel

    times = torch.linspace(-2.0, 2.0, SMOKE_N, device=device)
    envelope = torch.exp(-TRUE_DAMPING_RATES * torch.abs(times).view(-1, 1, 1))
    base_batch = true_base.unsqueeze(0).expand(SMOKE_N, -1, -1)
    vel_batch = times.view(-1, 1, 1) * true_vel_total.unsqueeze(0) * envelope
    exact = manifold.expmap(base_batch, vel_batch)
    data = exact + noise_amp * torch.randn_like(exact, generator=g)

    refs = {
        "coupling_skew": true_coupling_skew.detach(),
        "speed_scalars": torch.ones(K_MODES, device=device) * VELOCITY_SCALE_BASE,
        "inharm_b": TRUE_INHARM_B,
        "coupling_strength": TRUE_COUPLING_STRENGTH,
    }
    return data, times, get_pca_initial_basis(data, K_MODES), refs


def run_arm(
    label: str,
    data_points: torch.Tensor,
    times: torch.Tensor,
    initial_basis: torch.Tensor,
    state_dict: dict,
    refs: dict,
    steps: int,
    *,
    fr_invariant_weight: float = 0.0,
    fr_invariant_coupling: float = 0.0,
    fr_invariant_speed: float = 0.0,
    fr_invariant_inharm: float = 0.0,
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
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=USE_AMP and device.type == "cuda"):
            preds, damping_rates, coupling_strength, inharm_b, speed_scalars, _ = model(times)
            loss = total_loss(
                preds, data_points, damping_rates, coupling_strength, inharm_b, speed_scalars,
                log_base_rate=model.log_base_rate,
                log_slope=model.log_slope,
                coupling_skew=extract_coupling_skew(model),
                target_coupling_skew=refs["coupling_skew"],
                target_speed_scalars=refs["speed_scalars"],
                target_inharm_b=refs["inharm_b"],
                fr_invariant_weight_override=fr_invariant_weight,
                fr_invariant_coupling_override=fr_invariant_coupling,
                fr_invariant_speed_override=fr_invariant_speed,
                fr_invariant_inharm_override=fr_invariant_inharm,
                **fr_loss_kwargs_from_batch(
                    preds, data_points, fr_mode_weight=0.0, fr_spectral_weight=0.0,
                    fr_invariant_weight=fr_invariant_weight,
                    target_mode_amps=target_mode_amps, target_spectrum=target_spectrum,
                ),
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        best = min(best, loss.item())

    with torch.no_grad():
        preds, damping_rates, coupling_strength, inharm_b, speed_scalars, _ = model(times)
        final = total_loss(
            preds, data_points, damping_rates, coupling_strength, inharm_b, speed_scalars,
            log_base_rate=model.log_base_rate,
            log_slope=model.log_slope,
            coupling_skew=extract_coupling_skew(model),
            target_coupling_skew=refs["coupling_skew"],
            target_speed_scalars=refs["speed_scalars"],
            target_inharm_b=refs["inharm_b"],
            fr_invariant_weight_override=fr_invariant_weight,
            fr_invariant_coupling_override=fr_invariant_coupling,
            fr_invariant_speed_override=fr_invariant_speed,
            fr_invariant_inharm_override=fr_invariant_inharm,
        ).item()
        learned_skew = extract_coupling_skew(model)
        sv_err = (torch.linalg.svdvals(learned_skew.float())
                  - torch.linalg.svdvals(refs["coupling_skew"].float())).pow(2).sum().item()

    return {
        "label": label,
        "final_loss": final,
        "best_loss": best,
        "coupling_sv_mse": sv_err,
    }


def main():
    p = argparse.ArgumentParser(description="Fisher-Rao Phase 1–2 physics_audio smoke test")
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise", type=float, default=0.04)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, times, basis, refs = generate_synthetic(args.seed, args.noise, device)

    torch.manual_seed(args.seed)
    init_state = copy.deepcopy(StiefelDampedCoupledInharmGR(DIM, K_MODES, basis).to(device).state_dict())

    arms = [
        ("baseline", 0.0, 0.0, 0.0, 0.0),
        ("phase1_damping", 0.3, 0.0, 0.0, 0.0),
        ("phase2_full", 0.3, fr_invariant_coupling, fr_invariant_speed, fr_invariant_inharm),
    ]

    print(f"Phase 1–2 smoke | dim={DIM} k={K_MODES} steps={args.steps} seed={args.seed}\n")
    results = []
    for label, inv_w, coup_w, speed_w, inharm_w in arms:
        r = run_arm(
            label, data, times, basis, init_state, refs, args.steps,
            fr_invariant_weight=inv_w,
            fr_invariant_coupling=coup_w,
            fr_invariant_speed=speed_w,
            fr_invariant_inharm=inharm_w,
        )
        results.append(r)
        print(
            f"  {label:18s} | final={r['final_loss']:.4f} best={r['best_loss']:.4f} "
            f"| coupling_sv_mse={r['coupling_sv_mse']:.4e}"
        )

    delta = results[-1]["best_loss"] - results[0]["best_loss"]
    print(f"\nΔ best_loss (phase2 - baseline): {delta:+.4f}")


if __name__ == "__main__":
    main()