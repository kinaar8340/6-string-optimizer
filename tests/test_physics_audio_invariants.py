"""Tests for physics_audio double-coset invariant losses."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PHYSICS_AUDIO = ROOT / "physics_audio"
for p in (ROOT, PHYSICS_AUDIO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from training_evaluation.invariant_losses import (
    damping_invariant_loss,
    harmonic_design_matrix,
    learned_damping_coefficients,
    target_damping_coefficients,
)


def test_damping_invariant_zero_when_coefficients_match():
    log_base = torch.log(torch.tensor(0.05))
    log_slope = torch.log(torch.tensor(0.02))
    coeffs = learned_damping_coefficients(log_base, log_slope)
    design = harmonic_design_matrix(8, device=coeffs.device)
    target = design @ coeffs
    loss = damping_invariant_loss(log_base, log_slope, target)
    assert loss.item() < 1e-8


def test_damping_invariant_penalizes_mismatched_envelope_scale():
    log_base = torch.log(torch.tensor(0.05))
    log_slope = torch.log(torch.tensor(0.02))
    coeffs = learned_damping_coefficients(log_base, log_slope)
    design = harmonic_design_matrix(8, device=coeffs.device)
    target = design @ coeffs

    scaled_target = target * 2.5
    loss_raw = damping_invariant_loss(log_base, log_slope, scaled_target)
    mu2_scaled = target_damping_coefficients(scaled_target, design)
    mu2_base = target_damping_coefficients(target, design)
    assert not torch.allclose(mu2_scaled, mu2_base)
    assert loss_raw.item() > 1e-6


def test_harmonic_design_matrix_shape():
    design = harmonic_design_matrix(8, device=torch.device("cpu"))
    assert design.shape == (8, 2)
    assert torch.allclose(design[:, 0], torch.ones(8))
    assert torch.allclose(design[:, 1], torch.arange(8, dtype=torch.float32))


if __name__ == "__main__":
    test_damping_invariant_zero_when_coefficients_match()
    test_damping_invariant_penalizes_mismatched_envelope_scale()
    test_harmonic_design_matrix_shape()
    print("physics_audio invariant tests passed.")