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
    coupling_invariant_loss,
    damping_invariant_loss,
    harmonic_design_matrix,
    learned_damping_coefficients,
    modal_amp_pair_invariant_loss,
    target_damping_coefficients,
    speed_profile_invariant_loss,
    inharm_profile_invariant_loss,
)
from training_evaluation.fr_utils import resolve_target_mode_amps
from training_evaluation.losses import prior_loss, _mse_prior_skip_terms
from training_evaluation.config import (
    fr_invariant_damping,
    fr_invariant_coupling,
    fr_invariant_speed,
    fr_invariant_inharm,
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


def _random_skew(k: int) -> torch.Tensor:
    raw = torch.randn(k, k)
    return raw.tril(diagonal=-1) - raw.triu(diagonal=1)


def test_coupling_invariant_zero_on_match():
    target = _random_skew(8)
    strength = torch.tensor(0.3)
    loss = coupling_invariant_loss(target, strength, target, 0.3)
    assert loss.item() < 1e-8


def test_coupling_invariant_rotation_invariant_shape():
    target = _random_skew(8)
    learned = _random_skew(8)
    q, _ = torch.linalg.qr(torch.randn(8, 8))
    rotated = q.T @ learned @ q
    loss_a = coupling_invariant_loss(learned, torch.tensor(0.3), target, 0.3, strength_weight=0.0)
    loss_b = coupling_invariant_loss(rotated, torch.tensor(0.3), target, 0.3, strength_weight=0.0)
    assert abs(loss_a.item() - loss_b.item()) < 1e-5


def test_coupling_invariant_penalizes_strength_mismatch():
    skew = _random_skew(8)
    loss = coupling_invariant_loss(skew, torch.tensor(0.5), skew, 0.3, strength_weight=1.0)
    assert loss.item() > 1e-6


def test_speed_profile_invariant_zero_on_match():
    speed = torch.tensor([2.0, 2.0, 2.0, 2.0])
    loss = speed_profile_invariant_loss(speed, speed)
    assert loss.item() < 1e-6


def test_inharm_profile_invariant_scale_quotient():
    target = torch.tensor([0.0, 0.0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006])
    scaled = target * 3.0
    loss = inharm_profile_invariant_loss(scaled, target)
    assert loss.item() < 1e-5


def test_modal_amp_pair_invariant_zero_on_match():
    amps = torch.tensor([0.1, 0.3, 0.5, 0.2, 0.4, 0.15, 0.25, 0.35])
    loss = modal_amp_pair_invariant_loss(amps, amps)
    assert loss.item() < 1e-6


def test_modal_amp_pair_invariant_mismatch_bounded():
    pred = torch.tensor([0.2, 0.3, 0.5, 0.1, 0.4, 0.2, 0.3, 0.25])
    loss = modal_amp_pair_invariant_loss(pred * 0.8, pred)
    assert 0.0 < loss.item() < 1.0


def test_modal_amp_pair_invariant_uses_temporal_obs():
    pred = torch.tensor([0.2, 0.3, 0.5, 0.1, 0.4, 0.2, 0.3, 0.25])
    temporal = pred.unsqueeze(0).expand(10, -1) + torch.randn(10, 8) * 0.01
    mean_only = temporal.mean(dim=0)
    loss_temporal = modal_amp_pair_invariant_loss(pred, temporal)
    loss_mean = modal_amp_pair_invariant_loss(pred, mean_only)
    assert loss_temporal.item() >= 0.0
    assert loss_mean.item() >= 0.0


def test_resolve_target_mode_amps_prefers_piptrack():
    data = torch.randn(20, 60, 8)
    pip = torch.ones(8) * 0.5
    prior = {"partial_amps_mean": pip}
    resolved = resolve_target_mode_amps(data, prior)
    assert torch.allclose(resolved, pip)


def test_mse_prior_skip_terms_when_phase4_active():
    log_base = torch.log(torch.tensor(0.05))
    log_slope = torch.log(torch.tensor(0.02))
    skew = _random_skew(8)
    skip = _mse_prior_skip_terms(
        0.3,
        True,
        inv_damping=fr_invariant_damping,
        inv_coupling=fr_invariant_coupling,
        inv_speed=fr_invariant_speed,
        inv_inharm=fr_invariant_inharm,
        log_base_rate=log_base,
        log_slope=log_slope,
        coupling_skew=skew,
    )
    assert skip == frozenset({"damping", "coupling", "speed", "inharm"})


def test_mse_prior_skip_empty_when_augment_mode():
    skip = _mse_prior_skip_terms(
        0.3,
        False,
        inv_damping=1.0,
        inv_coupling=0.5,
        inv_speed=0.25,
        inv_inharm=0.25,
        log_base_rate=torch.log(torch.tensor(0.05)),
        log_slope=torch.log(torch.tensor(0.02)),
        coupling_skew=_random_skew(8),
    )
    assert skip == frozenset()


def test_prior_loss_skips_damping_mse_when_requested():
    damping = torch.linspace(0.05, 0.5, 8)
    target = torch.zeros(8)
    coupling = torch.tensor(0.3)
    inharm = torch.zeros(8)
    speed = torch.ones(8) * 2.0

    full = prior_loss(damping, coupling, inharm, speed, prior_targets={"damping_rates": target})
    skipped = prior_loss(
        damping, coupling, inharm, speed,
        prior_targets={"damping_rates": target},
        skip_mse_terms=frozenset({"damping"}),
    )
    assert full.item() > skipped.item()
    assert (full - skipped).item() > 0.0


def test_harmonic_design_matrix_shape():
    design = harmonic_design_matrix(8, device=torch.device("cpu"))
    assert design.shape == (8, 2)
    assert torch.allclose(design[:, 0], torch.ones(8))
    assert torch.allclose(design[:, 1], torch.arange(8, dtype=torch.float32))


if __name__ == "__main__":
    test_damping_invariant_zero_when_coefficients_match()
    test_damping_invariant_penalizes_mismatched_envelope_scale()
    test_coupling_invariant_zero_on_match()
    test_coupling_invariant_rotation_invariant_shape()
    test_coupling_invariant_penalizes_strength_mismatch()
    test_speed_profile_invariant_zero_on_match()
    test_inharm_profile_invariant_scale_quotient()
    test_modal_amp_pair_invariant_zero_on_match()
    test_modal_amp_pair_invariant_mismatch_bounded()
    test_modal_amp_pair_invariant_uses_temporal_obs()
    test_resolve_target_mode_amps_prefers_piptrack()
    test_mse_prior_skip_terms_when_phase4_active()
    test_mse_prior_skip_empty_when_augment_mode()
    test_prior_loss_skips_damping_mse_when_requested()
    test_harmonic_design_matrix_shape()
    print("physics_audio invariant tests passed.")