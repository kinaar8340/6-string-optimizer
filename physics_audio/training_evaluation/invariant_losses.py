"""Double-coset invariant losses for physics_audio (Nielsen & Okamura 2026)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
_FISHER_RAO_HOME = Path.home() / "Projects" / "Fisher_Rao"

for candidate in (_ROOT, _FISHER_RAO_HOME):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    from fisher_rao.invariants import (
        location_scale_pair_invariant,
        rotation_singular_invariants,
        fisher_rao_canonical_pair,
    )
except ImportError:

    def location_scale_pair_invariant(mu1, V1, mu2, V2):
        mu1 = mu1.float().flatten()
        mu2 = mu2.float().flatten()
        V1 = V1.float()
        V2 = V2.float()
        nu = torch.linalg.solve(V2, mu1 - mu2)
        S = torch.linalg.solve(V2, V1)
        singular_values = torch.linalg.svdvals(S)
        block_norms = nu.norm().unsqueeze(0)
        from collections import namedtuple

        Inv = namedtuple(
            "LocationScaleMaximalInvariant",
            ["singular_values", "multiplicities", "block_norms", "relative_scale", "whitened_location"],
        )
        return Inv(singular_values, [singular_values.numel()], block_norms, S, nu)

    def rotation_singular_invariants(matrix: torch.Tensor) -> torch.Tensor:
        return torch.linalg.svdvals(matrix.float())

    def fisher_rao_canonical_pair(mu1, V1, mu2, V2):
        inv = location_scale_pair_invariant(mu1, V1, mu2, V2)
        return inv.whitened_location, inv.relative_scale


def _collapse_modal_amps(amps: torch.Tensor) -> torch.Tensor:
    """Reduce (K,) or (T, K) amplitudes to a single per-mode profile (K,)."""
    amps = amps.float()
    if amps.dim() == 2:
        return amps.mean(dim=0)
    return amps.flatten()


def modal_amp_pair_invariant_loss(
    pred_amps: torch.Tensor,
    obs_amps: torch.Tensor,
    singular_weight: float = 1.0,
) -> torch.Tensor:
    """
    Prop 3.3 / 4.2 pair invariant on log-modal amplitude profiles.

    Bundles per-mode log amplitudes as location mu and diagonal scale V
    (temporal variance from piptrack when obs is (T, K)).  Invariant to
    positive scaling of the observed amplitude vector.
    """
    pred = _collapse_modal_amps(pred_amps)
    obs = obs_amps.float()
    k = pred.numel()
    device, dtype = pred.device, pred.dtype

    log_pred = torch.log(pred.clamp(min=1e-8))
    mu1 = log_pred
    var1 = torch.full((k,), log_pred.var().item() + 1e-4, device=device, dtype=dtype)

    if obs.dim() == 2:
        log_obs = torch.log(obs.clamp(min=1e-8))
        mu2 = log_obs.mean(dim=0)
        var2 = log_obs.var(dim=0) + 1e-4
    else:
        obs_k = _collapse_modal_amps(obs)
        log_obs = torch.log(obs_k.clamp(min=1e-8))
        mu2 = log_obs
        var2 = torch.full((k,), log_obs.var().item() + 1e-4, device=device, dtype=dtype)

    v1 = torch.diag(var1)
    v2 = torch.diag(var2)
    nu, s = fisher_rao_canonical_pair(mu1, v1, mu2, v2)

    loss = nu.pow(2).sum()
    if singular_weight > 0.0:
        loss = loss + singular_weight * (torch.linalg.svdvals(s) - 1.0).pow(2).sum()
    return loss


def harmonic_design_matrix(
    k_modes: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Columns [1, k-1] for log-linear damping envelope d_k = base + slope * (k-1)."""
    harmonics = torch.arange(1, k_modes + 1, device=device, dtype=dtype)
    return torch.stack([torch.ones_like(harmonics), harmonics - 1.0], dim=1)


def target_damping_coefficients(
    target_damping: torch.Tensor,
    design: torch.Tensor | None = None,
) -> torch.Tensor:
    """Least-squares [base, slope] coefficients for a K-mode damping target."""
    target = target_damping.float().flatten()
    if design is None:
        design = harmonic_design_matrix(target.numel(), target.device, target.dtype)
    solution = torch.linalg.lstsq(design, target.unsqueeze(-1)).solution
    return solution.squeeze(-1)


def learned_damping_coefficients(
    log_base_rate: torch.Tensor,
    log_slope: torch.Tensor,
) -> torch.Tensor:
    """Model damping coefficients in the same [base, slope] basis."""
    base = torch.exp(log_base_rate.float().reshape(()))
    slope = torch.exp(log_slope.float().reshape(()))
    return torch.stack([base, slope])


def damping_invariant_loss(
    log_base_rate: torch.Tensor,
    log_slope: torch.Tensor,
    target_damping: torch.Tensor,
    singular_weight: float = 0.0,
) -> torch.Tensor:
    """
    Prop 3.3 invariant loss on the 2D damping sufficient statistic.

    Compares learned (base, slope) to the LS fit of ``target_damping`` in
    coefficient space with V1 = V2 = I_2.  Block norms reduce to the
    whitened coefficient residual; invariant to global reparameterization
    of the shared harmonic design subspace.
    """
    target = target_damping.float().flatten()
    k_modes = target.numel()
    design = harmonic_design_matrix(k_modes, target.device, target.dtype)

    mu1 = learned_damping_coefficients(log_base_rate, log_slope)
    mu2 = target_damping_coefficients(target, design)
    eye = torch.eye(2, device=target.device, dtype=target.dtype)

    inv = location_scale_pair_invariant(mu1, eye, mu2, eye)

    loss = inv.block_norms.pow(2).sum()
    if singular_weight > 0.0:
        loss = loss + singular_weight * (inv.singular_values - 1.0).pow(2).sum()
    return loss


def _skew_matrix(raw: torch.Tensor) -> torch.Tensor:
    """Build skew-symmetric matrix from unconstrained raw parameters."""
    return raw.tril(diagonal=-1) - raw.triu(diagonal=1)


def coupling_invariant_loss(
    coupling_skew: torch.Tensor,
    coupling_strength: torch.Tensor,
    target_coupling_skew: torch.Tensor | None = None,
    target_coupling_strength: float | torch.Tensor | None = None,
    strength_weight: float = 0.1,
) -> torch.Tensor:
    """
    O(k) rotation-invariant loss on coupling topology (Thm 2.3 / skew spectrum).

    Compares singular values of the learned skew coupling matrix to a reference
    (zero = decoupled, or estimated / synthetic ground truth).  Coupling strength
    is compared separately in log-space (positive scale quotient).
    """
    learned = coupling_skew.float()
    if target_coupling_skew is None:
        target = torch.zeros_like(learned)
    else:
        target = target_coupling_skew.float()

    sv_learned = rotation_singular_invariants(learned)
    sv_target = rotation_singular_invariants(target)
    shape_loss = (sv_learned - sv_target).pow(2).sum()

    if target_coupling_strength is None:
        target_s = torch.tensor(1.0, device=learned.device, dtype=learned.dtype)
    elif torch.is_tensor(target_coupling_strength):
        target_s = target_coupling_strength.float().reshape(())
    else:
        target_s = torch.tensor(float(target_coupling_strength), device=learned.device, dtype=learned.dtype)

    strength = coupling_strength.float().reshape(())
    scale_loss = torch.log(strength / target_s.clamp(min=1e-8)).pow(2)

    return shape_loss + strength_weight * scale_loss


def _profile_fr_loss(learned: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Fisher-Rao distance between scale-quotient normalized 1D profiles."""
    from fisher_rao.losses import fisher_rao_loss
    from fisher_rao.invariants import scale_quotient

    p = scale_quotient(learned.float().clamp(min=1e-12).unsqueeze(0), mode="sum")
    q = scale_quotient(target.float().clamp(min=1e-12).unsqueeze(0), mode="sum")
    return fisher_rao_loss(p, q)


def speed_profile_invariant_loss(
    speed_scalars: torch.Tensor,
    target_speed_scalars: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Scale-invariant Fisher-Rao loss on per-mode speed distribution.

    Quotients global scale (uniform stretch symmetry) before comparing profiles.
    """
    target = target_speed_scalars if target_speed_scalars is not None else speed_scalars.detach()
    return _profile_fr_loss(speed_scalars, target)


def inharm_profile_invariant_loss(
    inharm_b: torch.Tensor,
    target_inharm_b: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Scale-invariant Fisher-Rao loss on inharmonicity coefficient profile.

    Targets near-zero inharmonicity collapse to a peaked distribution; FR
    geometry handles the comparison without arbitrary L2 scale.
    """
    target = target_inharm_b if target_inharm_b is not None else inharm_b.detach()
    return _profile_fr_loss(inharm_b, target)