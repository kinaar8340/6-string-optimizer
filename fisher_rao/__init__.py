"""Fisher-Rao geometry extensions for the 6-string optimizer framework."""

from .metrics import (
    fisher_rao_distance,
    sqrt_embed,
    simplex_to_sphere,
    sphere_to_simplex,
    softmax_to_fr_sphere,
)
from .fisher_info import (
    score_norm_from_grads,
    score_norm_proxy,
    fisher_info_trace_hvp,
    fisher_modulation_factors,
)
from .invariants import (
    scale_quotient,
    rotation_singular_invariants,
    location_scale_invariants,
    project_to_invariants,
)
from .losses import (
    fisher_rao_loss,
    bhattacharyya_coefficient,
    spectral_fr_loss,
    modal_distribution_fr_loss,
)
from .burst_modulation import FisherInfoBurstModulator

__version__ = "0.1.0"

__all__ = [
    "fisher_rao_distance",
    "sqrt_embed",
    "simplex_to_sphere",
    "sphere_to_simplex",
    "softmax_to_fr_sphere",
    "score_norm_from_grads",
    "score_norm_proxy",
    "fisher_info_trace_hvp",
    "fisher_modulation_factors",
    "scale_quotient",
    "rotation_singular_invariants",
    "location_scale_invariants",
    "project_to_invariants",
    "fisher_rao_loss",
    "bhattacharyya_coefficient",
    "spectral_fr_loss",
    "modal_distribution_fr_loss",
    "FisherInfoBurstModulator",
]