# losses.py
# Updated: Fixed missing stiefel_dist import + restored full-precision autocast wrapper
# - Added from .utils import stiefel_dist
# - Wrapped geo_loss in autocast(enabled=False) for SVD safety
# - Kept gentle mean speed prior (with tunable lambda)

import torch.nn.functional as F
from torch.amp import autocast

from .config import (
    speed_uniform_lambda, coupling_prior_lambda, TRUE_COUPLING_STRENGTH,
    damping_prior_lambda, TRUE_DAMPING_RATES,
    inharm_l2_lambda, inharm_ceiling_lambda, inharm_ceiling_threshold,
    VELOCITY_SCALE_BASE
)
from .utils import stiefel_dist  # ← Critical import for geo_loss

# Tunable strength for mean speed anchor (start 1.0; 0.5–3.0 typical range)
speed_mean_prior_lambda = 1.0

def geo_loss(preds, data_points):
    # Force full precision + disable AMP to ensure SVD stability (no Half crash)
    with autocast('cuda', enabled=False):
        return stiefel_dist(preds.float(), data_points.float()).pow(2).mean()

def prior_loss(damping_rates, coupling_strength, inharm_b, speed_scalars):
    speed_uniform_loss = speed_uniform_lambda * (speed_scalars.std() / (speed_scalars.mean() + 1e-8))
    speed_mean_prior = speed_mean_prior_lambda * (speed_scalars.mean() - VELOCITY_SCALE_BASE).pow(2)
    coupling_prior_loss = coupling_prior_lambda * (coupling_strength - TRUE_COUPLING_STRENGTH).pow(2)
    damping_prior_loss = damping_prior_lambda * F.mse_loss(damping_rates, TRUE_DAMPING_RATES)
    inharm_l2_loss = inharm_l2_lambda * inharm_b.pow(2).mean()
    inharm_ceiling_loss = inharm_ceiling_lambda * F.relu(inharm_b - inharm_ceiling_threshold).pow(2).mean()
    return (speed_uniform_loss +
            speed_mean_prior +
            coupling_prior_loss +
            damping_prior_loss +
            inharm_l2_loss +
            inharm_ceiling_loss)

def total_loss(preds, data_points, damping_rates, coupling_strength, inharm_b, speed_scalars):
    return geo_loss(preds, data_points) + prior_loss(damping_rates, coupling_strength, inharm_b, speed_scalars)
