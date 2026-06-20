# model.py
# Updated: Modern torch.amp API (device_type='cuda') for all autocast blocks
# - Replaced torch.cuda.amp.autocast → torch.amp.autocast('cuda', ...)

import torch
import torch.nn as nn
from geoopt import ManifoldParameter
from geoopt.manifolds import Stiefel
from torch.amp import autocast  # Add this import if not present

from .config import device, IDEAL_HARMONICS
from .utils import manifold

class StiefelDampedCoupledInharmGR(nn.Module):
    def __init__(self, dim: int, k_modes: int, initial_basis: torch.Tensor):
        super().__init__()
        self.dim = dim
        self.k_modes = k_modes

        self.base = ManifoldParameter(initial_basis, manifold=manifold)

        self.vel_dir_raw = nn.Parameter(torch.randn(dim, k_modes, device=device))
        self.log_speed = nn.Parameter(torch.log(torch.ones(k_modes, device=device) * 2.0))

        self.log_base_rate = nn.Parameter(torch.log(torch.tensor(0.06, device=device)))
        self.log_slope = nn.Parameter(torch.log(torch.tensor(0.064, device=device)))

        self.raw_lin_b = nn.Parameter(torch.tensor(-16.0, device=device))
        self.raw_quad_b = nn.Parameter(torch.tensor(-14.0, device=device))

        self.raw_coupling_strength = nn.Parameter(torch.log(torch.tensor(0.30, device=device)))
        self.coupling_raw = nn.Parameter(torch.randn(k_modes, k_modes, device=device) * 0.03)

    def forward(self, t):
        base = self.base
        dim, k_modes = self.dim, self.k_modes

        # Force full precision for manifold projection/norm
        with torch.amp.autocast('cuda', enabled=False):
            vel_dir = manifold.proju(base, self.vel_dir_raw)
            vel_dir = vel_dir / (vel_dir.norm(dim=0, keepdim=True) + 1e-8)

        speed_scalars = torch.exp(self.log_speed)
        damping_rates = torch.exp(self.log_base_rate) + torch.exp(self.log_slope) * (IDEAL_HARMONICS - 1)
        inharm_b = torch.exp(self.raw_lin_b) * IDEAL_HARMONICS + torch.exp(self.raw_quad_b) * IDEAL_HARMONICS.pow(2)

        freq = IDEAL_HARMONICS * torch.sqrt(1 + inharm_b * IDEAL_HARMONICS.pow(2))
        full_freq = speed_scalars * freq

        vel = vel_dir * speed_scalars.unsqueeze(0) * freq.unsqueeze(0)

        coupling_strength = torch.exp(self.raw_coupling_strength)
        coupling_skew = self.coupling_raw.tril(diagonal=-1) - self.coupling_raw.triu(diagonal=1)

        # Force full precision for coupling projection
        with torch.amp.autocast('cuda', enabled=False):
            coupling_vel = manifold.proju(base, base @ coupling_skew)

        vel_total = vel + coupling_strength * coupling_vel

        abs_t = torch.abs(t).view(-1, 1)
        envelope = torch.exp(-damping_rates * abs_t)

        vel_batch = t.view(-1, 1, 1) * vel_total.unsqueeze(0) * envelope.unsqueeze(1)
        base_batch = base.unsqueeze(0).expand(t.shape[0], dim, k_modes)

        # Force full precision for expmap (most sensitive manifold op)
        with torch.amp.autocast('cuda', enabled=False):
            preds = manifold.expmap(base_batch, vel_batch)

        return preds, damping_rates, coupling_strength, inharm_b, speed_scalars, full_freq