# src/optimizer/models.py

import torch
import torch.nn as nn
import geoopt

from .utils import to_device  # Optional: for easy device placement in scripts


class SphereRosenbrockModel(nn.Module):
    """
    A simple batched model representing points on the 3-sphere (S^3) parameterized as unit quaternions.
    
    Used primarily for testing optimizers on the stereographically compactified 3D Rosenbrock function,
    which creates a challenging landscape with narrow valleys and pole singularities.
    
    Attributes:
        q: ManifoldParameter on the Sphere manifold (shape: [num_instances, 4])
    """
    def __init__(self, num_instances: int = 32, device: torch.device | None = None):
        super().__init__()
        
        # Initialize near the north pole (challenging starting region)
        init = torch.randn(num_instances, 4, dtype=torch.float64)
        init[..., 0] = 0.95 + 0.05 * torch.randn(num_instances)  # w component biased high
        init = init / init.norm(dim=-1, keepdim=True)  # Project to unit sphere
        
        self.manifold = geoopt.manifolds.Sphere()
        self.q = geoopt.ManifoldParameter(init, manifold=self.manifold)
        
        # Optional: move to device immediately if specified
        if device is not None:
            self.to(device)

    def forward(self) -> torch.Tensor:
        """
        Forward pass: simply return the quaternion parameters on the sphere.
        
        Returns:
            q: Tensor of shape [num_instances, 4]
        """
        return self.q


# Future-proof placeholders for additional benchmark models
# ------------------------------------------------------------------
# class StiefelOrthogonalModel(nn.Module):
#     """Example: Model with parameters on the Stiefel manifold (orthogonal frames)."""
#     ...
#
# class PoincareBallModel(nn.Module):
#     """Example: Hyperbolic embedding model on the Poincaré ball."""
#     ...
# ------------------------------------------------------------------
