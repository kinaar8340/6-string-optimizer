# src/optimizer/losses.py

import torch

def rosenbrock_3d(u: torch.Tensor) -> torch.Tensor:
    """
    3D Rosenbrock function, commonly used as a challenging non-convex benchmark.
    
    The global minimum is at (x, y, z) = (1, 1, 1) with value 0.
    
    When composed with stereographic projection from S^3 → R^3, it creates a compactified
    landscape with narrow valleys and pole singularities — ideal for testing manifold optimizers.
    
    Args:
        u: Tensor of shape (... , 3) representing points in R^3
    
    Returns:
        loss: Tensor of shape (... ,) with the Rosenbrock values
    """
    x, y, z = u[..., 0], u[..., 1], u[..., 2]
    return 100.0 * (y - x**2)**2 + 100.0 * (z - y**2)**2 + (1.0 - x)**2


# Future-proof placeholders for additional benchmark losses
# ------------------------------------------------------------------
# def brockett_function(...):
#     """Brockett function on the Stiefel manifold — another classic Riemannian test."""
#     ...
#
# def hyperbolic_embedding_loss(...):
#     """Example loss for tree-like data in the Poincaré ball."""
#     ...
#
# def sphere_direction_statistics_loss(...):
#     """Von Mises-Fisher or other directional statistics objectives."""
#     ...
# ------------------------------------------------------------------
