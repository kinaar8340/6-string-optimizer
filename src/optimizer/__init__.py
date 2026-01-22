# src/optimizer/__init__.py

"""
6-string-optimizer - A highly adaptive Riemannian optimizer inspired by punctuated dynamics.

Key exports:
- Optimizers: GeooptBurstOptimizer, HierarchicalGeooptBurstOptimizer
- Model: SphereRosenbrockModel (benchmark on compactified Rosenbrock)
- Loss: rosenbrock_3d
- Utilities: stereographic_projection, device helpers, seeding
"""

from .burst_optimizer import GeooptBurstOptimizer, HierarchicalGeooptBurstOptimizer
from .models import SphereRosenbrockModel
from .losses import rosenbrock_3d
from .utils import stereographic_projection, get_device, to_device, set_seed

# Optional: package metadata
__version__ = "0.1.0"
__author__ = "X: @kinaar8340"

# Convenience: re-export commonly used geoopt components (optional, avoids extra imports)
# from geoopt import ManifoldParameter, Sphere  # Uncomment if users frequently need these
