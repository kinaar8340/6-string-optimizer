# src/optimizer/utils.py

import torch

def stereographic_projection(q: torch.Tensor) -> torch.Tensor:
    """
    Project from S^3 (unit quaternions) to R^3 via stereographic projection
    (north pole mapped to infinity).

    Args:
        q: Tensor of shape (... , 4) representing unit quaternions (w, x, y, z)

    Returns:
        u: Tensor of shape (... , 3) in Euclidean space
    """
    w = q[..., 0:1]
    v = q[..., 1:]
    denom = 1.0 - w
    mask_pole = denom.abs() < 1e-6
    u = v / denom.clamp(min=1e-6)
    # Handle near-pole cases gracefully (push to large values)
    u = torch.where(mask_pole, torch.sign(v) * 1e5, u)
    return u


def get_device() -> torch.device:
    """
    Return the best available device (CUDA if available, else CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(obj: torch.nn.Module | torch.Tensor, device: torch.device | None = None):
    """
    Move a model or tensor to the specified device (or auto-detected best device).

    Args:
        obj: nn.Module or Tensor to move
        device: Optional explicit device; if None, uses get_device()

    Returns:
        The object moved to the target device
    """
    if device is None:
        device = get_device()
    return obj.to(device)


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across torch, numpy, etc.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Optional: torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# Future-proof placeholders (uncomment/add as needed)
# import matplotlib.pyplot as plt
#
# def plot_sphere_points(q: torch.Tensor, u: torch.Tensor | None = None, title: str = "Points on S^3 -> R^3"):
#     """
#     Simple 3D scatter of stereographically projected points (for debugging/viz).
#     """
#     if u is None:
#         u = stereographic_projection(q)
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(u[..., 0], u[..., 1], u[..., 2])
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title(title)
#     plt.show()
