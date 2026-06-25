"""Integration tests for Fisher-Rao extensions."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FISHER_ROOT = Path.home() / "Projects" / "Fisher_Rao"
for p in (ROOT / "src", FISHER_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fisher_rao.metrics import fisher_rao_distance, sqrt_embed, sphere_to_simplex
from fisher_rao.burst_modulation import FisherInfoBurstModulator
from optimizer.manifolds import FisherRaoSphere
from optimizer.burst_optimizer import GeooptBurstOptimizer
from optimizer.models import SphereRosenbrockModel
from optimizer.utils import stereographic_projection
from optimizer.losses import rosenbrock_3d


def test_fisher_rao_sphere_manifold():
    m = FisherRaoSphere()
    p = torch.tensor([[0.2, 0.3, 0.5]])
    s = m.simplex_to_manifold(p)
    assert torch.allclose(s.norm(), torch.tensor(1.0), atol=1e-5)
    d = m.fr_distance_on_simplex(p, p)
    assert d.item() < 1e-5


def test_burst_optimizer_fisher_modulation_step():
    torch.manual_seed(0)
    model = SphereRosenbrockModel(num_instances=2)
    modulator = FisherInfoBurstModulator(info_scale=10.0)
    opt = GeooptBurstOptimizer(
        model.parameters(),
        lr=0.01,
        verbose=False,
        use_fisher_modulation=True,
        fisher_modulator=modulator,
        warm_up_steps=0,
    )

    def closure():
        opt.zero_grad()
        u = stereographic_projection(model())
        loss = rosenbrock_3d(u).mean()
        loss.backward()
        return loss

    loss = opt.step(closure)
    assert loss is not None
    assert loss.item() >= 0


if __name__ == "__main__":
    test_fisher_rao_sphere_manifold()
    test_burst_optimizer_fisher_modulation_step()
    print("Integration tests passed.")