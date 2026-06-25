"""Integration tests for Fisher-Rao extensions."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fisher_rao.metrics import fisher_rao_distance, sqrt_embed, sphere_to_simplex
from fisher_rao.invariants import location_scale_pair_invariant, fisher_rao_canonical_pair
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


def test_location_scale_pair_invariant_affine_action():
    """Proposition 3.3 (proof.pdf): invariant unchanged under common Aff(d) action."""
    mu1 = torch.tensor([1.0, 2.0])
    mu2 = torch.tensor([0.5, -1.0])
    V1 = torch.tensor([[2.0, 0.5], [0.5, 1.5]])
    V2 = torch.tensor([[1.0, 0.2], [0.2, 0.8]])
    A = torch.tensor([[0.6, -0.4], [0.4, 0.6]])
    b = torch.tensor([1.0, -0.5])

    inv1 = location_scale_pair_invariant(mu1, V1, mu2, V2)
    inv2 = location_scale_pair_invariant(A @ mu1 + b, A @ V1, A @ mu2 + b, A @ V2)

    assert torch.allclose(inv1.singular_values, inv2.singular_values, atol=1e-4)
    assert torch.allclose(inv1.block_norms, inv2.block_norms, atol=1e-4)


def test_fisher_rao_canonical_pair():
    """Proposition 4.2 (proof.pdf): canonical reduction to (nu, S) at identity."""
    nu, S = fisher_rao_canonical_pair(
        torch.zeros(2), torch.eye(2), torch.zeros(2), torch.eye(2)
    )
    assert torch.allclose(nu, torch.zeros(2), atol=1e-5)
    assert torch.allclose(S, torch.eye(2), atol=1e-5)


if __name__ == "__main__":
    test_fisher_rao_sphere_manifold()
    test_burst_optimizer_fisher_modulation_step()
    test_location_scale_pair_invariant_affine_action()
    test_fisher_rao_canonical_pair()
    print("Integration tests passed.")