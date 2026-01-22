# scripts/ensemble_top3_eb.py (updated for deeper start + longer polish)
# Average top-3 at 15k → spherical mean → continue 35k with Eb (total effective 50k)
# Spherical mean (normalize after averaging quaternions) for robust starting point in basin

import torch
import geoopt
from optimizer import GeooptBurstOptimizer, HierarchicalGeooptBurstOptimizer

torch.manual_seed(42)

# Shared model/loss (same as before)
class SphereRosenbrockModel(torch.nn.Module):
    def __init__(self, num_instances=32):
        super().__init__()
        init = torch.randn(num_instances, 4, dtype=torch.float64)
        init[..., 0] = 0.95 + 0.05 * torch.randn(num_instances)
        init = init / init.norm(dim=-1, keepdim=True)
        self.manifold = geoopt.manifolds.Sphere()
        self.q = geoopt.ManifoldParameter(init, manifold=self.manifold)
    def forward(self): return self.q

def stereographic_projection(q):
    w = q[..., 0:1]
    v = q[..., 1:]
    denom = 1.0 - w
    mask_pole = denom.abs() < 1e-6
    u = v / denom.clamp(min=1e-6)
    u = torch.where(mask_pole, torch.sign(v) * 1e5, u)
    return u

def rosenbrock_3d(u):
    x, y, z = u[..., 0], u[..., 1], u[..., 2]
    return 100.0 * (y - x ** 2) ** 2 + 100.0 * (z - y ** 2) ** 2 + (1.0 - x) ** 2

def run_to_15000(config, verbose=False):
    model = SphereRosenbrockModel(num_instances=32)
    def closure():
        model.zero_grad()
        q = model()
        u = stereographic_projection(q)
        loss = rosenbrock_3d(u).mean()
        loss.backward()
        return loss

    base_opt = GeooptBurstOptimizer(model.parameters(), lr=config["lr"], burst_factor_max=config["base_burst_max"],
                                    burst_schedule_interval=0, verbose=verbose)
    current_opt = base_opt
    for name, win, bb, tb in zip(["HighE","B","G","D","A","LowE"], config["windows"], config["burst_boosts"], config["theta_boosts"]):
        current_opt = HierarchicalGeooptBurstOptimizer(current_opt, stagnation_window=win, stagnation_thresh=1e-3,
                                                       burst_boost=bb, theta_boost=tb, name=name, verbose=verbose)
    for _ in range(15000):
        current_opt.step(closure)
    return model.q.detach().clone()  # Return final quaternion parameters

# Top-3 presets from toolbox results
presets_top3 = [
    {"name": "Eb Tuning", "lr": 0.04, "base_burst_max": 20.0,
     "windows": [450, 600, 750, 1000, 1350, 1800],
     "burst_boosts": [1.3, 1.5, 1.7, 2.0, 2.5, 3.2],
     "theta_boosts": [1.15, 1.20, 1.25, 1.30, 1.40, 1.55]},
    {"name": "DADGAD", "lr": 0.05, "base_burst_max": 15.0,
     "windows": [400, 550, 750, 1000, 1150, 1400],
     "burst_boosts": [1.3, 1.6, 2.3, 2.6, 2.3, 1.7],
     "theta_boosts": [1.1, 1.2, 1.4, 1.5, 1.4, 1.2]},
    {"name": "Open-C", "lr": 0.05, "base_burst_max": 18.0,
     "windows": [350, 450, 600, 900, 1300, 1900],
     "burst_boosts": [2.0, 2.2, 2.4, 2.6, 2.8, 3.0],
     "theta_boosts": [1.2, 1.25, 1.30, 1.35, 1.40, 1.45]},
]

print("Running top-3 tunings to 15000 steps for ensemble averaging...")
q_states = []
for config in presets_top3:
    print(f"   Running {config['name']}...")
    q = run_to_15000(config, verbose=False)
    q_states.append(q)
    with torch.no_grad():
        u_mean = stereographic_projection(q).mean(dim=0)
        loss = rosenbrock_3d(stereographic_projection(q)).mean().item()
        print(f"   → {config['name']} @15k: Loss {loss:.2f} | u ≈ [{u_mean[0]:.3f}, {u_mean[1]:.3f}, {u_mean[2]:.3f}]")

# Spherical ensemble mean
ensemble_q = torch.mean(torch.stack(q_states), dim=0)
ensemble_q = ensemble_q / ensemble_q.norm(dim=-1, keepdim=True)

# Continue with pure Eb Tuning from ensemble start (longer polish)
model = SphereRosenbrockModel(num_instances=32)
model.q.data = ensemble_q  # Inject ensemble parameters

def closure():
    model.zero_grad()
    q = model()
    u = stereographic_projection(q)
    loss = rosenbrock_3d(u).mean()
    loss.backward()
    return loss

print("\nEnsemble mean computed — starting longer Eb Tuning polish (35k steps) from averaged basin point")

# Eb config with increased burst capacity for deeper exploration
base_opt = GeooptBurstOptimizer(model.parameters(), lr=0.04, burst_factor_max=25.0,
                                burst_schedule_interval=0, verbose=True)
current_opt = base_opt
for name, win, bb, tb in zip(["HighE","B","G","D","A","LowE"], presets_top3[0]["windows"],
                             presets_top3[0]["burst_boosts"], presets_top3[0]["theta_boosts"]):
    current_opt = HierarchicalGeooptBurstOptimizer(current_opt, stagnation_window=win, stagnation_thresh=1e-3,
                                                   burst_boost=bb, theta_boost=tb, name=name, verbose=True)

max_steps = 35000  # total effective ~50k steps
print_every = 1000
best_loss = float("inf")

for step in range(max_steps):
    loss = current_opt.step(closure)
    loss_val = loss.item()
    if loss_val < best_loss:
        best_loss = loss_val

    if step % print_every == 0 or step == max_steps - 1:
        with torch.no_grad():
            u_mean = stereographic_projection(model.q).mean(dim=0)
            print(
                f"[Ensemble → Eb] Step {step:5d} | Loss: {loss_val:.6f} (best: {best_loss:.6f}) | "
                f"Mean u ≈ [{u_mean[0]:.3f}, {u_mean[1]:.3f}, {u_mean[2]:.3f}]"
            )

print("\nEnsemble + extended Eb polish complete. This deeper hybrid run should push toward even lower loss minima.")