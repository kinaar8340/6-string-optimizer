# scripts/run_tuning_toolbox.py
# VQC Optimizer Tuning Toolbox
# Compares multiple guitar-tuning-inspired hierarchical presets on the compactified Rosenbrock S^3 benchmark
# Runs each preset for 15000 steps from the same random seed for fair comparison
# Outputs progress every 3000 steps and a final ranked summary table
# Optimizers run with verbose=False for clean output — only key metrics shown

import torch
import geoopt
from optimizer.burst_optimizer import GeooptBurstOptimizer, HierarchicalGeooptBurstOptimizer

# ----------------------------- Model & Loss Helpers -----------------------------
class SphereRosenbrockModel(torch.nn.Module):
    def __init__(self, num_instances=32):
        super().__init__()
        init = torch.randn(num_instances, 4, dtype=torch.float64)
        init[..., 0] = 0.95 + 0.05 * torch.randn(num_instances)  # Start near north pole
        init = init / init.norm(dim=-1, keepdim=True)
        self.manifold = geoopt.manifolds.Sphere()
        self.q = geoopt.ManifoldParameter(init, manifold=self.manifold)

    def forward(self):
        return self.q


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


# ----------------------------- Optimization Runner -----------------------------
def run_preset(config, max_steps=15000, seed=42, print_every=3000):
    torch.manual_seed(seed)

    model = SphereRosenbrockModel(num_instances=32)

    def closure():
        model.zero_grad()
        q = model()
        u = stereographic_projection(q)
        loss = rosenbrock_3d(u).mean()
        loss.backward()
        return loss

    # Base optimizer (no scheduled bursts — pure twist/hierarchy driven)
    base_opt = GeooptBurstOptimizer(
        model.parameters(),
        lr=config["lr"],
        burst_factor_max=config["base_burst_max"],
        burst_schedule_interval=0,  # Disable forced bursts
        verbose=False,
    )

    # Build hierarchical layers
    string_names = ["HighE", "B", "G", "D", "A", "LowE"]
    current_opt = base_opt
    for name, win, bb, tb in zip(
        string_names, config["windows"], config["burst_boosts"], config["theta_boosts"]
    ):
        current_opt = HierarchicalGeooptBurstOptimizer(
            current_opt,
            stagnation_window=win,
            stagnation_thresh=1e-3,
            burst_boost=bb,
            theta_boost=tb,
            name=name,
            verbose=False,
        )

    opt = current_opt

    # Run optimization
    min_loss = float("inf")
    history = []

    print(f"\n=== Starting {config['name']} ===\n")

    for step in range(max_steps):
        loss = opt.step(closure)
        loss_val = loss.item()
        min_loss = min(min_loss, loss_val)

        if step % print_every == 0 or step == max_steps - 1:
            with torch.no_grad():
                u_mean = stereographic_projection(model.q).mean(dim=0)
                print(
                    f"[{config['name']}] Step {step:5d} | Loss: {loss_val:.6f} | "
                    f"Mean u ≈ [{u_mean[0]:.3f}, {u_mean[1]:.3f}, {u_mean[2]:.3f}]"
                )
            history.append((step, loss_val, u_mean.tolist()))

    final_step, final_loss, final_u = history[-1]
    return {
        "name": config["name"],
        "final_loss": final_loss,
        "min_loss": min_loss,
        "final_u": final_u,
        "history": history,
    }


# ----------------------------- Guitar Tuning Presets -----------------------------
presets = [
    {
        "name": "Standard",
        "description": "Balanced, frequency-proportional windows for classic feel",
        "lr": 0.05,
        "base_burst_max": 15.0,
        "windows": [400, 535, 675, 900, 1200, 1600],  # ~1/frequency ratios
        "burst_boosts": [1.2, 1.4, 1.6, 1.8, 2.2, 2.8],
        "theta_boosts": [1.1, 1.15, 1.20, 1.25, 1.35, 1.50],
    },
    {
        "name": "Drop-D",
        "description": "Heavy low-end authority for deep escapes",
        "lr": 0.05,
        "base_burst_max": 15.0,
        "windows": [400, 535, 675, 900, 1200, 1800],  # Extended LowE window
        "burst_boosts": [1.1, 1.2, 1.3, 2.0, 3.0, 4.0],
        "theta_boosts": [1.1, 1.15, 1.20, 1.30, 1.40, 1.60],
    },
    {
        "name": "DADGAD",
        "description": "Modal/drone feel — mid strings (G/D/A) dominate resonance",
        "lr": 0.05,
        "base_burst_max": 15.0,
        "windows": [400, 550, 750, 1000, 1150, 1400],
        "burst_boosts": [1.3, 1.6, 2.3, 2.6, 2.3, 1.7],
        "theta_boosts": [1.1, 1.2, 1.4, 1.5, 1.4, 1.2],
    },
    {
        "name": "Open-C",
        "description": "Resonant low overtones — wide low windows, uniform power",
        "lr": 0.05,
        "base_burst_max": 18.0,
        "windows": [350, 450, 600, 900, 1300, 1900],
        "burst_boosts": [2.0, 2.2, 2.4, 2.6, 2.8, 3.0],
        "theta_boosts": [1.2, 1.25, 1.30, 1.35, 1.40, 1.45],
    },
    {
        "name": "Eb Tuning",
        "description": "Down-tuned — slower learning, more aggressive bursts overall",
        "lr": 0.04,
        "base_burst_max": 20.0,
        "windows": [450, 600, 750, 1000, 1350, 1800],  # Scaled-up (~12% larger)
        "burst_boosts": [1.3, 1.5, 1.7, 2.0, 2.5, 3.2],
        "theta_boosts": [1.15, 1.20, 1.25, 1.30, 1.40, 1.55],
    },
]

# ----------------------------- Run Toolbox -----------------------------
if __name__ == "__main__":
    results = []
    for config in presets:
        result = run_preset(config)
        results.append(result)

    # Summary table (ranked by final loss)
    print("\n" + "="*80)
    print("VQC OPTIMIZER TUNING TOOLBOX — FINAL RESULTS (15000 steps)")
    print("="*80)
    results.sort(key=lambda x: x["final_loss"])

    print(f"{'Rank':4} {'Preset':18} {'Final Loss':12} {'Min Loss':12} {'Final u_x':>10} {'u_y':>10} {'u_z':>10}")
    print("-" * 80)
    for i, r in enumerate(results):
        ux, uy, uz = r["final_u"]
        print(
            f"{i+1:<4} {r['name']:18} {r['final_loss']:12.6f} {r['min_loss']:12.6f} "
            f"{ux:10.3f} {uy:10.3f} {uz:10.3f}"
        )

    best = results[0]
    print("\n🎸 Best performing tuning:", best["name"])
    print(f"   Final loss: {best['final_loss']:.6f} | Closest to global minimum [1,1,1]")
    print("\nRun the best preset individually (copy its config into examples/rosenbrock_s3.py) for longer training/polish.")
    print("For ensemble ideas: average parameters from top-3 runs or switch tunings mid-optimization.")