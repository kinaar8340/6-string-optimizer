#!/usr/bin/env python3
"""
A/B benchmark: GeooptBurstOptimizer with vs without Fisher-info modulation.

Same seed, identical quaternion init, Eb Master hierarchy. Logs loss curves,
burst counts, and time-to-threshold metrics to CSV + stdout summary.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import geoopt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from fisher_rao.burst_modulation import FisherInfoBurstModulator
from optimizer import GeooptBurstOptimizer, HierarchicalGeooptBurstOptimizer
from optimizer.losses import rosenbrock_3d
from optimizer.utils import stereographic_projection, set_seed

STRING_NAMES = ["HighE", "B", "G", "D", "A", "LowE"]
WINDOWS = [400, 580, 780, 1100, 1600, 2400]
BURST_BOOSTS = [1.02, 1.04, 1.06, 1.08, 1.10, 1.15]
THETA_BOOSTS = [1.01, 1.02, 1.03, 1.04, 1.05, 1.06]


class SphereRosenbrockModel(torch.nn.Module):
    def __init__(self, init_q: torch.Tensor):
        super().__init__()
        self.manifold = geoopt.manifolds.Sphere()
        self.q = geoopt.ManifoldParameter(init_q.clone(), manifold=self.manifold)

    def forward(self):
        return self.q


def make_init(seed: int, num_instances: int = 1) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    init = torch.randn(num_instances, 4, dtype=torch.float64, generator=gen)
    init[..., 0] = 0.95 + 0.05 * torch.randn(num_instances, generator=gen)
    return init / init.norm(dim=-1, keepdim=True)


def build_hierarchy(
    params,
    use_fisher: bool,
    fisher_info_scale: float,
    verbose: bool,
) -> HierarchicalGeooptBurstOptimizer:
    modulator = FisherInfoBurstModulator(info_scale=fisher_info_scale) if use_fisher else None
    base = GeooptBurstOptimizer(
        params,
        lr=0.03,
        burst_factor_max=8.0,
        damping_min=0.98,
        damping=0.995,
        max_theta=torch.pi / 20,
        burst_schedule_interval=0,
        verbose=verbose,
        use_fisher_modulation=use_fisher,
        fisher_modulator=modulator,
        fisher_info_scale=fisher_info_scale,
    )
    current = base
    for name, win, bb, tb in zip(STRING_NAMES, WINDOWS, BURST_BOOSTS, THETA_BOOSTS):
        current = HierarchicalGeooptBurstOptimizer(
            current,
            stagnation_window=win,
            stagnation_thresh=5e-5,
            burst_boost=bb,
            theta_boost=tb,
            name=name,
            verbose=verbose,
        )
    return current


def base_optimizer(hierarchical_opt) -> GeooptBurstOptimizer:
    opt = hierarchical_opt
    while hasattr(opt, "optimizer"):
        opt = opt.optimizer
    return opt


def run_trial(
    label: str,
    init_q: torch.Tensor,
    max_steps: int,
    use_fisher: bool,
    fisher_info_scale: float,
    log_every: int,
    verbose: bool,
) -> dict:
    model = SphereRosenbrockModel(init_q)
    opt = build_hierarchy(model.parameters(), use_fisher, fisher_info_scale, verbose=verbose)
    base = base_optimizer(opt)

    history: list[tuple[int, float, float]] = []
    best_loss = float("inf")
    t0 = time.perf_counter()

    for step in range(max_steps):

        def closure():
            opt.zero_grad()
            u = stereographic_projection(model())
            loss = rosenbrock_3d(u).mean()
            loss.backward()
            return loss

        loss = opt.step(closure)
        loss_val = loss.item()
        best_loss = min(best_loss, loss_val)

        if step % log_every == 0 or step == max_steps - 1:
            history.append((step, loss_val, best_loss))

    elapsed = time.perf_counter() - t0
    with torch.no_grad():
        u = stereographic_projection(model.q).squeeze(0)

    fisher_info = None
    if use_fisher and base.fisher_modulator is not None:
        fisher_info = base.fisher_modulator.smoothed_info

    return {
        "label": label,
        "use_fisher": use_fisher,
        "final_loss": loss_val,
        "best_loss": best_loss,
        "burst_count": base.burst_count,
        "steps": max_steps,
        "elapsed_s": elapsed,
        "final_u": u.tolist(),
        "history": history,
        "smoothed_fisher_info": fisher_info,
    }


def first_step_below(history: list[tuple[int, float, float]], threshold: float) -> int | None:
    for step, loss, _ in history:
        if loss < threshold:
            return step
    return None


def write_csv(path: Path, baseline: dict, fisher: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    b_map = {s: (l, bl) for s, l, bl in baseline["history"]}
    f_map = {s: (l, bl) for s, l, bl in fisher["history"]}
    steps = sorted(set(b_map) | set(f_map))
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "step", "loss_baseline", "best_baseline", "loss_fisher", "best_fisher",
        ])
        for s in steps:
            bl, bbest = b_map.get(s, ("", ""))
            fl, fbest = f_map.get(s, ("", ""))
            w.writerow([s, bl, bbest, fl, fbest])


def print_summary(baseline: dict, fisher: dict, csv_path: Path) -> None:
    print("\n" + "=" * 72)
    print("Fisher modulation A/B — Compactified Rosenbrock on S³ (Eb Master hierarchy)")
    print("=" * 72)
    for r in (baseline, fisher):
        print(
            f"\n{r['label']:>12} | final={r['final_loss']:.6f} best={r['best_loss']:.6f} "
            f"| bursts={r['burst_count']} | {r['elapsed_s']:.1f}s"
        )
        u = r["final_u"]
        print(f"             | u = [{u[0]:.4f}, {u[1]:.4f}, {u[2]:.4f}]")
        if r["smoothed_fisher_info"] is not None:
            print(f"             | smoothed Fisher proxy = {r['smoothed_fisher_info']:.4e}")

    print("\n--- Time to loss thresholds (logged steps) ---")
    for thresh in (100.0, 10.0, 1.0, 0.5, 0.2):
        bs = first_step_below(baseline["history"], thresh)
        fs = first_step_below(fisher["history"], thresh)
        b_str = str(bs) if bs is not None else "never"
        f_str = str(fs) if fs is not None else "never"
        winner = ""
        if bs is not None and fs is not None:
            winner = "fisher" if fs < bs else ("baseline" if bs < fs else "tie")
        print(f"  loss < {thresh:5.1f}: baseline step {b_str:>6} | fisher step {f_str:>6}  {winner}")

    delta_best = fisher["best_loss"] - baseline["best_loss"]
    delta_bursts = fisher["burst_count"] - baseline["burst_count"]
    print(f"\nΔ best_loss (fisher - baseline): {delta_best:+.6f}")
    print(f"Δ burst_count (fisher - baseline): {delta_bursts:+d}")
    if fisher["smoothed_fisher_info"] is not None and fisher["smoothed_fisher_info"] > 1e6:
        print(
            "\nNote: Fisher proxy is very large — try raising --fisher-info-scale "
            "if modulation factors are saturated (conservative bursts throughout)."
        )
    print(f"CSV → {csv_path}")


def main():
    p = argparse.ArgumentParser(description="A/B Fisher modulation on Rosenbrock S³")
    p.add_argument("--steps", type=int, default=8000, help="Optimization steps per arm")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--fisher-info-scale", type=float, default=50.0)
    p.add_argument("--output", type=Path, default=ROOT / "outputs" / "compare_fr_modulation.csv")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    init_q = make_init(args.seed)

    print(f"Running A/B benchmark: {args.steps} steps, seed={args.seed}")
    baseline = run_trial(
        "baseline", init_q, args.steps, use_fisher=False,
        fisher_info_scale=args.fisher_info_scale, log_every=args.log_every, verbose=args.verbose,
    )
    fisher = run_trial(
        "fisher", init_q, args.steps, use_fisher=True,
        fisher_info_scale=args.fisher_info_scale, log_every=args.log_every, verbose=args.verbose,
    )

    write_csv(args.output, baseline, fisher)
    print_summary(baseline, fisher, args.output)


if __name__ == "__main__":
    main()