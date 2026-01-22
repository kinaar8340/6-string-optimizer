
# src/optimizer/burst_optimizer.py

import torch
from torch.optim import Optimizer

import geoopt

class GeooptBurstOptimizer(Optimizer):
    """
    Highly adaptive Riemannian optimizer for the Sphere manifold, balancing speed and stability:
    - Adaptive quantile-based gradient clipping
    - Burst triggering driven by clipped Riemannian gradient norms (real-time adaptation)
    - Dynamic momentum damping (higher retention → more aggressive when loss high)
    - Dynamic burst factor (higher multiplier → more aggressive bursts when loss high)
    - Loss-scaled noise injection and adaptive burst threshold
    - Clear phased step() for readability
    """
    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.92,
        burst_threshold=7.0,
        burst_factor_max=6.0,
        burst_factor_min=2.0,
        max_theta=torch.pi / 12,
        damping=0.99,
        damping_min=0.92,
        damping_loss_scale=100000.0,
        damping_ema_alpha=0.05,
        twist_rate=1.0,
        stagnation_window=40,
        stagnation_thresh=1e-3,
        stagnation_noise_amp=0.01,
        warm_up_steps=3000,
        burst_schedule_interval=250,
        grad_normalization=True,
        max_rgrad_norm=float("inf"),
        min_burst_threshold=4.0,
        max_burst_threshold=10.0,
        adapt_burst_threshold=True,
        good_improve_multiplier=8.0,
        verbose=True,
        adaptive_grad_clip=True,
        clip_quantile=0.95,
        clip_multiplier=2.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            burst_threshold=burst_threshold,
            burst_factor_max=burst_factor_max,
            burst_factor_min=burst_factor_min,
            max_theta=max_theta,
            damping=damping,
            damping_min=damping_min,
            damping_loss_scale=damping_loss_scale,
            damping_ema_alpha=damping_ema_alpha,
            twist_rate=twist_rate,
            warm_up_steps=warm_up_steps,
            burst_schedule_interval=burst_schedule_interval,
            grad_normalization=grad_normalization,
            max_rgrad_norm=max_rgrad_norm,
            adaptive_grad_clip=adaptive_grad_clip,
            clip_quantile=clip_quantile,
            clip_multiplier=clip_multiplier,
        )
        super().__init__(params, defaults)

        self.stagnation_window = stagnation_window
        self.stagnation_thresh = stagnation_thresh
        self.base_stagnation_noise_amp = stagnation_noise_amp

        self.min_burst_threshold = min_burst_threshold
        self.max_burst_threshold = max_burst_threshold
        self.adapt_burst_threshold = adapt_burst_threshold
        self.good_improve_multiplier = good_improve_multiplier

        self.verbose = verbose

        self.current_step = 0
        self.loss_history = []
        self.smoothed_loss = None

        # Validate and store manifold
        self.manifold = None
        for group in self.param_groups:
            for p in group["params"]:
                if hasattr(p, "manifold"):
                    if self.manifold is None:
                        self.manifold = p.manifold
                    elif self.manifold is not p.manifold:
                        raise ValueError("All parameters must share the same manifold.")
        if not isinstance(self.manifold, geoopt.manifolds.Sphere):
            raise ValueError("Optimizer designed for geoopt Sphere manifold.")

        # Initial setup
        with torch.no_grad():
            for group in self.param_groups:
                group["twist"] = 0.0
                for p in group["params"]:
                    p.data = self.manifold.projx(p.data)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.current_step += 1

        loss_val = None
        if loss is not None:
            loss_val = loss.item() if torch.is_tensor(loss) else float(loss)

            if self.smoothed_loss is None:
                self.smoothed_loss = loss_val
            else:
                alpha = self.param_groups[0]["damping_ema_alpha"]
                self.smoothed_loss = (1 - alpha) * self.smoothed_loss + alpha * loss_val

        global_twist_proxy = 0.0
        global_num_instances = 0

        for group in self.param_groups:
            # Warm-up
            effective_lr = group["lr"]
            if group["warm_up_steps"] > 0:
                progress = min(1.0, self.current_step / group["warm_up_steps"])
                effective_lr *= progress

            # First pass: collect rgrad norms for adaptive clipping
            rgrad_norms = []
            for p in group["params"]:
                if p.grad is None:
                    continue
                x = p.data
                egrad = p.grad.data
                rgrad = self.manifold.egrad2rgrad(x, egrad)
                instance_norms = rgrad.norm(dim=-1)
                rgrad_norms.append(instance_norms)

            # Determine clip norm
            if rgrad_norms:
                all_norms = torch.cat(rgrad_norms)
                if group["adaptive_grad_clip"] and all_norms.numel() > 1:
                    clip_norm = torch.quantile(all_norms, group["clip_quantile"]) * group["clip_multiplier"]
                else:
                    clip_norm = group["max_rgrad_norm"]
            else:
                clip_norm = group["max_rgrad_norm"]

            # Dynamic damping & burst factor (aggressive when loss high)
            if self.smoothed_loss is not None:
                loss_factor = 1 - torch.exp(torch.tensor(-self.smoothed_loss / group["damping_loss_scale"]))
                effective_damping = group["damping_min"] + (group["damping"] - group["damping_min"]) * loss_factor
                effective_burst_factor = group["burst_factor_min"] + (group["burst_factor_max"] - group["burst_factor_min"]) * loss_factor
            else:
                effective_damping = group["damping"]
                effective_burst_factor = group["burst_factor_max"]

            # Second pass: clip, accumulate twist, update momentum
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if "v" not in state:
                    state["v"] = torch.zeros_like(p.data)
                    state["old_x"] = p.data.clone()

                x = p.data
                egrad = p.grad.data
                rgrad = self.manifold.egrad2rgrad(x, egrad)

                # Clip
                rgrad_norm = rgrad.norm(dim=-1, keepdim=True)
                scale = torch.min(torch.ones_like(rgrad_norm), clip_norm / rgrad_norm.clamp(min=1e-8))
                clipped_rgrad = rgrad * scale
                clipped_norm = clipped_rgrad.norm(dim=-1, keepdim=True)

                # Optional normalization
                if group["grad_normalization"]:
                    mean_clipped_norm = clipped_norm.mean().clamp(min=1e-8)
                    clipped_rgrad = clipped_rgrad * (1.0 / mean_clipped_norm)

                # Accumulate twist
                twist_inc = clipped_norm.mean() * group["twist_rate"]
                group["twist"] += twist_inc.item()
                global_twist_proxy += twist_inc.item()
                global_num_instances += x.shape[0]

                # Update momentum velocity
                state["v"] = group["momentum"] * state["v"] + effective_lr * clipped_rgrad

            # Burst logic
            twist_triggered = group["twist"] > group["burst_threshold"]
            scheduled = (group["burst_schedule_interval"] > 0 and
                         self.current_step % group["burst_schedule_interval"] == 0)
            apply_burst = twist_triggered or scheduled

            factor = min(20.0, effective_burst_factor if apply_burst else 1.0)

            if self.verbose:
                if scheduled:
                    print(f"[Scheduled Burst] Step {self.current_step}")
                if factor > 1.0 + 1e-3:
                    print(
                        f"[Burst] Factor={factor:.2f} "
                        f"(twist_trigger={twist_triggered}, scheduled={scheduled}, twist={group['twist']:.2f})"
                    )
                if self.current_step % 500 == 0:
                    print(
                        f"[Dynamic] Step {self.current_step} | Damping={effective_damping:.4f} | "
                        f"Burst Factor={effective_burst_factor:.2f} | Smoothed Loss≈{self.smoothed_loss:.2f}"
                    )

            if apply_burst:
                group["twist"] = 0.0

            # Apply updates
            for p in group["params"]:
                state = self.state[p]
                if "v" not in state:
                    continue

                update_vec = factor * state["v"]

                # Cap step size
                theta = update_vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                cap_mask = theta > group["max_theta"]
                if cap_mask.any():
                    update_vec = update_vec * torch.where(
                        cap_mask, group["max_theta"] / theta, torch.ones_like(theta)
                    )

                # Retraction
                new_x = self.manifold.retr(p.data, update_vec)

                state["old_x"].copy_(p.data)
                p.data = new_x

                # Transport velocity
                state["v"] = self.manifold.transp(state["old_x"], new_x, state["v"] * effective_damping)

        # Stagnation detection & adaptation
        if loss_val is not None:
            self.loss_history.append(loss_val)
            if len(self.loss_history) > self.stagnation_window:
                self.loss_history.pop(0)

            if len(self.loss_history) == self.stagnation_window:
                improvement = self.loss_history[0] - loss_val

                if improvement < self.stagnation_thresh:
                    noise_amp = self.base_stagnation_noise_amp
                    if self.verbose:
                        print(f"[Stagnation] Plateau detected → injecting noise (amp={noise_amp:.4f})")

                    for group in self.param_groups:
                        for p in group["params"]:
                            state = self.state[p]
                            if "v" in state:
                                noise = noise_amp * torch.randn_like(state["v"])
                                noise = noise - self.manifold.inner(p.data, noise, keepdim=True) * p.data
                                state["v"] += noise

                if self.adapt_burst_threshold:
                    ref_threshold = self.param_groups[0]["burst_threshold"]
                    if improvement > self.good_improve_multiplier * self.stagnation_thresh:
                        new_threshold = max(self.min_burst_threshold, ref_threshold * 0.9)
                    elif improvement < self.stagnation_thresh:
                        new_threshold = min(self.max_burst_threshold, ref_threshold * 1.111)
                    else:
                        new_threshold = ref_threshold

                    if abs(new_threshold - ref_threshold) > 1e-6:
                        for g in self.param_groups:
                            g["burst_threshold"] = new_threshold
                        if self.verbose:
                            direction = "↓" if new_threshold < ref_threshold else "↑"
                            print(f"[Adapt] Burst threshold {direction} {new_threshold:.2f}")

        return loss


# Updated src/optimizer/burst_optimizer.py excerpt — HierarchicalGeooptBurstOptimizer only
# Changes:
# - Stronger response to stagnation (higher multiplier cap + sensitivity)
# - Much higher headroom on boosts (50× burst, 10× theta)
# - Conservative resets: only restore originals on *significant* improvement (>2× thresh)
#   This prevents premature reset cascades and lets strong boosts persist longer when needed

# src/optimizer/burst_optimizer.py (updated Hierarchical caps)

class HierarchicalGeooptBurstOptimizer:
    def __init__(
        self,
        optimizer,
        stagnation_window=200,
        stagnation_thresh=1e-3,
        burst_boost=2.0,
        theta_boost=1.3,
        name="Macro",
        verbose=True,
    ):
        self.optimizer = optimizer
        self.stagnation_window = stagnation_window
        self.stagnation_thresh = stagnation_thresh
        self.burst_boost = burst_boost
        self.theta_boost = theta_boost
        self.name = name
        self.verbose = verbose

        self.loss_history = []
        self.originals = {id(g): (g["burst_factor_max"], g["max_theta"]) for g in self.param_groups}

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self, closure=None):
        loss = self.optimizer.step(closure=closure)

        if loss is not None:
            loss_val = loss.item() if torch.is_tensor(loss) else loss
            self.loss_history.append(loss_val)
            if len(self.loss_history) > self.stagnation_window:
                self.loss_history.pop(0)

            if len(self.loss_history) == self.stagnation_window:
                improvement = self.loss_history[0] - loss_val

                if improvement < self.stagnation_thresh:
                    lack_of_improvement = max(0.0, self.stagnation_thresh - improvement)
                    severity = lack_of_improvement
                    multiplier = 1.0 + min(8.0, severity / self.stagnation_thresh * 5.0)
                    if self.verbose:
                        print(
                            f"[{self.name} Burst] Long-term stagnation (severity={severity:.1e}) → "
                            f"boosting burst_max x{self.burst_boost * multiplier:.2f}, theta x{self.theta_boost * multiplier:.2f}"
                        )
                    for group in self.param_groups:
                        orig_bf, orig_mt = self.originals[id(group)]
                        boosted_bf = orig_bf * self.burst_boost * multiplier
                        boosted_theta = orig_mt * self.theta_boost * multiplier
                        group["burst_factor_max"] = min(orig_bf * 80.0, boosted_bf)
                        group["max_theta"] = min(orig_mt * 15.0, boosted_theta)
                else:
                    if improvement > 2 * self.stagnation_thresh:
                        modified = any(
                            abs(g["burst_factor_max"] - self.originals[id(g)][0]) > 1e-6 or
                            abs(g["max_theta"] - self.originals[id(g)][1]) > 1e-6
                            for g in self.param_groups
                        )
                        if modified and self.verbose:
                            print(f"[{self.name} Reset] Significant progress → restoring original burst_factor_max/max_theta")
                            for group in self.param_groups:
                                orig_bf, orig_mt = self.originals[id(group)]
                                group["burst_factor_max"] = orig_bf
                                group["max_theta"] = orig_mt

        return loss

    def zero_grad(self):
        self.optimizer.zero_grad()