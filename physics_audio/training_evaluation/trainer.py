# ~/mlpa/training_evaluation/trainer.py
# Full universal version with all fixes and improvements:
# - Modern torch.amp API throughout
# - Robust post-jump visualization and loss computation (fresh no-grad forward on selected best state)
# - Improved selection logic: always accept the best candidate if it improves over pre_jump_loss
#   (also accepts entropic if best doesn't but entropic does; otherwise continues current state)
# - Parallel rollouts with robust try/except sequential fallback
# - CPU transfer for state_dict/data to ensure safe multiprocessing pickling
# - Early stopping in rollouts
# - Random uniform noise stds for better exploration + fixed zero-noise continuation
# - All previous features preserved where compatible
# - Integrated rich.progress for enhanced visual feedback (main loop + nested rollout bars)
# - Live per-candidate printing during rollouts for immediate feedback

import time
import random
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from rich.progress import Progress, BarColumn, MofNCompleteColumn, TextColumn, TimeRemainingColumn
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from geoopt.optim import RiemannianAdam
from geoopt.manifolds import Stiefel
from concurrent.futures import ProcessPoolExecutor, as_completed
from .config import *
from .model import StiefelDampedCoupledInharmGR
from .utils import stiefel_dist, safe_proj, align_and_compute_freq, get_pca_initial_basis, manifold
from .losses import total_loss
from .audio_utils import modal_synthesis_torch
from .viz import save_trajectory_frame, save_detailed_pyramid_plot, plot_smith_chart

# Force spawn for CUDA safety
mp.set_start_method('spawn', force=True)


def loss_to_gamma(loss: float) -> float:
    if loss <= SWR_BEST_LOSS_EST:
        return 0.0
    swr = loss / SWR_BEST_LOSS_EST
    return (swr - 1.0) / (swr + 1.0)


def add_euclidean_noise(model: nn.Module, std: float):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad or name == "base":
                continue
            if RECIPROCAL_NOISE and name in ["log_speed", "log_base_rate", "log_slope",
                                             "raw_lin_b", "raw_quad_b", "raw_coupling_strength"]:
                exp_param = torch.exp(param)
                noise = torch.randn_like(exp_param) * std * RECIPROCAL_STRENGTH
                perturbed = 1.0 / (1.0 / exp_param + noise)
                perturbed = perturbed.clamp(min=exp_param * 0.1, max=exp_param * 10.0)
                param.copy_(torch.log(perturbed.clamp(min=1e-8)))
            else:
                param.add_(torch.randn_like(param) * std)

        if TANGENT_PROJECT_AFTER_NOISE:
            model.vel_dir_raw.data = manifold.proju(model.base, model.vel_dir_raw.data)


def _rollout_worker(arg):
    (idx, pre_state_dict_cpu, noise_std, data_points_cpu, times_cpu,
     initial_basis_cpu, worker_seed, lr_geo, lr_slow, prior_targets_cpu) = arg

    torch.manual_seed(worker_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from .model import StiefelDampedCoupledInharmGR
    from .config import DIM, K_MODES, ROLLOUT_HORIZON, USE_AMP
    from .losses import total_loss
    from torch.nn.utils import clip_grad_norm_

    model = StiefelDampedCoupledInharmGR(DIM, K_MODES, initial_basis_cpu.to(device))
    model.load_state_dict(pre_state_dict_cpu)
    model.to(device)

    add_euclidean_noise(model, noise_std)

    optimizer = RiemannianAdam([
        {'params': model.base, 'lr': lr_geo},
        {'params': [p for n, p in model.named_parameters() if p is not model.base], 'lr': lr_slow}
    ], stabilize=10)

    scaler = GradScaler('cuda', enabled=USE_AMP)

    data_points = data_points_cpu.to(device)
    times = times_cpu.to(device)

    best_local_loss = float('inf')
    best_state = None
    steps_no_improve = 0
    patience = 4000
    min_steps = 2000
    steps_performed = 0

    for local_step in range(ROLLOUT_HORIZON):
        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=USE_AMP):
            preds, damping_rates, coupling_strength, inharm_b, speed_scalars, full_freq = model(times)
            loss = total_loss(
                preds, data_points, damping_rates, coupling_strength, inharm_b, speed_scalars,
                prior_targets=prior_targets_cpu,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        steps_performed = local_step + 1
        current_local_loss = loss.item()

        if current_local_loss < best_local_loss - 1e-6:
            best_local_loss = current_local_loss
            best_state = copy.deepcopy(model.state_dict())
            steps_no_improve = 0
        else:
            steps_no_improve += 1

        if steps_no_improve >= patience and steps_performed >= min_steps:
            break

    # Load best state and compute accurate final loss
    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        preds, damping_rates, coupling_strength, inharm_b, speed_scalars, full_freq = model(times)
        final_loss = total_loss(
            preds, data_points, damping_rates, coupling_strength, inharm_b, speed_scalars,
            prior_targets=prior_targets_cpu,
        ).item()

    # Return CPU state_dict for safe pickling
    best_state_cpu = {k: v.cpu().clone().detach() if torch.is_tensor(v) else v for k, v in model.state_dict().items()}

    return {
        'idx': idx,
        'noise_std': noise_std,
        'final_loss': final_loss,
        'best_state': best_state_cpu,
        'steps': steps_performed
    }


def run_single_seed(
    seed: int,
    noise_amp: float = NOISE_AMP,
    force_punctuated: bool | None = FORCE_PUNCTUATED,
    *,
    real_audio_data: torch.Tensor | None = None,
    real_audio_times: torch.Tensor | None = None,
    real_audio_initial_basis: torch.Tensor | None = None,
    target_waveform: torch.Tensor | None = None,
    audio_sr: int = REAL_AUDIO_SR,
    audio_duration: float | None = None,
    prior_targets: dict | None = None,
    max_steps: int | None = None,
    stft_weight: float = 0.0,
    preinitialized_model: StiefelDampedCoupledInharmGR | None = None,
):
    start_time = time.time()
    training_steps = max_steps if max_steps is not None else MAX_STEPS
    use_real_audio = real_audio_data is not None

    torch.manual_seed(DATA_SEED + seed)
    np.random.seed(DATA_SEED + seed)
    random.seed(DATA_SEED + seed)

    if use_real_audio:
        data_points = real_audio_data.to(device)
        times = real_audio_times.to(device) if real_audio_times is not None else TIMES
        initial_basis = real_audio_initial_basis if real_audio_initial_basis is not None else get_pca_initial_basis(data_points, K_MODES)
        exact_points = data_points
        true_base = None
        true_vel_dir = None
        true_freq = None
    else:
        # === True physics generation ===
        raw = torch.randn(DIM, K_MODES, device=device)
        true_base = safe_proj(raw)

        true_vel_dir_raw = torch.randn(DIM, K_MODES, device=device)
        true_vel_dir = manifold.proju(true_base, true_vel_dir_raw)
        true_vel_dir = true_vel_dir / (true_vel_dir.norm(dim=0, keepdim=True) + 1e-8)

        true_freq = IDEAL_HARMONICS * torch.sqrt(1 + TRUE_INHARM_B * IDEAL_HARMONICS.pow(2))
        true_vel = true_vel_dir * VELOCITY_SCALE_BASE * true_freq

        true_coupling_raw = torch.randn(K_MODES, K_MODES, device=device) * 0.05
        true_coupling_skew = true_coupling_raw.tril(diagonal=-1) - true_coupling_raw.triu(diagonal=1)
        true_coupling_vel = manifold.proju(true_base, true_base @ true_coupling_skew)

        true_vel_total = true_vel + TRUE_COUPLING_STRENGTH * true_coupling_vel

        abs_times = torch.abs(TIMES).view(-1, 1, 1)
        envelope = torch.exp(-TRUE_DAMPING_RATES * abs_times)

        base_batch = true_base.unsqueeze(0).expand(N_POINTS, -1, -1)
        vel_batch = TIMES.view(-1, 1, 1) * true_vel_total.unsqueeze(0) * envelope
        exact_points = manifold.expmap(base_batch, vel_batch)

        data_points = exact_points + noise_amp * torch.randn_like(exact_points)
        times = TIMES
        initial_basis = get_pca_initial_basis(data_points, K_MODES)

    # === Visualization setup ===
    _, _, proj_matrix = torch.pca_lowrank(data_points.reshape(-1, DIM), q=3, center=True, niter=6)
    proj_matrix = proj_matrix.cpu()
    project_to_3d = lambda pts: (pts.reshape(-1, DIM).cpu() @ proj_matrix).reshape(pts.shape[0], -1, 3).numpy()
    true_proj = project_to_3d(exact_points)
    data_proj = project_to_3d(data_points)
    time_norm = ((TIMES - TIMES.min()) / (TIMES.max() - TIMES.min() + 1e-8)).cpu().numpy()

    # === Model & optimizer ===
    if preinitialized_model is not None:
        model = preinitialized_model.to(device)
    else:
        model = StiefelDampedCoupledInharmGR(DIM, K_MODES, initial_basis).to(device)

    def get_curriculum_lrs(step):
        if step < STAGE1_STEPS:
            return STAGE1_LR_GEO, STAGE1_LR_SLOW
        elif step < STAGE2_STEPS:
            return STAGE2_LR_GEO, STAGE2_LR_SLOW
        else:
            return STAGE3_LR_GEO, STAGE3_LR_SLOW

    optimizer = RiemannianAdam([
        {'params': model.base, 'lr': STAGE1_LR_GEO},
        {'params': [p for n, p in model.named_parameters() if p is not model.base], 'lr': STAGE1_LR_SLOW}
    ], stabilize=10)

    scaler = GradScaler('cuda', enabled=USE_AMP)

    global_step = 0
    jumps_performed = 0
    stagnation_steps = 0
    best_loss = float('inf')

    print("\nStarting training loop...\n")

    # Main progress bar with rich
    main_progress = Progress(
        TextColumn("[bold magenta]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TextColumn("• [red]loss:[/red] {task.fields[loss]} • [green]jumps:[/green] {task.fields[jumps]} • [blue]best:[/blue] {task.fields[best]}"),
    )

    with main_progress:
        task_label = f"Real-audio seed {seed}" if use_real_audio else f"Seed {seed}  noise={noise_amp:.3f}"
        main_task = main_progress.add_task(
            task_label,
            total=training_steps,
            loss="--.------",
            jumps=0,
            best="--.------",
        )

        while global_step < training_steps:
            lr_geo, lr_slow = get_curriculum_lrs(global_step)
            optimizer.param_groups[0]['lr'] = lr_geo
            optimizer.param_groups[1]['lr'] = lr_slow

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=USE_AMP):
                preds, damping_rates, coupling_strength, inharm_b, speed_scalars, full_freq = model(times)
                synth_waveform = None
                if stft_weight > 0.0 and target_waveform is not None and audio_duration is not None:
                    synth_waveform = modal_synthesis_torch(
                        full_freq, damping_rates, audio_duration, audio_sr,
                    )
                loss = total_loss(
                    preds, data_points, damping_rates, coupling_strength, inharm_b, speed_scalars,
                    prior_targets=prior_targets,
                    synth_waveform=synth_waveform,
                    target_waveform=target_waveform,
                    stft_weight=stft_weight,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            current_loss = loss.item()
            global_step += 1

            if current_loss < best_loss - 1e-6:
                best_loss = current_loss
                stagnation_steps = 0
            else:
                stagnation_steps += 1

            main_progress.update(
                main_task,
                advance=1,
                loss=f"{current_loss:.6f}",
                jumps=jumps_performed,
                best=f"{best_loss:.6f}",
            )

            if global_step % 5000 == 0:
                save_trajectory_frame(global_step, preds.detach(), current_loss, seed,
                                      project_to_3d, time_norm, true_proj, data_proj, is_jump=False)

            # === Punctuated jump on stagnation ===
            if (stagnation_steps >= STAGNATION_PATIENCE and
                global_step >= MIN_STEP_FOR_JUMP_CHECK and
                jumps_performed < MAX_JUMPS):

                print(f"\n>>> PUNCTUATED JUMP {jumps_performed + 1} at step {global_step} | loss {current_loss:.8f} <<<")

                pre_jump_loss = current_loss

                # CPU transfer for safe multiprocessing pickling
                pre_state_dict_cpu = {k: v.cpu().clone().detach() if torch.is_tensor(v) else v
                                      for k, v in model.state_dict().items()}
                data_points_cpu = data_points.cpu()
                times_cpu = times.cpu()
                initial_basis_cpu = initial_basis.cpu()
                prior_targets_cpu = prior_targets

                current_jump_std = JUMP_STD_MIN + (JUMP_STD_MAX - JUMP_STD_MIN) * (jumps_performed / max(1, MAX_JUMPS - 1))

                args = []
                for i in range(POP_SIZE):
                    std = 0.0 if INCLUDE_ZERO_JUMP and i == 0 else random.uniform(JUMP_STD_MIN, current_jump_std)
                    worker_seed = random.randint(0, 2**32 - 1)
                    arg = (i, pre_state_dict_cpu, std, data_points_cpu, times_cpu,
                           initial_basis_cpu, worker_seed, lr_geo, lr_slow, prior_targets_cpu)
                    args.append(arg)

                candidates = []
                rollout_progress = Progress(
                    TextColumn("[bold yellow]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TextColumn("{task.percentage:>3.0f}%"),
                    TimeRemainingColumn(),
                )

                if USE_PARALLEL_ROLLOUTS:
                    print(f"   Parallel rollouts with {min(PARALLEL_MAX_WORKERS, POP_SIZE)} workers...")
                    try:
                        with ProcessPoolExecutor(max_workers=min(PARALLEL_MAX_WORKERS, POP_SIZE)) as executor:
                            futures = [executor.submit(_rollout_worker, a) for a in args]
                            with rollout_progress:
                                rollout_task = rollout_progress.add_task("Parallel rollouts", total=POP_SIZE)
                                for future in as_completed(futures):
                                    try:
                                        res = future.result()
                                        candidates.append(res)
                                        print(f"   Candidate {res['idx']:2d} | std {res['noise_std']:.3f} | steps {res['steps']:5d} | loss {res['final_loss']:.8f}")
                                    except Exception as e:
                                        print(f"   Worker failed: {e}")
                                    rollout_progress.advance(rollout_task)
                    except Exception as e:
                        print(f"   Parallel failed ({e}), falling back to sequential")
                        with rollout_progress:
                            rollout_task = rollout_progress.add_task("Sequential rollouts", total=POP_SIZE)
                            for a in args:
                                try:
                                    res = _rollout_worker(a)
                                    candidates.append(res)
                                    print(f"   Candidate {res['idx']:2d} | std {res['noise_std']:.3f} | steps {res['steps']:5d} | loss {res['final_loss']:.8f}")
                                except Exception as e:
                                    print(f"   Sequential worker failed: {e}")
                                rollout_progress.advance(rollout_task)
                else:
                    print("   Sequential rollouts...")
                    with rollout_progress:
                        rollout_task = rollout_progress.add_task("Sequential rollouts", total=POP_SIZE)
                        for a in args:
                            try:
                                res = _rollout_worker(a)
                                candidates.append(res)
                                print(f"   Candidate {res['idx']:2d} | std {res['noise_std']:.3f} | steps {res['steps']:5d} | loss {res['final_loss']:.8f}")
                            except Exception as e:
                                print(f"   Sequential worker failed: {e}")
                            rollout_progress.advance(rollout_task)

                # Summary (sorted by idx)
                for c in sorted(candidates, key=lambda x: x['idx']):
                    print(f"   Candidate {c['idx']:2d} | std {c['noise_std']:.3f} | steps {c['steps']:5d} | loss {c['final_loss']:.8f}")

                # === Selection logic ===
                if USE_SWR_SELECTION:
                    scores = [loss_to_gamma(c['final_loss']) for c in candidates]
                else:
                    scores = [-c['final_loss'] for c in candidates]

                temp = max(ENTROPIC_MIN_TEMP, ENTROPIC_START_TEMP * (ENTROPIC_DECAY ** jumps_performed))
                probs = torch.softmax(torch.tensor(scores) / temp, dim=0).numpy()
                entropic_idx = np.random.choice(len(candidates), p=probs)
                entropic_candidate = candidates[entropic_idx]

                best_candidate = min(candidates, key=lambda c: c['final_loss'])

                selected_candidate = None
                if best_candidate['final_loss'] < pre_jump_loss - 1e-6:
                    selected_candidate = best_candidate
                    print(f"   Selected best candidate {best_candidate['idx']} (std {best_candidate['noise_std']:.3f}) | loss {best_candidate['final_loss']:.8f} (improvement)")
                elif entropic_candidate['final_loss'] < pre_jump_loss - 1e-6:
                    selected_candidate = entropic_candidate
                    print(f"   Selected entropic candidate {entropic_idx} (std {entropic_candidate['noise_std']:.3f}) | loss {entropic_candidate['final_loss']:.8f} (improvement)")
                else:
                    print("   No improvement found → continuing from current state")

                if selected_candidate is not None:
                    model.load_state_dict(selected_candidate['best_state'])
                    jumps_performed += 1

                # Robust post-jump visualization
                with torch.no_grad():
                    preds, damping_rates, coupling_strength, inharm_b, speed_scalars, full_freq = model(times)
                    current_loss = total_loss(
                        preds, data_points, damping_rates, coupling_strength, inharm_b, speed_scalars,
                        prior_targets=prior_targets,
                    ).item()

                save_trajectory_frame(global_step, preds.detach(), current_loss, seed,
                                      project_to_3d, time_norm, true_proj, data_proj, is_jump=(selected_candidate is not None))

                stagnation_steps = 0
                best_loss = current_loss

                # Update main progress with new loss/best after jump
                main_progress.update(
                    main_task,
                    loss=f"{current_loss:.6f}",
                    jumps=jumps_performed,
                    best=f"{best_loss:.6f}",
                )

    wall_time = time.time() - start_time

    # === Final evaluation ===
    with torch.no_grad():
        preds, damping_rates, coupling_strength, inharm_b, speed_scalars, full_freq = model(times)

    learned_vel_dir = manifold.proju(model.base, model.vel_dir_raw)
    learned_vel_dir = learned_vel_dir / (learned_vel_dir.norm(dim=0, keepdim=True) + 1e-8)

    coupling_val = coupling_strength.item()
    speed_rel_std = (speed_scalars.std(unbiased=False) / (speed_scalars.mean() + 1e-8)).item()
    max_inharm_b = inharm_b.max().item()
    per_mode_mse = (preds - exact_points).pow(2).mean(dim=[0, 1])
    true_var_per_mode = exact_points.var(dim=0).mean(dim=0).cpu().numpy()
    relative_mse_pred = (per_mode_mse / (per_mode_mse.detach().new_tensor(true_var_per_mode) + 1e-8)).cpu().numpy()
    total_recon_mse_pred = per_mode_mse.mean().item()

    if use_real_audio:
        target_damping = prior_targets['damping_rates'] if prior_targets else TRUE_DAMPING_RATES
        target_coupling = prior_targets.get('coupling_strength', TRUE_COUPLING_STRENGTH) if prior_targets else TRUE_COUPLING_STRENGTH
        true_geo_dist = 0.0
        coupling_err = abs(coupling_val - target_coupling)
        damping_rmse = torch.sqrt(F.mse_loss(damping_rates, target_damping)).item()
        damping_corr = float(np.corrcoef(damping_rates.cpu().numpy(), target_damping.cpu().numpy())[0, 1])
        freq_rmse = 0.0
        freq_corr = 1.0
        aligned_rates = damping_rates
        aligned_speed_scalars = speed_scalars
        aligned_learned_freq = full_freq
        aligned_inharm_b = inharm_b
        true_full_freq_np = full_freq.cpu().numpy()
        strict_success = damping_rmse < STRICT_DAMPING_RMSE and total_recon_mse_pred < 0.5
        loose_success = strict_success or total_recon_mse_pred < 1.0
    else:
        aligned_rates, aligned_speed_scalars, aligned_learned_freq, aligned_inharm_b = align_and_compute_freq(
            true_vel_dir, learned_vel_dir, damping_rates, speed_scalars, inharm_b, full_freq
        )
        true_geo_dist = stiefel_dist(model.base, true_base).item()
        coupling_err = abs(coupling_val - TRUE_COUPLING_STRENGTH)
        damping_rmse = torch.sqrt(F.mse_loss(aligned_rates, TRUE_DAMPING_RATES)).item()
        damping_corr = np.corrcoef(aligned_rates.cpu().numpy(), TRUE_DAMPING_RATES.cpu().numpy())[0, 1]
        freq_rmse = torch.sqrt(F.mse_loss(aligned_learned_freq, VELOCITY_SCALE_BASE * true_freq)).item()
        freq_corr = np.corrcoef(aligned_learned_freq.cpu().numpy(),
                                (VELOCITY_SCALE_BASE * true_freq).cpu().numpy())[0, 1]
        max_inharm_b = aligned_inharm_b.max().item()
        true_full_freq_np = (VELOCITY_SCALE_BASE * true_freq).cpu().numpy()
        strict_success = (true_geo_dist < STRICT_GEO_DIST and
                          max_inharm_b < STRICT_MAX_INHARM and
                          speed_rel_std < STRICT_SPEED_STD and
                          coupling_err < STRICT_COUPLING_ERR and
                          damping_rmse < STRICT_DAMPING_RMSE and
                          damping_corr > STRICT_DAMPING_CORR and
                          freq_corr > STRICT_FREQ_CORR and
                          freq_rmse < STRICT_FREQ_RMSE)
        loose_success = strict_success or (max_inharm_b < LOOSE_MAX_INHARM and speed_rel_std < LOOSE_SPEED_STD)

    save_detailed_pyramid_plot(
        seed=seed,
        status="FINAL",
        true_geo_dist=true_geo_dist,
        total_recon_mse_pred=total_recon_mse_pred,
        coupling_val=coupling_val,
        coupling_err=coupling_err,
        wall_time=wall_time,
        damping_rmse=damping_rmse,
        damping_corr=damping_corr,
        speed_rel_std=speed_rel_std,
        freq_rmse=freq_rmse,
        freq_corr=freq_corr,
        max_inharm_b=max_inharm_b,
        aligned_rates=aligned_rates,
        aligned_speed_scalars=aligned_speed_scalars,
        aligned_learned_freq=aligned_learned_freq,
        aligned_inharm_b=aligned_inharm_b,
        relative_mse_pred=relative_mse_pred,
        true_var_per_mode=true_var_per_mode,
        true_full_freq_np=true_full_freq_np,
        strict_success=strict_success,
        loose_success=loose_success,
    )

    result = {
        'seed': seed,
        'jumps': jumps_performed,
        'final_geo_dist': true_geo_dist,
        'max_inharm_b': max_inharm_b,
        'speed_rel_std': speed_rel_std,
        'coupling_err': coupling_err,
        'damping_rmse': damping_rmse,
        'damping_corr': damping_corr,
        'freq_rmse': freq_rmse,
        'freq_corr': freq_corr,
        'strict_success': strict_success,
        'loose_success': loose_success,
        'wall_time': wall_time,
        'total_recon_mse_pred': total_recon_mse_pred,
        'model': model,
        'full_freq': full_freq.detach(),
        'damping_rates': damping_rates.detach(),
        'coupling_strength': coupling_val,
        'inharm_b': inharm_b.detach(),
    }
    return result