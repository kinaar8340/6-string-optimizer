# viz.py
# Enhanced version with richer annotations:
# - Pyramid plot: Added success status badge (black/red), more metric annotations, tighter layout
# - Smith chart: Full classic Smith chart background (resistance circles, reactance arcs, subtle SWR),
#   unified small circular markers, black-filled pre-jump (white edge), lime-filled selected (black edge),
#   no star marker, thinner arrow, cleaner layout
# - Trajectory frame: Minor title enhancement (added geo dist if available, but kept simple)
# - All plots now have clearer fonts/titles and success indicators where relevant
# - Complete pyramid plot with rich subplots, metric annotations, and reconstruction pyramid
# - Full classic Smith chart background with proper grid lines
# - Improved text offsets, titles, and robustness
# - Trajectory frames unchanged (already solid)

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Arc
from .config import N_POINTS, TIMES, TRUE_DAMPING_RATES, K_MODES, SWR_BEST_LOSS_EST, TRUE_COUPLING_STRENGTH


def save_trajectory_frame(step: int, preds: torch.Tensor, current_loss: float, seed: int,
                          project_to_3d, time_norm, true_proj, data_proj, is_jump: bool = False):
    # === MOVE EVERYTHING TO CPU + NUMPY HERE (fixes the CUDA → matplotlib error) ===
    preds_proj = project_to_3d(preds.detach())

    true_proj_np = true_proj.detach().cpu().numpy() if torch.is_tensor(true_proj) else true_proj
    data_proj_np = data_proj.detach().cpu().numpy() if torch.is_tensor(data_proj) else data_proj
    time_norm_np = time_norm.detach().cpu().numpy() if torch.is_tensor(time_norm) else time_norm

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(true_proj_np[:, 0], true_proj_np[:, 1], true_proj_np[:, 2],
            c='grey', linewidth=0.3, alpha=0.8, label='True (clean)')
    ax.scatter(data_proj_np[:, 0], data_proj_np[:, 1], data_proj_np[:, 2],
               c='black', s=1, alpha=0.4)
    ax.plot(data_proj_np[:, 0], data_proj_np[:, 1], data_proj_np[:, 2],
            c='orange', linewidth=0.3, alpha=0.6, label='Noisy data')

    for i in range(N_POINTS - 1):
        color = plt.cm.viridis(time_norm_np[i])
        ax.plot(preds_proj[i:i+2, 0], preds_proj[i:i+2, 1], preds_proj[i:i+2, 2],
                c=color, linewidth=0.3)

    title = f'Seed {seed} | Step {step:,} | Loss {current_loss:.8f}'
    if is_jump:
        title += ' | JUMP!'
    ax.set_title(title, fontsize=16)
    ax.legend(loc='upper left')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')

    suffix = "_jump" if is_jump else ""
    filename = f'viz_frames/seed_{seed}_step_{step:07d}{suffix}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   Viz frame saved: {filename}")


def save_detailed_pyramid_plot(seed: int, status: str, true_geo_dist: float, total_recon_mse_pred: float,
                               coupling_val: float, coupling_err: float, wall_time: float,
                               damping_rmse: float, damping_corr: float, speed_rel_std: float,
                               freq_rmse: float, freq_corr: float, max_inharm_b: float,
                               aligned_rates, aligned_speed_scalars, aligned_learned_freq,
                               aligned_inharm_b, relative_mse_pred, true_var_per_mode,
                               true_full_freq_np: np.ndarray,
                               strict_success: bool = False, loose_success: bool = False):
    fig = plt.figure(figsize=(26, 22), dpi=150)
    gs = GridSpec(4, 4, figure=fig, height_ratios=[1, 1, 1, 1.8], hspace=0.55, wspace=0.45)

    modes = np.arange(1, K_MODES + 1)

    # Success badge
    success_color = 'black' if strict_success else 'orange' if loose_success else 'red'
    success_text = 'STRICT SUCCESS' if strict_success else 'LOOSE SUCCESS' if loose_success else 'FAIL'
    fig.suptitle(f"Seed {seed} | {status} | {success_text}",
                 fontsize=22, fontweight='bold', color=success_color, y=0.96)

    # === Damping rates ===
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(modes, TRUE_DAMPING_RATES.cpu().numpy(), 'o-', color='black', label='True', linewidth=3)
    ax0.plot(modes, aligned_rates.cpu().numpy(), 's--', color='blue', label='Learned', linewidth=3)
    ax0.set_title(f'Damping Rates\n(RMSE {damping_rmse:.4f}, corr {damping_corr:.3f})', fontsize=14)
    ax0.set_xlabel('Mode')
    ax0.legend(fontsize=12)

    # === Speed scalars ===
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.bar(modes, aligned_speed_scalars.cpu().numpy(), color='skyblue')
    ax1.axhline(aligned_speed_scalars.mean().item(), color='red', linestyle='--', label=f'Mean')
    ax1.set_title(f'Speed Scalars\n(rel std {speed_rel_std:.5f})', fontsize=14)
    ax1.set_xlabel('Mode')
    ax1.legend(fontsize=12)

    # === Coupling strength ===
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.bar(['Learned', 'True'], [coupling_val, TRUE_COUPLING_STRENGTH], color=['blue', 'black'])
    ax2.set_title(f'Coupling Strength\n(err {coupling_err:.4f})', fontsize=14)

    # === Frequencies ===
    ax3 = fig.add_subplot(gs[1, :3])
    ax3.scatter(true_full_freq_np, aligned_learned_freq.cpu().numpy(), c='purple', s=80, alpha=0.8)
    minf = min(true_full_freq_np.min(), aligned_learned_freq.min().item())
    maxf = max(true_full_freq_np.max(), aligned_learned_freq.max().item())
    ax3.plot([minf, maxf], [minf, maxf], 'r--', linewidth=2)
    ax3.set_xlabel('True Frequency')
    ax3.set_ylabel('Learned Frequency')
    ax3.set_title(f'Frequencies (RMSE {freq_rmse:.2f}, corr {freq_corr:.3f})', fontsize=14)

    # === Inharmonicity B ===
    ax4 = fig.add_subplot(gs[2, :2])
    ax4.plot(modes, aligned_inharm_b.cpu().numpy(), 'o-', color='orange', linewidth=3)
    ax4.set_title(f'Inharmonicity B (max {max_inharm_b:.8f})', fontsize=14)
    ax4.set_xlabel('Mode')
    ax4.set_ylabel('B coefficient')

    # === Reconstruction error pyramid ===
    ax_py = fig.add_subplot(gs[2:4, 2:])
    log_rel_mse = np.log10(relative_mse_pred + 1e-10)
    ax_py.bar(modes, log_rel_mse, color='lightcoral', edgecolor='darkred', label='log₁₀ Relative MSE')
    ax_py.set_xlabel('Mode', fontsize=14)
    ax_py.set_ylabel('log₁₀(Relative Recon Error)', fontsize=14)
    ax_py.set_title('Reconstruction Error Pyramid', fontsize=16)

    ax_py_twin = ax_py.twinx()
    ax_py_twin.plot(modes, true_var_per_mode, 'o-', color='black', linewidth=3, label='True Mode Variance Share')
    ax_py_twin.set_ylabel('True Variance Fraction', fontsize=14)

    # === Metric summary box ===
    textstr = '\n'.join([
        f'Geo Dist: {true_geo_dist:.6f}',
        f'Total Recon MSE: {total_recon_mse_pred:.10f}',
        f'Coupling Err: {coupling_err:.4f}',
        f'Damping RMSE: {damping_rmse:.4f}',
        f'Speed Rel Std: {speed_rel_std:.6f}',
        f'Freq RMSE: {freq_rmse:.2f}',
        f'Max Inharm B: {max_inharm_b:.8f}',
        f'Wall Time: {wall_time:.1f}s'
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax_py.text(0.02, 0.98, textstr, transform=ax_py.transAxes, fontsize=13,
               verticalalignment='top', bbox=props)

    filename = f"plots/final_pyramid_seed_{seed}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   Detailed pyramid plot saved: {filename}")


def _draw_classic_smith_background(ax):
    """Draw a full classic Smith chart background with resistance circles, reactance arcs, and subtle SWR circles."""
    # Outer unit circle (|Γ| = 1)
    ax.add_patch(Circle((0, 0), 1.0, fill=False, color='black', linewidth=1.5))

    # Real axis line
    ax.plot([-1.1, 1.1], [0, 0], color='black', linewidth=1)

    # Subtle SWR circles (gray, not overpowering)
    swrs = [1.5, 2.0, 3.0, 5.0, 10.0]
    for swr in swrs:
        rho = (swr - 1.0) / (swr + 1.0)
        ax.add_patch(Circle((0, 0), rho, fill=False, color='gray', linewidth=0.8))
        ax.text(rho + 0.02, 0.02, f'{swr}', fontsize=9, color='gray', ha='left', va='bottom')

    # Perfect match label
    ax.text(0, 0.02, 'SWR=1.0\nPerfect\nMatch', ha='center', va='bottom', fontsize=10, color='black')

    # Constant resistance circles (right half, dashed blue)
    resistances = [0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]
    for r in resistances:
        center_x = r / (r + 1.0)
        radius = 1.0 / (r + 1.0)
        ax.add_patch(Circle((center_x, 0), radius, fill=False, color='blue', linewidth=0.8, ls='--'))

    # Constant reactance arcs (upper and lower, dashed blue)
    reactances = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    for base_x in reactances:
        for sign in [1.0, -1.0]:
            x = sign * base_x
            center = (1.0, 1.0 / x)
            radius = abs(1.0 / x)
            theta1 = 0 if x > 0 else 180
            theta2 = 180 if x > 0 else 360
            ax.add_patch(Arc(center, 2 * radius, 2 * radius, angle=0.0,
                             theta1=theta1, theta2=theta2,
                             color='blue', linewidth=0.8, ls='--'))

    # Axis limits and aspect
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')


def plot_smith_chart(gamma_points, seed: int, jumps_performed: int,
                     gamma_labels=None, pre_jump_idx=None, selected_idx=None,
                     pre_jump_loss=None, selected_loss=None,
                     jump_step=None, max_std=None, filename=None):
    if gamma_points is None or len(gamma_points) == 0:
        return

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    # === Full classic Smith chart background ===
    _draw_classic_smith_background(ax)

    # === Process gamma points ===
    gamma_array = np.array(gamma_points)
    if np.iscomplexobj(gamma_array):
        re = np.real(gamma_array).ravel()
        im = np.imag(gamma_array).ravel()
        mag = np.abs(gamma_array)
    else:
        re = gamma_array.ravel()
        im = np.zeros_like(re)
        mag = np.abs(re + 0j)  # ensure complex for consistency

    mag_max = mag.max()
    if mag_max > 0:
        mag_norm = mag / mag_max
    else:
        mag_norm = np.zeros_like(mag)
    colors = plt.cm.viridis_r(1.0 - mag_norm)

    # === Candidate points (small circular) ===
    cand_size = 110
    ax.scatter(re, im, c=colors, s=cand_size, marker='o',
               edgecolors='black', linewidth=1.0, alpha=0.92, zorder=5)

    if gamma_labels is not None:
        for idx, (r, i, label) in enumerate(zip(re, im, gamma_labels)):
            offset_y = 0.10 if i >= 0 else -0.10
            va = 'bottom' if i >= 0 else 'top'
            ax.text(r, i + offset_y, label,
                    ha='center', va=va, fontsize=10, color='darkmagenta',
                    fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=0.8),
                    zorder=6)

    # === Pre-jump point ===
    pr = pi = None
    if pre_jump_idx is not None:
        pr = re[pre_jump_idx]
        pi = im[pre_jump_idx]
        special_size = 140
        ax.scatter(pr, pi, s=special_size, marker='o',
                   facecolors='black', edgecolors='black', linewidth=1.0,
                   zorder=10, label='Pre-jump')
        if pre_jump_loss is not None:
            offset_y = -0.20 if pi >= 0 else 0.20
            va = 'top' if pi >= 0 else 'bottom'
            ax.text(pr, pi + offset_y, f'Loss {pre_jump_loss:.6f}',
                    ha='center', va=va, fontsize=12, color='red', fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.9))

    # === Selected point ===
    sr = si = None
    if selected_idx is not None:
        sr = re[selected_idx]
        si = im[selected_idx]
        ax.scatter(sr, si, s=special_size, marker='o',
                   facecolors='red', edgecolors='black', linewidth=1.0,
                   zorder=11, label='Selected')
        if selected_loss is not None:
            offset_y = 0.22 if si >= 0 else -0.22
            va = 'bottom' if si >= 0 else 'top'
            ax.text(sr, si + offset_y, f'Loss {selected_loss:.6f}',
                    ha='center', va=va, fontsize=12, color='black', fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.9))

    # === Arrow from pre-jump → selected (fixed typo) ===
    if pre_jump_idx is not None and selected_idx is not None:
        dx = sr - pr
        dy = si - pi
        ax.arrow(pr, pi, dx, dy, head_width=0.06, head_length=0.08,
                 fc='black', ec='black', lw=1, length_includes_head=True,
                 alpha=0.9, overhang=0.3)

    # === Legend & colorbar ===
    ax.legend(loc='upper right', fontsize=12)

    sm = plt.cm.ScalarMappable(cmap='viridis_r', norm=plt.Normalize(mag.min(), mag.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.65, pad=0.02)
    cbar.set_label('|Γ| (Mismatch) – lower = better match', fontsize=13)

    # === Title ===
    title = f"Seed {seed} - Jump {jumps_performed:02d}"
    if jump_step is not None:
        title += f" | Jump {jump_step}"
    if max_std is not None:
        title += f" | max std {max_std:.2f}"
    ax.set_title(title, fontsize=18, pad=30)

    if filename:
        plt.savefig(filename, dpi=400, bbox_inches='tight')
        plt.close(fig)
        print(f"   Smith chart saved: {filename}")
    else:
        plt.show()
