# ~/mlpa/training_evaluation/config.py
# Last Modified: 2026-02-09
# + NOISE_AMP = 0.07
# + coupling_prior_lambda = 0.12    # Slight boost for stability
# + speed_uniform_lambda = 18.0     # To counter potential std increase
# + inharm_l2_lambda = 18.0         # Mild increase to suppress leakage
# + inharm_ceiling_lambda = 500.0   # Stronger ceiling for robustness
# + ROLLOUT_HORIZON = 30000         # Better candidate eval at high noise
# + damping_prior_lambda = 12.0     # Stricter success rate

import torch
import os

os.makedirs('plots', exist_ok=True)
os.makedirs('viz_frames', exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Geometry ===
DIM = 60
K_MODES = 8         # 8
N_POINTS = 400      # 200
TIMES = torch.linspace(-2.0, 2.0, N_POINTS, device=device)
IDEAL_HARMONICS = torch.arange(1, K_MODES + 1, device=device).float()

# === True physics ===
TRUE_DAMPING_RATES = torch.tensor([0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50], device=device)
TRUE_INHARM_B = torch.tensor([0.0, 0.0, 0.0, 0.00005, 0.0001, 0.0002, 0.0004, 0.0006], device=device)
TRUE_COUPLING_STRENGTH = 0.30
VELOCITY_SCALE_BASE = 2.0
DATA_SEED = 42

# === Hyperparameters ===
# update from tuner output here
coupling_prior_lambda = 0.10
speed_uniform_lambda = 12.0         # 10.0  | For flatter speed scalars at higher noise→ 12–15 | 7-7-26 17:34
damping_prior_lambda = 8.0
inharm_l2_lambda = 15.0             # 3.0   |
inharm_ceiling_lambda = 400.0       # 200.0 |
inharm_ceiling_threshold = 0.002

# === Training ===
MAX_STEPS = 100000              # Increased to give more post-jump optimization time

# === Population-based punctuated equilibrium ===
MAX_JUMPS = 20                 # Allow many escapes (real runs use ~5–20; safety cap)
INCLUDE_ZERO_JUMP = True        # Always keep the no-noise baseline as a candidate
JUMP_STD_MIN = 3.0              # stronger baseline kicks
JUMP_STD_MAX = 10.0              # more punch when deeply stuck
POP_SIZE = 20                   # more candidates → better chance of strong escape
ROLLOUT_HORIZON = 5000         # 30000 | longer evaluation of candidates

# === Entropic selection schedule ===
ENTROPIC_START_TEMP = 0.2      # Less exploration early, was 1.5 Higher early temperature → more exploration in first jumps
ENTROPIC_DECAY = 0.85           # Faster to greedy, was 0.85 Slower decay → stays somewhat explorative longer
ENTROPIC_MIN_TEMP = 0.0005     # Almost pure greedy late, was 0.01 Near-greedy late-game but still tiny randomness

# === Stagnation detection ===
STAGNATION_PATIENCE = 5000     # 5000 Jump sooner → prevent deep entrapment
BASE_STAGNATION_THRESHOLD_LOW = 1e-6
MIN_STEP_FOR_JUMP_CHECK = 15000   # 20000 for aggressive testing

# === Runtime defaults ===
NUM_SEEDS = 1                   # increase for full sweeps
NOISE_AMP = 0.06
FORCE_PUNCTUATED = None

# === Success criteria ===
# unchanged — goal is reliable strict
STRICT_MAX_INHARM = 0.005
STRICT_SPEED_STD = 0.01
STRICT_COUPLING_ERR = 0.05
STRICT_GEO_DIST = 0.3
STRICT_DAMPING_RMSE = 0.05
STRICT_DAMPING_CORR = 0.95
STRICT_FREQ_CORR = 0.98
STRICT_FREQ_RMSE = 5.0

LOOSE_MAX_INHARM = 0.01
LOOSE_SPEED_STD = 0.05

# === Curriculum staging ===
# LR scheduling — unchanged, still solid
STAGE1_STEPS = 250000
STAGE1_LR_GEO = 0.18        # 0.16 |
STAGE1_LR_SLOW = 0.0001

STAGE2_STEPS = 500000
STAGE2_LR_GEO = 0.14        # 0.15 |
STAGE2_LR_SLOW = 0.02

STAGE3_STEPS = 750000
STAGE3_LR_GEO = 0.10        # 0.12 |
STAGE3_LR_SLOW = 0.12

# Speed / parallelism toggles (highly recommended ON)
USE_AMP = True
USE_PARALLEL_ROLLOUTS = True
PARALLEL_MAX_WORKERS = 20       # max: 24 for nvidia 4090 |

# === Smith chart inspired experimental features ===
RECIPROCAL_NOISE = True             # Admittance-style reciprocal perturbation on slow params
USE_SWR_SELECTION = True            # Use SWR-based scoring for physically-biased candidate selection
TANGENT_PROJECT_AFTER_NOISE = True  # Project vel_dir_raw (and optionally coupling) back to tangent space after noiseroject vel_dir_raw back to tangent space after noise
RECIPROCAL_STRENGTH = 0.5           # Strength for reciprocal noise (gentler flips)
SWR_BEST_LOSS_EST = 0.12           # 0.15 Optimistic "characteristic impedance" (tune downward from best observed loose losses), was 0.03

# === Fisher-Rao invariant priors (Phase 1: damping envelope) ===
fr_invariant_weight = 0.3
fr_invariant_damping = 1.0

# === Real-audio extension ===
REAL_AUDIO_MAX_STEPS = 20000
REAL_AUDIO_STFT_WEIGHT = 0.5
REAL_AUDIO_STFT_FFT_SIZES = [512, 1024, 2048]
REAL_AUDIO_STFT_HOP_RATIO = 0.25
REAL_AUDIO_SR = 44100

# Real-audio punctuated equilibrium (shorter patience than synthetic)
REAL_AUDIO_STAGNATION_PATIENCE = 1500
REAL_AUDIO_MIN_STEP_FOR_JUMP = 2000
REAL_AUDIO_MAX_JUMPS = 10
REAL_AUDIO_ROLLOUT_HORIZON = 2000
REAL_AUDIO_POP_SIZE = 12
REAL_AUDIO_JUMP_STD_MIN = 2.0
REAL_AUDIO_JUMP_STD_MAX = 7.0

# Streaming partial tracker
STREAM_HOP_LENGTH = 256
STREAM_N_FFT = 1024
STREAM_CHUNK_SECONDS = 0.05

# GPU streaming STFT (sub-10ms target)
GPU_STFT_N_FFT = 512
GPU_STFT_HOP = 128

# Jump testing toggles (real-audio debug)
JUMP_TEST_LOW_PATIENCE = 300
JUMP_TEST_MIN_STEP = 500
JUMP_TEST_FORCE_EVERY = 800
JUMP_TEST_PLATEAU_AT = 600

# Live microphone
LIVE_MIC_BLOCKSIZE = 1024
LIVE_MIC_DEFAULT_SECONDS = 5.0