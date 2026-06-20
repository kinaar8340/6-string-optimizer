# tuner.py
# Multi-Fidelity BoTorch Bayesian Optimization (MFKG) tuner for physics_audio.py
# Updated: added tuning for inharm_l2_lambda & inharm_ceiling_threshold (now 8D space),
#          improved cost model (exact 1:30 ratio), increased final exploitation candidates,
#          added checkpointing of BO history and best config.

import torch
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition import PosteriorMean
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood
from dataclasses import dataclass
import numpy as np
import time
from pathlib import Path
import importlib.util
import multiprocessing
import os
import pickle  # Added for checkpointing

# ==================== Hyperparameter Space ====================
@dataclass
class HyperConfig:
    coupling_prior_lambda: float = 0.10
    speed_uniform_lambda: float = 10.0
    damping_prior_lambda: float = 2.0
    inharm_ceiling_lambda: float = 200.0
    JUMP_SIGMA_MIN: float = 4.0
    JUMP_SIGMA_MAX: float = 12.0
    inharm_l2_lambda: float = 20.0          # New: now tuned
    inharm_ceiling_threshold: float = 0.002 # New: now tuned

HYPER_FIELDS = list(HyperConfig.__dataclass_fields__.keys())
HYPER_DIM = len(HYPER_FIELDS)

HYPER_BOUNDS = [
    (0.01, 1.0),      # coupling_prior_lambda
    (5.0, 20.0),      # speed_uniform_lambda
    (0.5, 5.0),       # damping_prior_lambda
    (100.0, 500.0),   # inharm_ceiling_lambda
    (2.0, 8.0),       # JUMP_SIGMA_MIN
    (8.0, 16.0),      # JUMP_SIGMA_MAX
    (5.0, 50.0),      # inharm_l2_lambda
    (0.0005, 0.005),  # inharm_ceiling_threshold
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Multi-fidelity settings
MAX_NUM_SEEDS = 30
NOISE_AMP = 0.06
FIDELITY_DIM_INDEX = HYPER_DIM
TARGET_FIDELITY = 1.0
FIDELITY_BOUNDS = torch.tensor([[1.0 / MAX_NUM_SEEDS], [1.0]], device=DEVICE)  # Exact 1–30 seeds

hyper_bounds_tensor = torch.tensor(HYPER_BOUNDS, device=DEVICE).t()
BOUNDS = torch.cat([hyper_bounds_tensor, FIDELITY_BOUNDS], dim=1)

# BO settings
INITIAL_POINTS = 30
MAX_BO_ITER = 80
MAX_PARALLEL = 60
NUM_FANTASIES = 64
TARGET_FIDELITIES_DICT = {FIDELITY_DIM_INDEX: TARGET_FIDELITY}

# ==================== Module Loading ====================
def load_optimizer():
    opt_path = Path(__file__).parent / "physics_audio.py"
    if not opt_path.exists():
        raise FileNotFoundError(f"Optimizer not found at {opt_path}")
    spec = importlib.util.spec_from_file_location("optimizer", opt_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# ==================== Config Utilities ====================
def config_from_tensor(tensor: torch.Tensor) -> HyperConfig:
    values = tensor.cpu().numpy()
    return HyperConfig(**dict(zip(HYPER_FIELDS, values)))

def config_to_dict(config: HyperConfig) -> dict:
    return {field: getattr(config, field) for field in HYPER_FIELDS}

# ==================== Parallel Evaluation ====================
def worker_eval(seed: int, config_dict: dict, num_seeds_eval: int):
    opt_module = load_optimizer()
    config = HyperConfig(**config_dict)
    for field in HYPER_FIELDS:
        if hasattr(opt_module, field):
            setattr(opt_module, field, getattr(config, field))

    res = opt_module.run_single_seed(
        seed=seed,
        noise_amp=NOISE_AMP,
        force_punctuated=True
    )
    return res['strict_success']

def evaluate_config(candidate: torch.Tensor) -> float:
    hyper_tensor = candidate[:HYPER_DIM]
    s = candidate[FIDELITY_DIM_INDEX].item()
    config = config_from_tensor(hyper_tensor)

    num_seeds_eval = max(1, min(MAX_NUM_SEEDS, int(round(s * MAX_NUM_SEEDS))))
    print(f"\nEvaluating fidelity s={s:.3f} → {num_seeds_eval} seeds | {config}")

    config_dict = config_to_dict(config)
    seed_offset = 54321
    effective_parallel = min(MAX_PARALLEL, num_seeds_eval)

    start_time = time.time()
    if effective_parallel > 1:
        with multiprocessing.Pool(processes=effective_parallel) as pool:
            args = [(i + seed_offset, config_dict, num_seeds_eval) for i in range(num_seeds_eval)]
            results = pool.starmap(worker_eval, args)
    else:
        results = [worker_eval(i + seed_offset, config_dict, num_seeds_eval) for i in range(num_seeds_eval)]

    strict_rate = np.mean(results)
    eval_time = time.time() - start_time
    print(f"Result: {strict_rate:.3f} ({int(strict_rate * num_seeds_eval)}/{num_seeds_eval}) | Time: {eval_time:.1f}s\n")
    return strict_rate

# ==================== BoTorch Optimization ====================
def run_botorch_tuning():
    train_x = draw_sobol_samples(bounds=BOUNDS, n=INITIAL_POINTS, q=1).squeeze(1)
    train_y = torch.tensor([evaluate_config(x) for x in train_x], device=DEVICE, dtype=torch.float64).unsqueeze(-1)

    best_y = train_y.max().item()
    print(f"Initial best: {best_y:.3f}")

    project_func = lambda X: project_to_target_fidelity(X=X, target_fidelities=TARGET_FIDELITIES_DICT)

    # Improved cost model: exact proportionality to number of seeds (1 seed ≈ cost 1, 30 seeds ≈ cost 30)
    cost_model = AffineFidelityCostModel(
        fidelity_weights={FIDELITY_DIM_INDEX: float(MAX_NUM_SEEDS)},
        fixed_cost=0.0
    )
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

    for iteration in range(MAX_BO_ITER):
        print(f"\n=== Iteration {iteration + 1}/{MAX_BO_ITER} ===")
        model = SingleTaskMultiFidelityGP(train_x, train_y, data_fidelity_dim=FIDELITY_DIM_INDEX)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))

        current_value = FixedFeatureAcquisitionFunction(
            PosteriorMean(model), d=BOUNDS.shape[1], fixed_features=TARGET_FIDELITIES_DICT
        )(train_x.unsqueeze(-2)).max().item()

        mfkg = qMultiFidelityKnowledgeGradient(
            model=model,
            num_fantasies=NUM_FANTASIES,
            cost_aware_utility=cost_aware_utility,
            project=project_func,
            current_value=torch.tensor(current_value, device=DEVICE, dtype=torch.float64),
        )

        candidate, _ = optimize_acqf(mfkg, bounds=BOUNDS, q=1, num_restarts=20, raw_samples=512)
        new_y = evaluate_config(candidate.squeeze(0))
        new_y_tensor = torch.tensor([[new_y]], device=DEVICE, dtype=torch.float64)

        train_x = torch.cat([train_x, candidate])
        train_y = torch.cat([train_y, new_y_tensor])

        if new_y > best_y:
            best_y = new_y
            print(f"*** NEW BEST: {best_y:.3f} ***")

    # Final exploitation: find predicted best at full fidelity
    final_model = SingleTaskMultiFidelityGP(train_x, train_y, data_fidelity_dim=FIDELITY_DIM_INDEX)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(final_model.likelihood, final_model))

    best_candidate, _ = optimize_acqf(
        FixedFeatureAcquisitionFunction(PosteriorMean(final_model), d=BOUNDS.shape[1], fixed_features=TARGET_FIDELITIES_DICT),
        bounds=BOUNDS[:, :HYPER_DIM],
        q=1,
        num_restarts=60,      # Increased for better coverage
        raw_samples=2048      # Increased for better coverage
    )

    best_config = config_from_tensor(best_candidate.squeeze(0))

    # Checkpointing
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'train_x': train_x.cpu(),
        'train_y': train_y.cpu(),
        'best_observed_y': best_y
    }, 'checkpoints/mfkg_bo_history.pt')
    with open('checkpoints/mfkg_best_hyperconfig.pkl', 'wb') as f:
        pickle.dump(config_to_dict(best_config), f)
    print("\nSaved BO history to checkpoints/mfkg_bo_history.pt")
    print("Saved best recommended config to checkpoints/mfkg_best_hyperconfig.pkl")

    print("\n=== TUNING COMPLETE ===")
    print(f"Best observed strict rate: {best_y:.3f}")
    print("Recommended hyperparameters (paste into physics_audio.py):")
    for field in HYPER_FIELDS:
        print(f"  {field} = {getattr(best_config, field):.6f}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    run_botorch_tuning()