# run_multi.py
#

import os
from training_evaluation.trainer import run_single_seed
from training_evaluation.evaluator import print_multi_seed_summary
from training_evaluation.config import NUM_SEEDS, NOISE_AMP, FORCE_PUNCTUATED, DATA_SEED
from shutil import copyfile
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("configs", exist_ok=True)
# copyfile("training_evaluation/config.py", f"configs/config_{timestamp}_noise{NOISE_AMP:.2f}.py")

if __name__ == "__main__":
    effective_punctuated = (NOISE_AMP > 0.0) if FORCE_PUNCTUATED is None else FORCE_PUNCTUATED
    print(f"\nMulti-seed sweep: {NUM_SEEDS} seeds | noise_amp={NOISE_AMP} | Punctuated: {'ON' if effective_punctuated else 'OFF'}\n")

    results = []
    for i in range(NUM_SEEDS):
        res = run_single_seed(DATA_SEED + i, noise_amp=NOISE_AMP, force_punctuated=FORCE_PUNCTUATED)
        results.append(res)

    print_multi_seed_summary(results, NUM_SEEDS)