# evaluator.py
import numpy as np

def print_multi_seed_summary(results, num_seeds):
    if not results:
        return

    strict_successes = [r for r in results if r['strict_success']]
    loose_successes = [r for r in results if r['loose_success']]

    strict_rate = len(strict_successes) / num_seeds * 100
    loose_rate = len(loose_successes) / num_seeds * 100

    avg_jumps_all = np.mean([r['jumps'] for r in results])
    avg_time = np.mean([r['wall_time'] for r in results])

    print("\n=== Multi-Seed Statistics Summary ===")
    print(f"Strict success rate: {strict_rate:.2f}% ({len(strict_successes)}/{num_seeds})")
    print(f"Loose success rate: {loose_rate:.2f}% ({len(loose_successes)}/{num_seeds})")
    print(f"Jumps avg: {avg_jumps_all:.2f}")
    print(f"Average wall time per seed: {avg_time:.1f}s")

    if strict_successes:
        print(f"\nAverages over strict successes:")
        print(f"  Geo dist: {np.mean([r['final_geo_dist'] for r in strict_successes]):.6f}")
        print(f"  Max B: {np.mean([r['max_inharm_b'] for r in strict_successes]):.8f}")
        print(f"  Speed std: {np.mean([r['speed_rel_std'] for r in strict_successes]):.6f}")
        print(f"  Damping RMSE:  {np.mean([r['damping_rmse'] for r in strict_successes]):.4f}")
        print(f"  Freq RMSE: {np.mean([r['freq_rmse'] for r in strict_successes]):.4f}")

        # print(f"  Raw_lin_b: {np.mean([r['raw_lin_b'] for r in strict_successes]):.2f}")
        # print(f"  Raw_quad_b: {np.mean([r['raw_quad_b'] for r in strict_successes]):.2f}")