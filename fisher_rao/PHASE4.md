# Fisher-Rao Phase 4 — replace redundant MSE priors

When Fisher-Rao invariants are active, drop the overlapping MSE prior terms in
`prior_loss` so physics constraints are not double-counted.

## Skipped MSE terms

| Prior term | Skipped when |
|------------|--------------|
| `damping_prior_loss` | `fr_invariant_damping > 0` and damping coeffs present |
| `coupling_prior_loss` | `fr_invariant_coupling > 0` and `coupling_skew` present |
| `speed_mean_prior` | `fr_invariant_speed > 0` |
| `inharm_l2_loss` | `fr_invariant_inharm > 0` |

**Kept regardless:** `speed_uniform_loss`, `inharm_ceiling_loss` (no matching invariant).

Replacement is gated by `fr_invariant_weight > 0` **and** `fr_replace_mse_priors=True`.

## Config

| Key | Default |
|-----|---------|
| `fr_replace_mse_priors` | `True` |

## Smoke test A/B

`run_fr_smoke_test.py` compares:

- `phase3_full` — all invariants + Phase 4 replace (default)
- `phase3_augment` — same invariants but MSE priors retained (legacy mode)

## Trainer

```python
run_single_seed(
    seed=42,
    fr_invariant_weight=0.3,
    fr_replace_mse_priors=True,  # default
)
```

## Real audio CLI

```bash
# Default: invariants replace overlapping MSE priors
python run_real_audio.py --audio note.wav

# Legacy: keep MSE priors alongside invariants
python run_real_audio.py --audio note.wav --fr_augment_priors
```

Derivation: redundant Euclidean priors are superseded once the double-coset /
Fisher-Rao invariant losses encode the same symmetries (Nielsen & Okamura 2026).