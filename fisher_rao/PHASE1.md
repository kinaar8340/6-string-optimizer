# Fisher-Rao Phase 1 — physics_audio integration

## Scope

1. **Prop 3.3 damping invariant** — `invariant_losses.damping_invariant_loss` on `(log_base_rate, log_slope)` vs target damping envelope
2. **Modal Fisher-Rao loss** — `fr_mode_weight` + `mode_amplitudes_from_stiefel` in `total_loss`
3. **Spectral Fisher-Rao loss** — `fr_spectral_weight` + per-time mode energy envelope
4. **Smoke test** — `physics_audio/run_fr_smoke_test.py` (synthetic A/B)

## Config (`physics_audio/training_evaluation/config.py`)

| Key | Default | Purpose |
|-----|---------|---------|
| `fr_invariant_weight` | 0.3 | Damping double-coset invariant |
| `fr_mode_weight` | 0.0 | Modal amplitude FR loss |
| `fr_spectral_weight` | 0.0 | Per-frame mode spectral FR loss |

## Run smoke test

```bash
cd physics_audio
python run_fr_smoke_test.py --steps 1200 --seed 42
```

## Trainer usage

```python
run_single_seed(
    seed=42,
    max_steps=5000,
    fr_invariant_weight=0.3,
    fr_mode_weight=0.1,
    fr_spectral_weight=0.05,
)
```

Derivation: `~/Projects/Fisher_Rao/proof.pdf` (Props 3.3, 4.2).