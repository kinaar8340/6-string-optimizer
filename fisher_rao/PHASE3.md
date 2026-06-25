# Fisher-Rao Phase 3 — piptrack modal amp pair invariant

Extends Phase 2 with Prop 3.3 / 4.2 location–scale pair reduction on log-modal
amplitude profiles from piptrack (or Stiefel RMS energy on synthetic data).

## Components

| Piece | Module | Role |
|-------|--------|------|
| Pair invariant loss | `modal_amp_pair_invariant_loss` | FR canonical pair on log-amp (μ, V) |
| Target resolution | `resolve_target_mode_amps` | piptrack mean → override → Stiefel RMS |
| Temporal obs | `partial_amps_temporal` in `build_prior_targets` | per-frame piptrack amplitudes (T, K) |
| Predicted amps | `mode_amplitudes_from_stiefel` | per-mode RMS energy from model preds |

Uses V1 = V2 = I_K on log-mean modal profiles (same construction as the
damping coefficient invariant).  Piptrack `(T, K)` observations collapse to
per-mode temporal means for μ₂.

## Config

| Key | Default |
|-----|---------|
| `fr_invariant_modal` | 0.4 |
| `REAL_AUDIO_FR_MODE_WEIGHT` | 0.2 (separate simplex FR loss, not the pair invariant) |

Phase 4 (`fr_replace_mse_priors=True`) drops redundant MSE priors when matching
invariants are active; use `--fr_augment_priors` on real audio to keep both.

## Synthetic references

Smoke test and trainer use Stiefel RMS energy from observations as the modal
observation when piptrack is unavailable.

## Run

```bash
cd physics_audio
python run_fr_smoke_test.py --steps 1200
```

Arms: baseline → phase1 (damping) → phase2 (coupling/speed/inharm) → phase3 (+ modal).

## Trainer

```python
run_single_seed(
    seed=42,
    max_steps=5000,
    fr_invariant_weight=0.3,
    fr_invariant_coupling=0.5,
    fr_invariant_speed=0.25,
    fr_invariant_inharm=0.25,
    fr_invariant_modal=0.4,
)
```

## Real audio CLI

```bash
python run_real_audio.py --audio note.wav \
  --fr_invariant_modal 0.4 \
  --fr_mode_weight 0.2
```

`build_prior_targets` stores `partial_amps_mean` and `partial_amps_temporal` from
piptrack; `total_loss` prefers temporal obs for the pair invariant when present.

Derivation: `~/Projects/Fisher_Rao/proof.pdf` (Props 3.3, 4.2).