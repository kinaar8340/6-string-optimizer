# Fisher-Rao Phase 2 — coupling & profile invariants

Extends Phase 1 with rotation- and scale-invariant losses on string physics parameters.

## Components

| Invariant | Module | Symmetry |
|-----------|--------|----------|
| Coupling skew spectrum | `coupling_invariant_loss` | O(k) rotation on mode index |
| Coupling strength | log-ratio in `coupling_invariant_loss` | positive scale |
| Speed profile | `speed_profile_invariant_loss` | global scale quotient + FR distance |
| Inharmonicity profile | `inharm_profile_invariant_loss` | scale quotient + FR distance |

Phase 1 (damping double-coset) remains under `fr_invariant_weight` × `fr_invariant_damping`.

## Config

| Key | Default |
|-----|---------|
| `fr_invariant_coupling` | 0.5 |
| `fr_invariant_coupling_strength_weight` | 0.1 |
| `fr_invariant_speed` | 0.25 |
| `fr_invariant_inharm` | 0.25 |

## Synthetic references

Trainer and smoke test pass ground-truth references when available:

- `target_coupling_skew` — from synthetic generator or `prior_targets["coupling_skew"]`
- `target_speed_scalars` — uniform `VELOCITY_SCALE_BASE` per mode
- `target_inharm_b` — `TRUE_INHARM_B` or estimated from audio

## Run

```bash
cd physics_audio
python run_fr_smoke_test.py --steps 1200
```

## Trainer

```python
run_single_seed(
    seed=42,
    max_steps=5000,
    fr_invariant_weight=0.3,
    fr_invariant_coupling=0.5,
    fr_invariant_speed=0.25,
    fr_invariant_inharm=0.25,
)
```

## Real audio CLI

```bash
python run_real_audio.py --audio note.wav \
  --fr_invariant_coupling 0.5 \
  --fr_invariant_speed 0.25 \
  --fr_invariant_inharm 0.25
```

`build_prior_targets` supplies `speed_scalars` (uniform at f0-derived scale) and
`inharm_b` (estimated B per harmonic) as references.

Derivation: `~/Projects/Fisher_Rao/proof.pdf` (Thm 2.3, Prop 3.3 — singular values as maximal invariants).