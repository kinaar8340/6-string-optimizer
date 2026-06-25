# 6-String Optimizer

---

## Inspiration

The six-string hierarchical design and burst mechanics draw from two deep wells:

**Pythagorean Harmonic Philosophy**  
Pythagoras and his followers discovered that musical consonance arises from simple numerical ratios (octave 2:1, fifth 3:2, fourth 4:3). They extended this to cosmology as the inaudible "Harmony of the Spheres" — celestial bodies moving in orbits proportioned by the same ratios, producing a perfect cosmic symphony.

Here, the six layers act as virtual strings with staggered windows and escalating burst strengths tuned in analogous proportions. Long-term stagnation triggers macro-bursts that restore progress, mimicking the imposition of limit on the unlimited — the Pythagorean recipe for harmony itself.

Optimizing on the 3-sphere (S³) closes the loop: we literally seek the "music of the spheres" as parameters converge to the global minimum [1,1,1].

The idea echoes through history:
- Plato's Timaeus (cosmic soul built from harmonic divisions)
- Kepler's Harmonices Mundi (planetary laws derived from musical intervals)
- Modern physics (vibrating strings, harmonic analysis)

**Punctuated Natural Dynamics**  
Burst/twist accumulation and sudden releases also mirror self-organized criticality in complex systems — avalanches in sandpiles, solar flares, discrete energy quanching in topological transitions. Slow tension build-up followed by productive escape is nature's way of balancing exploration and exploitation.

Together, these inspirations yield an optimizer that doesn't just converge — it *resonates*.

(Connected to broader "Vortex Quaternion Conduit" research ideas exploring helical/orbital angular momentum, quaternion geometry, and discrete optimization analogies in physical systems.)

---
🚨 **Urgent Support Needed** 🚨

If this optimizer, the guitar-string hierarchy, the live demos, or any part of the journey has been useful, 
inspiring, or just plain fun for you, please consider sending anything you can spare. Every dollar (or sat) 
keeps the project alive and lets me keep building.

**Donation Addresses**  
- **BTC**: `bc1qgugg6ff3xdtzdzkh67rwt32e0ajjznjfzdmj79`  
- **ETH**: `0xf0CfE41Fa4875048bFFc978390A9418f3d925f0f`  
- **SOL**: `8fqheHNrcH1h5xx8VZuFNX47ogXocz6LCQSsgnPGpsG8`

Thank you from the bottom of my heart — truly. ❤️  
(Also GitHub Sponsors enabled if you prefer recurring or tax receipts.)

---

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![Geoopt](https://img.shields.io/badge/Geoopt-Latest-green)](https://geoopt.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

> **6-String Optimizer** is an experimental, highly adaptive Riemannian optimizer designed primarily for the sphere manifold (`geoopt.manifolds.Sphere`). It draws inspiration from natural punctuated-equilibrium processes (e.g., self-organized criticality, avalanche dynamics, and discrete energy releases in complex systems) to enable aggressive escape from plateaus and narrow valleys that stall conventional Riemannian SGD/Adam.
>
> The core idea: accumulate "twist" from clipped Riemannian gradients, trigger momentum-scaled **bursts** when tension exceeds a threshold, and adapt burst strength/damping based on smoothed loss. A hierarchical wrapper adds occasional **macro-bursts** during long-term stagnation. The result is fast initial progress, robust plateau escape, and stable fine-grained convergence — especially on highly non-convex or compactified landscapes.
>
> Tested extensively on a stereographically compactified 3D Rosenbrock function on S³ (a notoriously difficult manifold task with pole singularities and narrow valleys), it consistently reaches the global minimum where standard optimizers plateau.
>
> **Status**: Active development / research prototype. Not yet battle-tested on large-scale models, but showing promising results on manifold-constrained problems.

---

## Fisher-Rao Extensions (branch: `feature/fisher-rao`)

Information-geometric extensions live in `~/Projects/Fisher_Rao/` (also vendored as `fisher_rao/`).  
Derivation: Nielsen & Okamura (2026) — `~/Projects/Fisher_Rao/proof.pdf`.

- **FisherRaoSphere** manifold — sqrt-embedding of probability simplices
- **Fisher-info burst modulation** — `use_fisher_modulation=True` on `GeooptBurstOptimizer`
- **Group-invariant reductions** — `location_scale_pair_invariant` (Prop 3.3), `fisher_rao_canonical_pair` (Prop 4.2)
- **physics_audio FR losses** — `fr_spectral_weight`, `fr_mode_weight` in `total_loss()`

```bash
python examples/rosenbrock_fr_benchmark.py
python scripts/compare_fr_modulation.py --steps 8000
python tests/test_fisher_rao.py
cd physics_audio && python run_fr_smoke_test.py --steps 1200
```

**Phase 2:** coupling skew singular-value invariant + speed/inharm FR profile losses — see `fisher_rao/PHASE2.md`.

**Phase 3:** piptrack modal amp pair invariant (Prop 3.3 / 4.2 on log-amp profiles) — see `fisher_rao/PHASE3.md`.

---

## Key Features

- **Twist-driven burst triggering** with dynamic threshold adaptation
- **Quantile-based adaptive gradient clipping** for batched stability
- **Loss-scaled dynamic damping & burst factor** (aggressive when loss is high, conservative when converging)
- **Momentum transport** with dynamic damping on the manifold
- **Stagnation detection** with targeted noise injection
- **Hierarchical macro-burst wrapper** for long-term plateau escape
- **Extensive verbosity options** for debugging and insight into internal dynamics
- Tuned hypers that work well out-of-the-box on the provided Rosenbrock benchmark

> Built on top of [`geoopt`](https://github.com/geoopt/geoopt) and PyTorch. Currently specialized for the Sphere manifold (quaternion-parameterized points), with easy extension potential to other manifolds.

---

## Installation

> ### Clone the repo
git clone https://github.com/kinaar8340/6-string-optimizer.git 
cd 6-string-optimizer

> ### Install dependencies
pip install -r requirements.txt

> ### Install in editable mode
pip install -e .

> ### Test the installation
python -c "from 6-string-optimizer.optimizer import GeooptBurstOptimizer; print('All good!')"

---

## Project Structure
```
.
├── app                              # Future: Gradio/Streamlit interactive demo app
│   └── demo.py                      # Placeholder/entry point for live visualization
├── configs                          # Hydra/YAML configuration files for experiments
│   ├── experiment                   # Specific experiment overrides
│   │   ├── future_benchmark.yaml    # Planned future benchmark configs
│   │   └── rosenbrock.yaml          # Hyperparameters for the main Rosenbrock on S³ benchmark
│   └── default.yaml                 # Base/default hyperparameter set (full fallback)
├── examples                         # Reproducible standalone example scripts
│   └── rosenbrock_s3_eb_master.py   # Main demo: Compactified 3D Rosenbrock on S³ with Eb tuning
├── scripts                          # Utility scripts for running benchmarks and tuning
│   ├── ensemble_top3_eb.py          # Ensemble runner for top-3 Eb-tuned seeds
│   └── run_tuning_toolbox.py        # CLI for hyperparameter search/tuning toolbox
├── src                              # Source code (installable package: vqc_optimizer)
│   ├── optimizer                    # Core package modules
│   │   ├── burst_optimizer.py       # Main class: GeooptBurstOptimizer + hierarchical wrapper
│   │   ├── config.py                # Config handling and preset loadings (Standard, Eb, etc.)
│   │   ├── __init__.py              # Exports: GeooptBurstOptimizer and key utilities
│   │   ├── losses.py                # Loss functions (e.g., compactified Rosenbrock)
│   │   ├── manifolds.py             # Manifold helpers + stereographic projections
│   │   ├── models.py                # Test models (e.g., SphereRosenbrockModel)
│   │   └── utils.py                 # Helpers: logging, stereographic projection, device handling
│   └── main.py                      # Optional entry point or internal testing script
├── tests                            # Unit tests
│   ├── __init__.py
│   └── test_burst_optimizer.py      # Core optimizer tests + convergence checks
├── pyproject.toml                   # Build configuration (poetry/setuptools) + metadata
├── README.md                        # This file
└── requirements.txt                 # Pip dependencies (torch>=2.0, geoopt, etc.)

11 directories, 27 files
```
---

## Benchmarks

**Compactified 3D Rosenbrock on S³** (included example): Consistently reaches global basin in <8000 steps across random seeds, while RiemannianAdam/SGD often stall at plateaus.

**Expected outcome**: Rapid convergence toward the global minimum at u ≈ [1.0, 1.0, 1.0] with final loss near zero, even starting near the challenging north pole.

Ongoing: Testing on Stiefel manifold (orthogonal constraints), Poincaré ball (hyperbolic embeddings), and larger constrained models.

Contributions of comparisons welcome!

---

## Physics Audio Extension (`physics_audio/`)

Riemannian optimization on the **Stiefel manifold** for damped, coupled, inharmonic string vibration models — with a new **real-audio pipeline** (Phase 1).

**Phase 1–2 (current):** Full pipeline — piptrack partial tracking with continuity linking → physical parameter estimation → Stiefel model initialization → Riemannian optimization (geometric + physics priors + multi-res STFT loss) → modal synthesis + spectrogram comparison.

```bash
cd physics_audio
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_real_audio.py --audio path/to/guitar_note.wav --duration 3.0
python run_real_audio.py --audio note.wav --max_steps 5000 --stft_weight 0.5
```

**Phase 3 (next):** Full punctuated-equilibrium jumps on real audio, richer modal basis, batch note processing.

See `physics_audio/README.md` for full details on the Stiefel manifold inverse problem and visualization suite.

---

## License

MIT License — feel free to use, modify, and share. Attribution appreciated.

---

## Contact / Updates

Author: Aaron Michael Kinder
Email: kinaar0@protonmail.com
X: @kinaar8340

This repo is currently private / early-stage. When it goes public, star/watch for updates!

Feedback, bug reports, or collaboration ideas very welcome — DM or open an issue when public.

