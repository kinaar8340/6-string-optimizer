# Fisher-Rao Package (vendored)

Derivation reference: Nielsen & Okamura (2026), *Group invariance of f-divergences and the Fisher–Rao distance* — see `~/Projects/Fisher_Rao/proof.pdf`.

## Theorem-to-code map

| Result | Implementation |
|--------|----------------|
| Thm 2.1 — f-divergence group invariance | `losses.fisher_rao_loss`, `losses.hellinger_loss` |
| Thm 2.3 — double coset maximal invariant | `invariants.location_scale_pair_invariant` |
| Prop 3.3 — SVD singular values + block norms of \(U^\top\nu\) | `LocationScaleMaximalInvariant` |
| Prop 4.2 — FR canonical reduction | `invariants.fisher_rao_canonical_pair` |
| Phase 3 — modal amp pair invariant | `physics_audio/.../modal_amp_pair_invariant_loss` |
| Simplex FR metric (sqrt-sphere) | `metrics.sqrt_embed`, `metrics.fisher_rao_distance` |
| Fisher-info burst modulation | `burst_modulation.FisherInfoBurstModulator` |