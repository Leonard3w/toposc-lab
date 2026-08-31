# Phase 8.4: Pairing disorder

## Decision

Pairing disorder is an additive real uniform amplitude perturbation in one explicitly
selected existing pairing channel. `PairingDisorderChannel` supports the repository's
spinless oriented p-wave, onsite spin-singlet s-wave, chiral p-wave, and d-wave builders.
No channel, spin content, spatial plane, chirality, or basis convention is inferred from
the model name, graph, embedding dimension, or matrix size.

Site-supported s-wave disorder reuses the Phase-8.2 site sampler. Edge-supported p-wave,
chiral p-wave, and d-wave disorder reuse the Phase-8.3 edge sampler and therefore retain
the exact stored `GeometryEdge` objects and order. The Phase-8.1 executor remains the
sole owner of the mandatory NumPy PCG64 stream.

## Physical embedding and applicability

Sampled amplitudes are passed through the corresponding established pairing builder,
which produces an antisymmetric normal-state pairing block. Spinless and chiral p-wave
require exactly one normal component per site; onsite s-wave and d-wave require exactly
two. Chiral p-wave additionally requires explicit chirality and plane axes. Chiral and
d-wave applicability is validated by the existing direction-aware builders, so abstract
graphs without coordinates or edge displacements receive no invented spatial physics.

The pairing perturbation is embedded into an explicitly supplied `NambuBasis` using the
repository convention

```text
delta H_BdG = [[0, delta Delta], [-delta Delta*, 0]].
```

This preserves fermionic antisymmetry, Hermiticity, and the declared particle-hole
convention. Edge orientation is physically retained: spinless and chiral p-wave use the
stored source-to-target amplitude and its negative under reversal without complex
conjugation.

Provenance records the channel, site/edge support, width, exact Phase-7.11 geometry ID,
basis ordering and component count, chirality and plane where applicable, and the BdG
embedding rule. Phase-8.1 separately records source/result Hamiltonian IDs, seed, RNG,
and transform version. No ensemble, robustness metric, structural removal, coordinate
perturbation, or parameter perturbation is introduced in this phase.
