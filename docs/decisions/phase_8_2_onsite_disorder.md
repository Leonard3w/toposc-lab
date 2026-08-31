# Phase 8.2: Onsite disorder

## Decision

Uniform onsite disorder is an additive scalar potential sampled independently once per
physical site from `[-width / 2, width / 2]`. The concrete operation lives in
`toposc_lab.robustness.onsite` and executes through the Phase-8.1
`realize_disorder(...)` contract. It therefore requires an explicit seed, receives the
single named NumPy PCG64 stream owned by that executor, and returns a read-only
Hamiltonian snapshot with separate disorder provenance.

The existing `uniform_site_disorder(...)` map API remains stable. Its sampling work is
shared through `sample_uniform_site_disorder(...)`, which accepts an already-created
NumPy generator and never creates or reads random state. The legacy seeded function and
the new Phase-8 path consequently use exactly the same distribution and reference stream
rather than maintaining parallel stochastic implementations.

## Basis and provenance semantics

For an undoubled Hamiltonian, basis order is explicitly site-major and one offset is
added to every internal component at a site. For a Hamiltonian in an explicit
`NambuBasis`, the same offset is added to each particle component and subtracted from its
hole partner. This preserves the repository's declared unrotated particle-hole basis
convention. No basis convention is inferred from matrix dimension, geometry appearance,
or model name.

The disorder parameters record the uniform width, exact Phase-7.11 geometry snapshot ID,
site count, representation, internal-component count, basis ordering, and particle-hole
embedding rule. The Phase-8.1 provenance separately records the exact source and result
Hamiltonian IDs, seed, RNG algorithm, and transform key/version. The geometry ID remains
representation-sensitive and is not a physical deduplication key.

This phase does not modify a model, run `evaluate_geometry(...)`, execute ensembles, or
define hopping, pairing, removal, coordinate, or physical-parameter disorder.
