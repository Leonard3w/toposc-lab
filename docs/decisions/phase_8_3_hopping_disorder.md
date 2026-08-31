# Phase 8.3: Hopping disorder

## Decision

Uniform hopping disorder is an additive real scalar offset sampled independently once
per stored `GeometryEdge` from `[-width / 2, width / 2]`. The operation lives in
`toposc_lab.robustness.hopping` and executes through the Phase-8.1 reproducible disorder
contract with a mandatory seed and its single recorded NumPy PCG64 stream.

The established `uniform_edge_disorder(...)` mapping API remains stable. Its stochastic
work is shared through `sample_uniform_edge_disorder(...)`, which receives an existing
generator. Both paths therefore use the same distribution, edge order, and reference
stream without a parallel randomness implementation.

## Orientation, basis, and provenance

Every sampled mapping retains the geometry's exact immutable edge objects and iteration
order. The forward hopping block follows `GeometryEdge.source -> GeometryEdge.target`;
the reverse block receives the Hermitian-conjugate contribution. No endpoint sorting,
graph isomorphism fingerprint, or physical deduplication identifier is used. For the
current real scalar distribution, the two numerical contributions are equal, but the
stored orientation remains the contract for future direction-sensitive physics.

Normal-state matrices use explicit site-major order and receive a scalar identity offset
in internal-component space. With an explicit `NambuBasis`, the normal hopping
perturbation is added to the particle sector and its negative complex conjugate is added
to the hole sector. Onsite and hopping disorder share one internal basis-validation
contract; neither infers layout from matrix size, model name, or geometry appearance.

Provenance records width, distribution, exact Phase-7.11 geometry snapshot ID, site and
edge counts, representation, component count, basis ordering, particle-hole embedding,
and the oriented/Hermitian edge-application rule. Phase-8.1 separately records the exact
source/result Hamiltonian IDs, transform version, seed, and RNG algorithm. The geometry
ID remains representation-sensitive and is not a canonical physical identity.

This phase does not implement pairing disorder, structural removal, coordinate or model-
parameter perturbation, ensemble execution, robustness metrics, or scientific analysis.
