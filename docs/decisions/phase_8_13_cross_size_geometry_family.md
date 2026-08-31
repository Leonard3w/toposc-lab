# Phase 8.13: Cross-size geometry family handling

## Decision

Phase 8.13 binds the numerical points of one Phase-8.12 scaling result to explicit,
protocol-generated geometry snapshots. Family membership is declared and validated from
construction provenance; it is never inferred from graph similarity, coordinates,
embedding dimension, site count, or a fitted robustness trend.

`GeometryFamilySpec` records a stable caller-supplied family key and version, human
description, the matching scaling size key, Phase-6 generator key and algorithm version,
top-level generator parameters allowed to vary, and a geometry-generation seed policy.
The supported policies distinguish deterministic generation with no seed, one common
seed across sizes, and distinct explicit seeds per size. Geometry-generation seeds are
separate from Phase-8 disorder ensemble seeds.

`GeometryFamilyMember` retains an unchanged `Geometry`, its exact associated
`FiniteSizeRobustnessPoint`, reconstructed immutable `GeometryGenerationRequest`, and the
exact Phase-7.11 geometry snapshot ID. Protocol generation metadata is mandatory. Direct
legacy generator calls without that provenance are rejected for cross-size family
claims rather than receiving invented construction history.

`CrossSizeGeometryFamily` requires one member per scaling point in exact order, the same
declared generator key and version, identical non-varying generator parameters, a unique
tuple of declared varying parameters at every size, a matching seed policy, and distinct
exact snapshot IDs. It exposes raw site counts only for auditing and never substitutes
them for the explicitly declared Phase-8.12 size variable.

## Identity and scientific boundaries

The Phase-6 Weisfeiler--Lehman graph hash is not used. It remains only an isomorphism
candidate fingerprint and cannot establish family membership or physical equivalence.
The Phase-7.11 ID identifies each exact representation, including oriented edges,
coordinates, metadata, and other serialized geometry state. Here it detects reuse of the
same snapshot and preserves auditability; it is not a canonical family ID or a physical
deduplication identity. Family keys are explicit versioned declarations, not hashes.

The contract does not rerun a generator, prove that random families are nested or
statistically coupled, or show that the declared size variable and varying parameters
define a scientifically meaningful thermodynamic sequence. It also cannot prove from a
Phase-8.12 aggregate that every underlying model, disorder definition, Hamiltonian, or
evaluation used the bound geometry; those provenance layers remain separate.

A common construction contract is necessary evidence, not proof of thermodynamic
topological behavior. One unusual finite graph cannot establish such behavior, and even
several protocol-compatible graphs still require a justified family construction and
scaling model. Phase 8.13 does not create the Phase-8.14 robustness report, infer
topology, rank candidates, begin search, or generate datasets. It uses no RNG.
