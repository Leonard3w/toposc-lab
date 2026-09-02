# Phase 10.1 — Geometry genome representation

## Status and scope

Accepted as the Phase-10.1 architecture for a lossless, model-independent
geometry genome.

This step adds only the representation, conversion boundary, and
representation-level validation needed to carry an existing `Geometry` snapshot
into later geometry-search work. It adds no mutation, population, fitness,
selection, elitism, crossover, generation loop, checkpoint, benchmark, or
learned model.

## Scientific purpose

The research charter asks whether constrained inverse connectivity design can
discover resource-matched, nonperiodic, spatially embedded planar graph families
with stronger independently validated class-D robustness. A genome is therefore
an engineering representation of a candidate geometry, not scientific evidence
and not a source of physical semantics.

Later search operators may act on this representation, but the genome itself
contains no score, model parameter, disorder outcome, topology result, or claim
of novelty. Abstract graphs remain representable for general infrastructure,
while the initial scientific program remains restricted by its separate
two-dimensional spatial-applicability and candidate-space contracts.

## Separation from `Geometry`

`Geometry` remains the production object used by generators, Hamiltonians,
validation, serialization, hashing, and evaluation. It owns derived neighbor and
edge lookups plus spatial helper methods.

`GeometryGenome` lives in `toposc_lab.search.geometry_genome`. It is an immutable,
cache-free state record. It deliberately reuses the existing immutable leaf
records (`GeometryEdge`, boundary components, dimension records, rooted-tree
structure, and faces) so those concepts do not acquire competing definitions.
Conversion always constructs a separate `Geometry` instance.

## Lossless representation contract

The round trip `Geometry -> GeometryGenome -> Geometry` preserves:

- site count and site-index convention;
- edge sequence and every stored source-to-target orientation;
- edge type, boundary-crossing marker, displacement, and metadata;
- exact float64 coordinate data, including signed zero;
- declared embedding dimension;
- boundary sites and boundary components;
- site types and sourced dimension records;
- rooted-tree structure;
- geometry metadata and ordered faces.

No endpoint is reordered and no displacement is recomputed. Coordinates,
boundaries, dimensions, faces, or physical meaning are never inferred when they
are absent.

Purely abstract graphs retain both `coordinates=None` and
`embedding_dimension=None`. Embedded geometries retain their existing coordinate
axis count, including dimensions above three. Embedding dimension remains
distinct from intrinsic, topological, fractal, spectral, or other sourced
`dimension_records`.

## Validation boundary

`validate_geometry_genome` checks only whether the stored genes can materialize a
`Geometry` representation. It reports representation failures through the
dedicated genome validation types.

After materialization, `geometry_from_genome` separately calls the existing
`validate_geometry` production API. Generator provenance, model compatibility,
Phase-9.8 clean constraints, scientific evidence gates, and future mutation
constraints are not genome-validation rules.

## Identity boundary

Tests require a valid round trip to retain the existing serialized bytes and
therefore the existing exact geometry snapshot ID. This does not make that ID a
canonical physical identity: it remains representation-sensitive.

The canonical graph hash is also checked across the round trip. It remains only
a relabeling-invariant isomorphism-candidate fingerprint; equality is not an
isomorphism proof and the genome does not use it for deduplication.

## Deferred work

All mutation operators and their validity policies begin no earlier than their
separately numbered Phase-10 steps. In particular, Phase 10.1 defines no implicit
edge canonicalization, dimension-changing action, resource budget, physical
planarity rule, or optimization objective.
