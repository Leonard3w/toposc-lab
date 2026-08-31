# Phase 8.5: Random edge removal

## Decision

Random edge removal is an independent Bernoulli transformation of an immutable
`Geometry`. `apply_random_edge_removal(...)` requires an explicit probability in
`[0, 1]` and seed, then consumes one NumPy PCG64 draw per stored edge in exact
`geometry.edges` order through the Phase-8.1 executor. An edge is removed exactly when
its draw is below the recorded probability.

Surviving entries are the original immutable `GeometryEdge` objects in their original
order. Their source-to-target orientation, displacement, boundary-crossing flag, edge
type, and metadata are therefore retained exactly. No endpoint sorting, graph hash,
isomorphism fingerprint, or physical deduplication identity participates in sampling.

## Geometry invariants

Sites are not removed or reindexed. Coordinates, boundary sites and components, site
types, dimension records, and geometry metadata retain their source values. Connectivity
is not imposed: whether a disconnected realization is physically admissible belongs to
the explicit `ModelGeometryRequirements` used by the Phase-7 evaluation pipeline.

A `GeometryFace` is retained only when all of its boundary edges survive; otherwise the
face must be dropped to keep the output geometry structurally valid. Retained faces are
the original immutable objects. If any edge is removed from a geometry carrying
`RootedTreeStructure`, that hierarchy is cleared because its required exact parent-edge
set no longer describes the result. Site labels or a replacement hierarchy are not
inferred.

Provenance records the probability, original geometry ID, site and input-edge counts,
edge iteration and survival rules, and face/rooted-tree policies. Phase-8.1 separately
records the exact source and result geometry snapshot IDs, seed, RNG algorithm, and
transform version. These IDs remain representation-sensitive snapshots, not canonical
physical identities.

This phase does not remove or reindex nodes, perturb coordinates or model parameters,
run ensembles, compute robustness metrics, or begin any later master-plan step.
