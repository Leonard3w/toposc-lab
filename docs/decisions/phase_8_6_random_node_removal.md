# Phase 8.6: Random node removal

## Decision

Random node removal is an independent Bernoulli transformation of an immutable
`Geometry`. `apply_random_node_removal(...)` requires a probability in `[0, 1]` and
an explicit seed, then consumes one NumPy PCG64 draw per original site in ascending
site-index order through the Phase-8.1 executor. A site is selected for removal exactly
when its draw is below the recorded probability.

A valid `Geometry` cannot contain zero sites. If every site is selected for removal, the
site with the largest draw survives; an exact tie is resolved in favor of the lowest
original site index. This minimum-survivor rule is part of the versioned algorithm and
is recorded in provenance. It deliberately conditions the otherwise independent sample
only in the invalid empty-geometry case.

## Reindexing and geometry invariants

Survivors retain their original relative order and receive contiguous indices starting
at zero. `NodeRemovalRealization` keeps the common `DisorderRealization` separate from
the explicit `surviving_sites`, `removed_sites`, and immutable `old_to_new` mapping.
This makes the relationship auditable without writing disorder data into geometry
metadata or scientific evaluation results.

An edge survives only when both endpoints survive. Its remapped source and target keep
the stored physical orientation, while edge order, type, boundary-crossing flag,
displacement, and metadata are preserved. Coordinates and site types are selected in
survivor order. Boundary sites are remapped; boundary components are remapped and
dropped only when they become empty. A face survives only when all its sites survive,
and its ordered boundary is remapped without changing orientation. Geometry metadata
and declared dimension records remain attached as source annotations; no dimension is
inferred or recomputed.

Any actual site removal clears `RootedTreeStructure`, because its parent array and exact
parent-edge set no longer describe the result. No replacement hierarchy or connectivity
requirement is inferred. Whether the resulting geometry is compatible with a physical
model remains the responsibility of the explicit Phase-7 validation pipeline.

Provenance records the probability, original geometry ID, input site and edge counts,
sampling order, minimum-survivor and reindexing rules, and the edge, face, boundary, and
rooted-tree policies. Phase 8.1 additionally records the exact source/result snapshot
IDs, seed, PCG64 algorithm, and transform version. Snapshot IDs remain exact and
representation-sensitive; the Phase-6 graph hash remains only an isomorphism-candidate
fingerprint.

This phase does not perturb coordinates or model parameters, execute ensembles,
calculate robustness metrics or uncertainty, perform finite-size analysis, or begin any
Phase-9 work.
