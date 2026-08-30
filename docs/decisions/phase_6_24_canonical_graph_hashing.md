# Phase 6.24: Canonical graph hashing

## Status

Accepted.

## Context

Later dataset construction must group graph-isomorphic geometry candidates so
that duplicates do not leak across data splits. Hashing site-indexed edge lists
or serialized NPZ bytes would incorrectly distinguish the same graph after a
site permutation, a different edge order, or reversed reference orientations.

General exact graph canonization is substantially more expensive than color
refinement on difficult symmetric graph families. One-dimensional
Weisfeiler--Leman (1-WL) refinement is invariant under graph isomorphism but is
not a complete isomorphism test. In particular, distinct regular graphs can
remain indistinguishable.

## Decision

`canonical_graph_hash` returns a self-identifying
`wl1-v1-sha256:<64 hex digits>` fingerprint. Initial site colors are refined
from multisets of neighboring colors and optional edge labels. Every color ID
is assigned from sorted structural signatures, never from site indices. The
SHA-256 input also records graph size, edge count, sorted connected-component
sizes, refinement traces, stabilization state, and the final colored-edge
multiset.

Refinement stops when the color partition stabilizes or after the configurable
`max_refinement_rounds`, which defaults to 64. A truncated result remains
isomorphism-invariant but may create more candidate collisions.

The default scope hashes only the undirected one-skeleton. It ignores
coordinates, displacements, faces, dimension records, rooted-tree root choice,
metadata, edge order, and reference orientation. Callers may explicitly add:

- `include_site_types` for per-site type labels;
- `include_edge_types` for undirected edge-type labels;
- `include_boundary` for site boundary membership and edge crossing markers.

The selected scope flags are part of the hash domain, preventing accidental
comparison of fingerprints with different semantics.

## Collision policy

Hash equality creates an isomorphism-candidate bucket; it never proves that two
graphs are isomorphic. Tests retain the known collision between the triangular
prism and `K3,3`, two non-isomorphic connected 3-regular six-site graphs. Later
duplicate detection must run an exact attributed or unattributed isomorphism
check, matching the selected scope, before deleting data or assigning splits.

## Consequences

- Isomorphic relabelings always receive the same hash under the same scope and
  refinement budget.
- Common non-isomorphic graphs are separated efficiently, while collision risk
  remains explicit and test-enforced.
- Embedding changes do not hide graph-topology duplicates.
- The versioned prefix permits future stronger fingerprints without silently
  mixing incompatible hashes.
