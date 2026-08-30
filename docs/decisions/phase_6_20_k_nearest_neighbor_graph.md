# Phase 6.20 decision: k-nearest-neighbor geometry builder

## Directed selection contract

`k_nearest_neighbor_graph(coordinates, k)` accepts the same finite arbitrary-
dimension point-cloud contract as the coordinate-cutoff builder. Every site
ranks all distinct site indices by the pair `(Euclidean distance, site index)`
and selects exactly `k` entries. The ascending-index tie break makes symmetric
and coincident point sets deterministic without an implicit tolerance.

`cKDTree` obtains an initial kth-distance threshold. A radius query includes all
points tied at that threshold, then exact Euclidean distances and site indices
determine the final rank. The tree is therefore a candidate index rather than
the authority for tie semantics.

## Undirected symmetrization and provenance

`Geometry` is an undirected simple graph, while nearest-neighbor selection is
directed. The public `symmetrization` parameter makes this conversion explicit:

- `union` retains an edge when either endpoint selected the other and is the
  default;
- `mutual` retains an edge only when both endpoints selected one another.

Every edge records `selected_by` and `mutual` metadata, so union conversion does
not discard the original directed-selection provenance. Edges remain ordered by
their canonical lexicographic site pair.

## Degeneracies and resources

Distinct coincident sites remain distinct and participate at zero distance,
with the same site-index tie break. `k` must be positive and smaller than the
number of sites.

The exact directed work size `n_sites * k` is checked against
`max_neighbor_selections` before building the tree. `max_edges` independently
limits the symmetrized graph. Both budgets can be explicitly disabled.

## No inferred semantics

The builder reuses the private point-cloud validation introduced in phase 6.19.
It assigns no boundary, faces, site types, rooted hierarchy, translation rank,
or topological dimension. Models remain responsible for mapping
`k_nearest_neighbor` edges to physical couplings.
