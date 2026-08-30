# Phase 6.12 decision: Watts–Strogatz small-world network

## Model

The generator implements the Watts–Strogatz construction. An even
`neighbor_degree = k` creates a cyclic lattice in which every site initially
has `k / 2` neighbors in each direction. Each original forward ring edge is
considered exactly once in deterministic source/positive-offset order.

With probability `beta`, that edge is removed and its source chooses uniformly
among currently valid targets. Self-edges and duplicate edges are impossible.
The removed original target is a valid candidate, so a rewiring decision can
leave the topology unchanged. Metadata therefore distinguishes the number of
random rewiring decisions from the number of actually changed edges.

## Preserved and non-preserved properties

Rewiring preserves the exact edge count `n * k / 2` and hence mean degree `k`.
It does not preserve every individual degree. Connectivity is not repaired or
conditioned after generation because doing so would change the requested
Watts–Strogatz distribution.

The cyclic node order defines network construction but is not a physical
spatial embedding. Results therefore have no coordinates, boundary sites, site
types, or dimension records. Unchanged edges are typed `ring_lattice`; actual
shortcuts are typed `rewired`.

## Reproducibility and resources

The generator requires an explicit nonnegative seed and uses a named NumPy
PCG64 stream. Canonical final edges are sorted. The invariant edge count is
checked against `max_edges` before construction, and the chosen budget and
rewiring statistics are retained in immutable metadata.
