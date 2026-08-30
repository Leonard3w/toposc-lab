# Phase 6.15 decision: finite Cayley / Bethe-like tree

## Coordination and shells

`coordination = z` denotes Bethe coordination. For at least one shell, the root
has `z` children. Each later nonleaf site has one parent and `z - 1` children,
so every nonleaf site has graph degree `z`. `shells` counts generations outside
the root; `shells = 0` is a singleton.

Shell zero contains one site. Shell `s >= 1` contains
`z * (z - 1)**(s - 1)` sites. For `z = 2`, the construction is a centered path
with two arms and `1 + 2 * shells` sites.

## Hierarchy and boundary

Sites are generated breadth-first by shell and passed through the general
Phase-6.14 `tree` generator. The result therefore carries the same validated
`RootedTreeStructure`, `tree_branch` edges, and `root` / `internal` / `leaf`
site types. The outermost shell is exactly `boundary_sites`; for a singleton,
the root is the boundary leaf.

The graph is abstract. No spatial coordinates or dimension records are
inferred.

## Resource guard

Cayley trees grow exponentially for `z > 2`. Shell populations and cumulative
site count are computed before any Parent, Edge, or Geometry allocation. The
calculation stops as soon as it exceeds `max_sites`; the default is 25,000.
Callers may explicitly raise the limit or pass `None`. The shell populations,
coordination convention, construction order, and chosen budget are retained in
immutable metadata.
