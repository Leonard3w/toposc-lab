# Phase 6.18 decision: Ammann--Beenker tiling patch

## Tiling choice and construction

The first planar aperiodic tiling prototype is a finite radial patch of the
Ammann--Beenker tiling. It consists of unit-edge squares and 45-degree rhombi,
has octagonal quasiperiodic order, and uses the silver-mean inflation factor
`1 + sqrt(2)`. It therefore follows naturally from the phase 6.17 silver-mean
chain while adding genuinely planar aperiodic connectivity.

The generator uses a canonical cut-and-project construction. Integer points of
`Z^4` are projected into physical and internal two-dimensional spaces. A point
is retained when its internal projection lies inside the centered regular
octagonal acceptance window and its physical projection lies inside the radial
cutoff. Physical projection vectors are normalized to unit tile edges.

The construction and geometric conventions follow:

- https://doi.org/10.1002/ijch.202300119
- https://doi.org/10.1017/CBO9781139025256.008

## Explicit faces

A tiling cannot be represented faithfully by only its one-skeleton: a graph
does not preserve which cycles are tiles. `Geometry` therefore gains an
immutable `GeometryFace` value. A face stores an ordered polygon boundary, an
optional type, and immutable metadata. `Geometry` rejects faces with invalid
sites, repeated sites, duplicate vertex sets, or boundary segments not present
as graph edges.

The Ammann--Beenker patch stores every complete projected hypercube two-face as
either `square` or `rhombus_45_degree`. This keeps plaquette identity available
for later flux, local-marker, and tiling analysis without reconstructing it
from ambiguous graph cycles.

## Boundary and dimensions

The outer boundary is not inferred from finite graph degree. A retained vertex
is a boundary site exactly when it has a valid neighbor in the infinite
cut-and-project tiling that lies outside the radial cutoff. Coordination site
types likewise use the infinite-tiling neighbor test, so truncation does not
change a site's local environment label.

Coordinates have embedding dimension two. An exact topological-dimension
record of two describes the support of the infinite tiling family. No lattice
dimension is claimed because the tiling has no translational lattice.

## Resources and determinism

The lattice-index bound is derived from both the requested physical radius and
the acceptance-window circumradius. `max_candidates` is checked before the
four-dimensional scan; `max_sites` stops selection before graph and face
materialization. The centered window, closed-window tolerance, site ordering,
edge orientations, and radial cutoff are deterministic and recorded in
metadata.
