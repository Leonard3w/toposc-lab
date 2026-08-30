# Phase 6.8 decision: cell-centered Sierpiński carpet

## Decision

The finite Sierpiński carpet is represented by one graph site at the center of
each retained square cell. Two sites share a nearest-neighbor edge exactly when
their cells share a full horizontal or vertical side. Diagonal contact does
not create an edge.

Recursion order zero contains one cell. Every iteration replaces each retained
cell with the eight noncentral cells of a 3-by-3 block. This gives exactly
`8**order` sites and a deterministic site order sorted first by y and then x.

## Boundary semantics

`Geometry.boundary_sites` is the union of every geometric boundary. New typed
`GeometryBoundaryComponent` records preserve the outer boundary and every
geometrically distinct hole boundary separately. Components may overlap at
retained cells; this is required because a discrete cell can border two
different holes. A component describes sites adjacent to one boundary curve;
those cell centers need not induce a connected graph because diagonal contact
does not create an edge.

The typed representation avoids placing scientifically meaningful site sets in
free-form metadata. Existing generators remain compatible because boundary
components are optional.

## Dimensions

The coordinate embedding remains two-dimensional. Independent immutable
dimension records describe the infinite carpet family:

- topological dimension 1 by covering dimension;
- Hausdorff dimension `log(8) / log(3)` from analytic self-similarity.

Neither value is inferred from the embedding or used implicitly for topology
dispatch.
