# Phase 6.9 decision: voxel-centered Menger sponge

## Decision

The 3D fractal prototype is an exact finite Menger-sponge graph. Each retained
cubic voxel contributes one site at its center. Two sites share an edge exactly
when their voxels share a square face; edge and corner contact do not create
graph adjacency.

Order zero contains one voxel. Every iteration retains the 20 subcubes of a
3-by-3-by-3 block that are neither face centers nor the body center. This gives
exactly `20**order` sites in deterministic z/y/x site order.

## Boundary semantics

Removed Menger cells form tunnels connected to the exterior rather than closed
cavities. Every retained voxel adjacent to the bounding-box exterior or to a
removed voxel therefore belongs to one exterior-accessible `outer` boundary
component. The component includes the tunnel surfaces. It must not be split
into artificial hole components.

## Resource guard

Three-dimensional recursion grows rapidly. The generator computes `20**order`
before allocating voxels and rejects requests above `max_sites`. Its default of
25,000 permits order 3 (8,000 sites) and rejects order 4 (160,000 sites).
Callers can deliberately raise the limit or pass `None`, making resource policy
explicit and reproducible in generation metadata.

## Dimensions

The embedding dimension is 3. Separate immutable records describe the infinite
Menger family:

- topological dimension 1 by covering dimension;
- Hausdorff dimension `log(20) / log(3)` by analytic self-similarity.

These values are descriptive and are not used implicitly for topology
dispatch.
