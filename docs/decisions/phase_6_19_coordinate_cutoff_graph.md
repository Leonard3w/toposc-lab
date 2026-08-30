# Phase 6.19 decision: coordinate-cutoff graph builder

## Public contract

`coordinate_cutoff_graph(coordinates, cutoff)` treats every row of a finite
rectangular coordinate array as one site in arbitrary positive embedding
dimension. It connects two distinct site indices exactly when their Euclidean
distance is less than or equal to the positive finite cutoff. Site pairs and
edges are emitted in deterministic lexicographic index order.

Coincident coordinates do not collapse sites. Distinct coincident site indices
obey the same inclusive rule and therefore receive a zero-length edge. This is
useful for multiple orbitals or layers represented at one spatial position and
avoids an undocumented geometry rewrite.

## Spatial index and exact semantics

SciPy `cKDTree` supplies candidate neighborhoods without enumerating all
`n_sites * (n_sites - 1) / 2` pairs. The query radius is advanced by one
floating-point value and every candidate is then checked against the original
cutoff with the exact public Euclidean comparison. This preserves the inclusive
threshold even if the spatial index handles its internal radius boundary
differently.

`max_edges` limits the actual materialized graph, which is the relevant memory
risk. A quadratic candidate-pair budget is intentionally absent because it
would reject large sparse point clouds despite using a spatial index.

## Shared point-cloud validation

Coordinate shape, nonempty axes, finiteness, and positive distance validation
live in a private point-cloud helper. Phase 6.20 can reuse the same coordinate
contract for k-nearest-neighbor construction without exposing a premature
generic public abstraction.

## No inferred geometry semantics

An arbitrary coordinate cloud does not define a canonical boundary, tile set,
site classification, translation rank, topological dimension, or physical
dimension. The builder therefore records only embedding coordinates and cutoff
connectivity. Boundary sites, faces, site types, and dimension records remain
empty. Physical models remain responsible for assigning hoppings or other
couplings to the resulting `distance_cutoff` edges.
