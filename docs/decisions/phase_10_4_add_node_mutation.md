# Phase 10.4 — Add-node mutation

## Status and scope

Accepted as the Phase-10.4 architecture for appending one explicitly described
site to an existing `GeometryGenome`.

This step adds no remove-node, move-node, rewire, or dimension-changing
mutation. It adds no edge alongside the site and no population, fitness,
selection, crossover, generation loop, checkpoint, benchmark, node sampler, or
learned behavior.

## Scientific role

Add-node mutation expands candidate size while keeping existing connectivity
and embedding data fixed. It is infrastructure for later controlled searches,
not evidence that a larger geometry or an isolated site is scientifically
useful.

Any later comparison must account for site count, edge or coupling budget,
spatial density, boundary exposure, physical scale, model parameters, disorder,
and every evidence gate in the research charter.

## Deterministic site contract

`add_node_mutation` appends one site at index `genome.n_sites`. The new site is
isolated: Phase 10.4 does not call or duplicate add-edge mutation. Endpoint or
neighbor selection therefore has no hidden randomness or seed.

The caller may provide the new site's coordinate, site type, aggregate boundary
membership, and membership in existing boundary components. All existing sites,
edges, and per-site records retain their index and value.

## Coordinate and dimension policy

When a complete coordinate table exists, one finite real vector with exactly
the existing coordinate-axis count is required and appended. Embedding
dimension does not change.

When no coordinate table exists, the new coordinate must remain absent. This
includes purely abstract graphs and geometries whose dimension is declared only
through existing edge displacements. Coordinates for older sites are never
fabricated, and this step cannot introduce or change embedding dimension.

## Site-type policy

When `site_types` is a complete tuple, the caller-supplied new value is appended;
`None` remains a valid explicit unknown type. When no site-type column exists,
it remains absent and a non-`None` new type is rejected. A partial type column
would otherwise require silently inventing values for all older sites.

## Boundary policy

Boundary membership is explicit and never inferred from coordinates or degree.
For aggregate-only boundary data, `boundary=True` adds the new site to
`boundary_sites`.

When boundary components exist, a boundary site must name one or more existing
`(kind, component_index)` keys. The site is added to exactly those components;
overlapping membership is supported. Unknown or duplicate keys are rejected,
and this mutation does not invent new physical boundary components.

## Associated structures and validation

Existing edges and faces remain unchanged because no old site or edge is
removed. A rooted-tree annotation is cleared because it lacks a parent entry for
the appended site and requires an exact match to the complete site/edge set.

The source and result use the existing genome-to-Geometry boundary. An isolated
new site may make the result disconnected while remaining base-valid. Minimum
degree, connectivity, separation, resource, planarity, topology, and other
experiment rules remain deferred to Phase 10.9 or a frozen protocol.

## Identity boundary

The appended site changes the representation-sensitive exact Geometry snapshot
ID and normally the graph fingerprint. Neither identifier controls acceptance.
The exact ID remains a snapshot identity and the graph hash remains only an
isomorphism-candidate fingerprint.
