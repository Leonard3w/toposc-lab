# Phase 8.7: Coordinate perturbation

## Decision

Coordinate perturbation is a geometry-targeted uniform disorder transform for finite
geometries with explicit coordinates. `apply_uniform_coordinate_perturbation(...)`
requires a nonnegative finite width and explicit seed. Through the Phase-8.1 executor it
draws one independent NumPy PCG64 offset per coordinate component from
`[-width / 2, width / 2]`, in ascending site order and then ascending embedding-axis
order. A zero width is an exact no-op snapshot.

Coordinate-free geometries are rejected. In particular, abstract random graphs and tree
geometries do not receive invented coordinates, directions, boundaries, or physical
dimensions. The scalar site-disorder samplers from earlier phases are not reused because
they provide one value per site rather than an independent spatial vector; the common
seeded executor and provenance contract are reused unchanged.

## Edge displacement semantics

`Geometry.displacement_between(...)` gives an explicit oriented edge displacement
precedence over endpoint-coordinate subtraction. Merely changing coordinates would
therefore leave distance- and direction-dependent Hamiltonian terms unchanged on many
generated geometries. For every edge carrying an explicit displacement, this phase uses

`new_displacement = old_displacement + target_offset - source_offset`.

This makes Rashba and spatial chiral-p-wave directions respond to the perturbed sites,
preserves each source-to-target orientation, and retains the periodic image vector of a
boundary-crossing edge. Edge type, boundary-crossing flag, metadata, and iteration order
remain unchanged. An edge without an explicit displacement remains implicit and follows
the perturbed coordinate difference automatically. No periodic image is inferred when
the source geometry did not provide one.

## Geometry and provenance boundaries

The graph, site indices, faces, boundary labels and components, site types, dimension
records, rooted-tree hierarchy, and geometry metadata are not reclassified. Coordinate
noise can still produce coincident sites, zero directions, crossed faces, or a geometry
that is incompatible with a chosen model. The transform neither resamples nor silently
repairs such a realization; explicit Phase-7 model requirements and candidate validity
own that scientific decision.

Provenance records the distribution, width, support, source geometry ID, site/edge and
embedding sizes, draw order, coordinate update, explicit/implicit displacement policies,
and the zero-width rule. Phase 8.1 separately records the seed, PCG64 algorithm,
transform version, and exact source/result geometry snapshot IDs. Those IDs remain
representation-sensitive snapshots, while the Phase-6 graph hash remains only an
isomorphism-candidate fingerprint.

This phase does not perturb model parameters, execute ensembles, compute robustness
metrics or statistical uncertainty, perform finite-size analysis, or begin Phase 9.
