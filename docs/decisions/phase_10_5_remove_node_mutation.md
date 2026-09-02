# Phase 10.5 — Remove-node mutation

## Status and scope

Accepted as the Phase-10.5 architecture for removing one explicitly selected
site from an existing `GeometryGenome`.

This step adds no move-node, rewire, or dimension-changing mutation. It adds no
population, fitness, selection, crossover, generation loop, checkpoint,
benchmark, node sampler, or learned behavior.

## Scientific role

Remove-node mutation contracts candidate size and removes all connectivity
incident to one site. It is infrastructure for later controlled inverse-design
experiments, not evidence that a smaller geometry has better class-D topology or
robustness.

Any later scientific comparison must still control site and coupling budgets,
spatial density and scale, boundary exposure, model parameters, disorder, and
the research charter's independent evidence gates.

## Deterministic site contract

`remove_node_mutation(genome, site_index)` removes exactly the site at one
nonnegative stored index. It uses no random number generator. A later search
driver may select that index under a separate reproducibility contract.

At least one site must remain because the production `Geometry` contract
requires `n_sites >= 1`. Surviving old sites retain their order: indices below
the removed site are unchanged and indices above it decrease by one.

## Edge orientation and attributes

Every edge incident to the removed site is dropped. Each surviving edge keeps
its relative sequence position and fixed source-to-target orientation while both
endpoints are mapped to the compact site indices.

Edge type, boundary-crossing marker, displacement, and metadata are retained
without recomputation. In particular, the mutation does not derive displacement
from coordinates or reinterpret an oriented reference after reindexing.

## Spatial, site, and boundary data

An existing coordinate table loses exactly the selected row; an absent table
remains absent. Embedding dimension and sourced dimension records remain
unchanged. No coordinate, dimension, or physical interpretation is invented for
an abstract graph.

A complete site-type column loses the corresponding entry. Aggregate boundary
sites and explicit boundary-component sites use the same compact mapping. Empty
components are dropped, matching the established Phase-8.6 node-removal policy.
No newly exposed physical boundary is inferred from degree or coordinates.

## Faces and rooted trees

A face is retained exactly when all of its sites survive. Its ordered boundary,
face type, and metadata are preserved under compact reindexing. Faces containing
the removed site are dropped rather than repaired or triangulated.

The rooted-tree annotation is cleared because removing a site invalidates its
complete parent array and edge-set correspondence. No replacement root or
parent structure is inferred.

## Validation boundary

The source and result pass the Phase-10.1 genome-to-Geometry boundary. A removal
may produce a disconnected or edgeless but base-valid geometry.

Connectivity, minimum degree, planarity, edge length, site separation,
resource-matching, fabrication constraints, topology, and clean scientific
gates remain outside this primitive and are not pulled forward from Phase 10.9.

## Identity boundary

Removing a site changes the representation-sensitive exact Geometry snapshot
ID and normally changes the graph fingerprint. Neither identifier controls the
operation. The exact ID remains a snapshot identity and the graph hash remains
only an isomorphism-candidate fingerprint.
