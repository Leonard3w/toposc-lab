# Phase 10.2 — Add-edge mutation

## Status and scope

Accepted as the Phase-10.2 architecture for adding one explicit edge to an
existing `GeometryGenome`.

This step adds no remove-edge, node, move-node, rewire, or dimension-changing
mutation. It also adds no population, fitness, selection, crossover, generation
loop, checkpoint, benchmark, endpoint sampler, or learned behavior.

## Scientific role

The initial research program seeks connectivity effects under controlled
physical and resource conditions. Add-edge mutation is therefore a narrow
connectivity transformation: it changes one edge gene while retaining the
coordinate embedding and all unrelated geometry state.

An added edge is not evidence of improved physics. Later experiment contracts
must still control edge or coupling budget, degree, spatial range, boundary
exposure, planarity policy, and every scientific evidence gate.

## Deterministic API

`add_edge_mutation(genome, edge)` accepts an existing `GeometryGenome` and one
fully constructed `GeometryEdge`. The caller, rather than this primitive,
chooses both endpoints and all optional attributes.

No random number generator or seed is needed because Phase 10.2 defines no
endpoint-selection policy. A future search driver may select an edge under its
own explicit reproducibility contract without changing this primitive.

## Orientation and spatial semantics

The new edge is appended after every existing edge and retains its exact
`source -> target` orientation. Existing edges are neither reordered nor
reoriented.

The mutation does not infer an edge displacement from coordinates. It also does
not infer edge type, boundary crossing, metadata, coordinates, boundary sites,
embedding dimension, or physical dimension. Abstract genomes consequently
remain abstract unless the caller explicitly supplies separately authorized
geometry data in a later operation.

## Validation boundary

The source and result must both pass the Phase-10.1 genome-to-Geometry boundary.
This rejects representation-invalid states such as an endpoint outside the site
range or a duplicate undirected edge.

Those checks are not the future Phase-10.9 mutation-validity policy. Phase 10.2
does not enforce connectivity objectives, degree ranges, edge lengths,
crossing/planarity rules, resource budgets, fabrication rules, or Phase-9.8
clean-candidate constraints.

## Associated structures and metadata

Existing faces remain stored because adding an edge does not remove any of
their required boundary edges. A rooted-tree annotation is cleared after a
successful addition because `RootedTreeStructure` requires its parent edges to
equal the complete graph edge set; retaining it would make the result false and
representation-invalid.

All other genome data, including metadata, is retained unchanged. This step
does not introduce mutation lineage or persistence metadata; those concerns
must be designed with later orchestration and checkpointing rather than hidden
inside geometry metadata.

## Identity boundary

Adding the edge normally changes both the representation-sensitive exact
Geometry snapshot ID and the graph fingerprint. Neither identifier is consulted
to accept the mutation. The exact ID remains a snapshot identity, while the
graph hash remains only an isomorphism-candidate fingerprint.
