# Phase 10.7 — Rewire-edge mutation

## Status and scope

Accepted as the Phase-10.7 architecture for replacing one stored edge in an
existing `GeometryGenome` with a caller-specified edge having different
undirected endpoints.

This step adds no endpoint sampler, random number generator, dimension-changing
mutation, population, fitness, selection, crossover, generation loop,
checkpoint, benchmark, or learned behavior.

## Scientific role

Rewire-edge mutation changes connectivity while holding site and edge count
fixed. It may later support resource-controlled comparisons, but a fixed edge
count alone does not guarantee equal total coupling strength, degree profile,
spatial range, boundary exposure, or scientific fairness.

The primitive performs no physics evaluation and makes no claim about topology,
robustness, novelty, or fabrication suitability. Those remain governed by
separate experiment and evidence contracts.

## Deterministic replacement contract

`rewire_edge_mutation(genome, edge_index, edge)` replaces exactly the edge at
one nonnegative stored sequence index. The caller supplies a complete
`GeometryEdge`, including its source-to-target orientation, type,
boundary-crossing marker, displacement, and metadata.

The replacement must have a different undirected endpoint pair from the stored
edge. Merely reversing the old orientation or changing only edge attributes is
not a rewire operation. One endpoint or both endpoints may change.

The replacement occupies the old sequence position. Every other edge retains
its exact object state, orientation, and relative order. The operation does not
derive a displacement from coordinates or copy attributes from the removed
edge.

## Associated structures

A face is retained exactly when its polygon boundary does not use the replaced
undirected edge. Dependent faces are dropped; no new face is inferred from the
replacement.

The rooted-tree annotation is cleared because its contract requires its parent
edges to equal the complete graph edge set. Coordinates, embedding dimension,
boundary records, site types, sourced dimension records, metadata, and site
count remain unchanged.

## Validation boundary

The source and result pass the established Phase-10.1 genome-to-Geometry
boundary. This rejects an empty edge sequence, invalid stored index, self-edge,
endpoint outside the site range, or duplicate undirected replacement edge.

A rewire may disconnect a graph or change degrees while remaining base-valid.
Connectivity, degree bounds, edge length, crossings, planarity, resource
matching, fabrication constraints, topology, and other scientific eligibility
rules remain deferred to Phase 10.9 or a frozen experiment protocol.

## Identity boundary

Rewiring normally changes both the representation-sensitive exact Geometry
snapshot ID and the canonical graph fingerprint. Neither identifier controls
acceptance, and the graph hash remains only an isomorphism-candidate
fingerprint.
