# Phase 10.3 — Remove-edge mutation

## Status and scope

Accepted as the Phase-10.3 architecture for removing one stored edge from an
existing `GeometryGenome`.

This step adds no node, move-node, rewire, or dimension-changing mutation. It
also adds no population, fitness, selection, crossover, generation loop,
checkpoint, benchmark, edge sampler, or learned behavior.

## Scientific role

Remove-edge mutation provides the inverse connectivity direction to Phase 10.2.
It can later support controlled tests of whether a coupling or motif is
necessary, but this primitive itself performs neither an ablation study nor a
physics evaluation.

Any scientific comparison must still control coupling budget, degree,
connectivity, spatial range, boundary exposure, disorder, model parameters, and
the charter's evidence gates.

## Deterministic index contract

`remove_edge_mutation(genome, edge_index)` removes exactly one edge at its
position in the genome's ordered edge sequence. Stored-index selection avoids
reconstructing an edge from endpoints or partially comparing edge attributes
and metadata.

Negative indexing is deliberately unsupported. The index must be a non-boolean
integer in `[0, len(genome.edges))`. The primitive uses no random number
generator; a later reproducible search driver may select a valid index.

## Orientation and survivor identity

The removed edge disappears as a complete `GeometryEdge` record. Every surviving
edge retains its original object state, source-to-target orientation, and
relative order. No endpoint is sorted and no displacement or metadata is
recomputed.

Coordinates, embedding dimension, boundary records, dimension records, site
types, and geometry metadata remain unchanged. Connectivity changes do not
implicitly redefine a physical boundary.

## Associated structures

A face is retained exactly when all of its polygon-boundary edges survive. Faces
that use the removed undirected edge are dropped; no replacement faces are
inferred.

The rooted-tree annotation is cleared for every successful removal because its
contract requires the complete geometry edge set to equal its parent-edge set.
Retaining it after removing an edge would be false.

## Validation boundary

The source and result pass the established Phase-10.1 genome-to-Geometry
boundary. Removing a bridge or the final edge may yield a disconnected or
edgeless but base-valid Geometry. Phase 10.3 does not reject that result.

Connectivity requirements, minimum degree, resource matching, topology,
planarity, fabrication constraints, and all experiment-specific rules remain
outside this primitive and are not silently pulled forward from Phase 10.9 or a
future protocol.

## Identity boundary

Removing an edge normally changes both the exact representation-sensitive
Geometry snapshot ID and the graph fingerprint. Neither identifier controls the
operation. The exact ID remains a snapshot identity and the graph hash remains
only an isomorphism-candidate fingerprint.
