# Phase 10.6 — Move-node mutation

## Status and scope

Accepted as the Phase-10.6 architecture for replacing the explicit coordinate
of one selected site in an existing `GeometryGenome`.

This step adds no node selection or displacement sampling, rewire operation,
dimension-changing mutation, population, fitness, selection, crossover,
generation loop, checkpoint, benchmark, or learned behavior.

## Scientific role

Move-node mutation changes spatial embedding while preserving graph topology.
It can later support controlled searches over geometry at fixed connectivity,
but the primitive itself performs no physical evaluation and makes no claim
about topology, robustness, novelty, or fabrication suitability.

Any scientific comparison must still control spatial scale and density, minimum
site separation, coupling range, edge crossings, boundary exposure, model
parameters, disorder, and the research charter's independent evidence gates.

## Deterministic coordinate contract

`move_node_mutation(genome, site_index, *, coordinate)` replaces exactly the
coordinate row at one nonnegative stored site index. The caller supplies one
finite real vector whose length equals the existing coordinate-axis count.

The operation requires a complete coordinate table. It rejects abstract graphs
and displacement-only embeddings rather than inventing coordinates for any
site. Embedding dimension is preserved, including dimensions above three.
There is no random number generator or hidden movement policy.

## Oriented edge-displacement policy

An implicit edge displacement remains `None` and consequently follows the new
coordinate difference through the production `Geometry` API.

For every explicit displacement incident to the moved site, the mutation uses
the established coordinate-perturbation rule

```text
new displacement = old displacement + target offset - source offset
```

where only the selected site has a nonzero offset. This preserves the stored
source-to-target orientation and any periodic-image contribution encoded in the
old displacement. Nonincident edges remain unchanged. A non-finite derived
displacement is rejected.

## Preserved structures

Site count, edge order and endpoints, boundary records, site types, sourced
dimension records, rooted-tree structure, metadata, and faces remain unchanged.
Physical boundaries are not reclassified from coordinates, and faces are not
reconstructed or tested for geometric regularity.

## Validation boundary

The source and result pass the Phase-10.1 genome-to-Geometry boundary. Phase
10.6 validates only the selected stored index, coordinate representation, and
finite derived explicit displacements.

Minimum separation, coordinate bounds, edge-length limits, planarity,
fabrication constraints, and other search-space or scientific eligibility rules
remain outside this primitive and are not pulled forward from Phase 10.9.

## Identity boundary

Moving a site normally changes the representation-sensitive exact Geometry
snapshot ID while leaving the coordinate-independent canonical graph hash
unchanged. Neither identifier controls acceptance of the operation.
