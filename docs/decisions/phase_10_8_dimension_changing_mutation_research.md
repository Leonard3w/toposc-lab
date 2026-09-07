# Phase 10.8 — Dimension-changing mutation research

## Status and scope

Accepted as the Phase-10.8 research decision. The Phase-10 search must not
change embedding dimension through a mutation operator.

This phase adds no production code, mutation primitive, sampler, population,
fitness, selection, crossover, generation loop, machine-learning behavior, or
dataset functionality. Arbitrary-dimensional geometry representation remains
supported; the decision only limits the current search space.

## Dimensions are separate contracts

Toposc-Lab distinguishes at least three concepts:

- **embedding dimension** is the number of coordinate columns;
- **physical dimension** is an explicit model and topology-dispatch input;
- **lattice, graph, topological, fractal, or other effective dimensions** are
  sourced geometry records with their own definitions.

Changing one does not establish a change in either of the others. In
particular, adding a coordinate axis to a planar geometry does not make the
physical system three-dimensional, while a graph drawn in three coordinate
dimensions does not by itself select a valid three-dimensional Hamiltonian or
topological invariant.

The generic `Geometry` and `GeometryGenome` contracts already preserve
arbitrary positive embedding dimensions. Higher-dimensional representation,
validation, serialization, and hashing therefore remain available without a
dimension-changing mutation.

## Repository evidence

The initial discovery charter fixes a spinless chiral `p_x + i p_y` BdG model
on a declared two-dimensional physical plane. Its candidate-space, boundary,
fairness, and evidence contracts are consequently two-dimensional.

The current model records a pairing plane through `plane_axes`. Topology
dispatch independently records `physical_dimension` and `embedding_dimension`
and explicitly forbids inferring the former from coordinates. The implemented
methods cover declared one- and two-dimensional cases; there is no validated
three- or higher-dimensional topology path. The Phase-9.8 protocol likewise
admits two-coordinate geometries and dispatches with both dimensions fixed to
two.

Allowing a mutation to add or remove coordinate columns would therefore cross
an experiment boundary. It could change distance and pairing conventions,
boundary codimension, applicable invariants, computational cost, and resource
matching without supplying the corresponding scientific contracts.

## Scientific basis

The classification and invariant of a free-fermion topological phase depend on
both symmetry class and spatial dimension, rather than on graph connectivity
alone. This is explicit in the dimensional classification developed by
[Kitaev](https://arxiv.org/abs/0901.2686) and by
[Schnyder, Ryu, Furusaki, and Ludwig](https://arxiv.org/abs/0803.2786).
The weak- and strong-pairing analysis underlying the present spinless chiral
p-wave model is specifically formulated in two dimensions by
[Read and Green](https://arxiv.org/abs/cond-mat/9906453).

Consequently, a coordinate-column mutation cannot be treated as a scientifically
neutral analogue of adding, removing, moving, or rewiring graph elements.

## Decision

No `change_dimension_mutation` is introduced in Phase 10. Existing mutation
operators must preserve `embedding_dimension`, and Phase 10.9 validity checks
must treat a dimension change as outside their mutation contract rather than
silently normalizing or accepting it.

The current two-dimensional discovery program must compare candidates under one
declared physical model, topology policy, boundary policy, and resource regime.
Keeping dimension fixed prevents candidates from escaping those controls by
moving into a different physical problem.

## Later research path

Dimension may be reconsidered in Phase 17, where the master plan explicitly
introduces embedding dimension as a search parameter and dimension-comparison
experiments. Before that can be scientifically meaningful, each admitted
dimension needs:

1. a declared physical interpretation distinct from coordinate embedding;
2. a validated dimension-compatible Hamiltonian and coupling convention;
3. applicable topology methods and dimension-specific evidence gates;
4. an explicit boundary contract, including edge, surface, hinge, or defect
   observables as appropriate;
5. dimension-specific benchmarks and finite-size validation;
6. resource and computational-cost matching across dimensions; and
7. a predeclared protocol for comparing outcomes from different dimensions.

If those prerequisites are met, dimension should initially be an outer
experiment or stratification parameter with separately validated candidate
spaces. It should not begin as an unrestricted local mutation inside one
population, because candidates from different physical dimensions need not
share a comparable fitness or evidence contract.

## Consequences

- Phase 10 continues with fixed-dimension genome mutations only.
- No current geometry, model, topology, serialization, hashing, or generator API
  is changed.
- Higher-dimensional geometry support remains intact and testable.
- This decision makes no claim that cross-dimensional optimization is impossible;
  it defers it until the required physics and comparison protocol exist.
