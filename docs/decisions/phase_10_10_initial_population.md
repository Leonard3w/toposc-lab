# Phase 10.10 — Initial population

## Status and scope

Accepted as the Phase-10.10 architecture for assembling an ordered,
policy-valid collection of `GeometryGenome` objects at generation zero.

This phase adds no candidate generator, mutation sampling, retry loop, fitness,
selection, elitism, crossover, later-generation loop, diversity rule, novelty
score, checkpoint, machine-learning behavior, or dataset functionality.

## Separation from candidate generation

Initial-population assembly does not own randomness. Callers supply a finite,
ordered, nonempty iterable of genomes. When random initialization is wanted,
the existing Phase-9.1 geometry sampler remains responsible for its explicit
PCG64 root seed, derived generator seeds, recipe order, and generation
provenance; its geometries can be converted losslessly to genomes before
assembly.

This separation prevents Phase 10.10 from introducing a second random stream,
hidden retries, silent seed consumption, or generator-specific assumptions.
Generation provenance already stored in genome metadata is preserved exactly.

## Population contract

`create_initial_population(...)` returns an immutable `InitialPopulation` with:

- one or more ordered `InitialPopulationMember` records;
- consecutive member indices matching input order;
- the exact supplied genome objects;
- one successful Phase-10.9 validity report per member;
- the explicit shared validity policy;
- one common embedding dimension;
- fixed generation index zero; and
- an explicit population schema version.

The first supplied genome anchors only the embedding dimension. Every member,
including the anchor, is checked through `validate_geometry_mutation(...)` with
the same policy. The anchor does not define fitness, parentage, a scientific
reference family, or a preferred geometry.

## Failure contract

The first invalid candidate raises `InitialPopulationError` containing its
original input index and complete `MutationValidityReport`. No partial
population is returned, and the function does not filter, repair, replace, or
resample the candidate. This makes population size and any upstream seed
consumption auditable by the caller.

Malformed API inputs raise type or value errors before assembly. A successfully
constructed population contains only base-valid, policy-valid genomes sharing
one embedding dimension, consistent with the Phase-10.8 fixed-dimension
decision.

## Identity and diversity boundary

Duplicate genome objects and equivalent geometries are retained. Phase 10.10
does not use exact snapshot IDs or canonical graph fingerprints to remove or
rank members. Duplicate control and diversity preservation belong to their
explicit later phases and must not be smuggled into initialization.

## Scientific boundary

Membership establishes only structural and configured geometric admissibility.
It conveys no fitness, topology, robustness, novelty, fabrication suitability,
or scientific success. Fitness evaluation begins no earlier than Phase 10.11.
