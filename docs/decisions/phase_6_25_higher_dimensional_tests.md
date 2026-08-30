# Phase 6.25: Higher-dimensional generator tests

## Status

Accepted.

## Context

The geometry core accepts arbitrary positive embedding dimensions, and the
individual cubic, body-centered-cubic, Menger-sponge, coordinate-cutoff,
k-nearest-neighbor, and artificial-rule generators already have focused unit
tests. Those tests did not provide one explicit guarantee that representative
3D and `d > 3` outputs survive the complete geometry infrastructure.

Embedding dimension also has a different meaning from lattice, topological,
Hausdorff, or other intrinsic dimension records. A higher-dimensional point
cloud must not acquire an unsupported physical interpretation merely because
its coordinate vectors have many components.

## Decision

Maintain a cross-layer regression matrix covering:

- periodic cubic and body-centered-cubic lattices in 3D;
- the Menger sponge in 3D;
- an artificial-rule graph in 4D;
- a k-nearest-neighbor graph in 7D;
- a coordinate-cutoff graph in 11D.

Every matrix entry must be connected and pass structured geometry validation.
Coordinates and displacement vectors must retain the declared embedding
dimension. Exact in-memory serialization must preserve every geometry field,
and both topology-only and attributed canonical hashes must remain stable
after the round trip.

The `d > 3` cases are constructed through the public built-in generator
registry, not private generator internals. They must retain empty
`dimension_records`: coordinate-vector length alone does not justify an
intrinsic dimension claim.

For the Menger sponge, tests distinguish its 3D embedding from both its
topological dimension 1 and its Hausdorff dimension `log(20) / log(3)`.

## Consequences

- The supported generator pipeline has explicit regression coverage above
  three dimensions without introducing a hard-coded maximum dimension.
- Serialization, validation, and hashing are exercised together instead of
  only in isolated module tests.
- Embedding and intrinsic-dimension semantics cannot silently collapse into
  one field.
- This phase adds no production abstraction or higher-dimensional physics
  claim; physical 3D benchmarks and synthetic dimension experiments remain
  later master-plan work.
