# Phase 6.1 decision: common geometry generator protocol

Date: 2026-08-29  
Status: accepted

## Decision

All new geometry generators use `Geometry` as their scientific output type. A generator
implements the structural `GeometryGenerator` protocol and exposes:

- a stable lowercase technical key;
- a positive algorithm version;
- whether it is deterministic or stochastic;
- `generate(request) -> Geometry`.

`GeometryGenerationRequest` carries JSON-compatible generator parameters and a separate
optional nonnegative random seed. Deterministic generators reject seeds. Stochastic
generators require an explicit seed, so automated discovery cannot silently create an
irreproducible candidate.

Every protocol-based generation adds a nested `generation` metadata record containing the
generator key, version, normalized parameters, and seed. Existing scientific metadata on
the returned geometry is preserved.

`GeometryGeneratorRegistry` provides ordered discovery by stable key and rejects duplicate
registrations. The existing chain, ring, square, and irregular-cluster functions are
adapted into the built-in registry.

## Compatibility

The established direct calls such as `chain(20)` and `square(4, 5)` remain unchanged. They
continue to return the same geometries and metadata as before. Protocol provenance is added
only when generation goes through a registered generator.

The legacy `lattices` package is not extended by Phase 6. New generator work belongs in
`geometry.generators`. Existing lattice-based models will be migrated separately with
model-specific parity tests so sublattice, orientation, and longer-range bond information
cannot be lost through an overly broad adapter.

## Dimensionality boundary

The generator protocol does not infer or store the physical dimension used for a topology
calculation. `Geometry.embedding_dimension` describes coordinate space only. Physical,
effective, topological, and fractal dimensions belong to later model or experiment context
objects and must not be inferred from coordinate columns or untyped metadata.

## Consequences

- New deterministic and random generators share one invocation and provenance contract.
- Generator configurations can be serialized without arbitrary Python objects.
- Generator algorithm changes can increment a version without changing the stable key.
- Geometry hashing and exact serialization in later Phase 6 steps have a deterministic
  provenance source.
- Existing user code is not forced to migrate immediately.

## Acceptance evidence

- All four existing generators are available through the built-in registry.
- Direct generator calls retain their prior behavior.
- Requests defensively copy and freeze nested parameter data.
- Invalid keys, versions, parameters, seeds, duplicate registrations, and non-`Geometry`
  builder results are rejected.
- Seeded stochastic generators are reproducible and unseeded stochastic generation is
  rejected.
