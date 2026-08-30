# Phase 6.22: Geometry validation pipeline

## Status

Accepted.

## Context

`Geometry` is an immutable validated snapshot and already rejects obvious
malformed constructor input. Automated generation, future persistence, and
experiment-specific compatibility checks also need a non-mutating audit with
machine-readable results. Connectivity is not a universal geometry invariant:
valid random or intentionally composite geometries may be disconnected.

Metadata must have deterministic exact-serialization semantics before Phase
6.23 can persist geometries safely. Arbitrary Python objects, unordered sets,
reference cycles, non-finite numbers, object arrays, and arrays with unsupported
dtypes do not meet that contract.

## Decision

`validate_geometry` returns an immutable `GeometryValidationReport` containing
stable `GeometryValidationIssue` records and a canonical connected-component
partition. It never mutates or rejects the supplied geometry by itself.

The pipeline checks site indices used by edges, boundaries, and faces;
undirected duplicate edges; self-loops; connected components; coordinate and
edge-displacement dimensions; finite real coordinates; and metadata attached
to the geometry, edges, and faces. These checks deliberately repeat core
constructor invariants as defense in depth for future deserialization and
untrusted generated artifacts.

Disconnected components produce a warning by default, so the report remains
valid. A calling experiment that requires connectivity passes
`require_connected=True`, which promotes the same finding to an error. Calling
`report.raise_for_errors()` then provides explicit rejection through
`GeometryValidationError`.

Valid metadata consists of nested string-keyed mappings and tuples containing
`None`, exact Python booleans, strings, bytes, integers, finite floats or
complex numbers, typed NumPy scalars, and supported non-object NumPy arrays.
Numeric arrays and scalars must be finite. Arbitrary numeric subclasses are not
accepted because their type and semantics could not be reconstructed exactly.
This typed contract preserves scientific metadata without limiting it to JSON,
while remaining deterministic for exact serialization.

Complex site coordinates are rejected during normal `Geometry` construction;
silently discarding their imaginary part is not permitted.

## Consequences

- Connectivity policy belongs to the calling experiment, not the graph type.
- Validation findings are suitable for automation, logs, filtering, and tests.
- Existing valid disconnected generators remain usable without exceptions.
- Phase 6.23 receives an explicit metadata domain for exact save/load support.
- Invalid data cannot be made acceptable by relying only on stale derived
  neighbor caches or by bypassing a single constructor check.
