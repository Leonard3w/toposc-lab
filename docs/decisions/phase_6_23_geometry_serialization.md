# Phase 6.23: Exact geometry serialization

## Status

Accepted.

## Context

Geometry discovery needs persistent artifacts that preserve every public
`Geometry` field, including coordinates, oriented edge displacements, faces,
boundary components, dimension records, rooted-tree structure, generator
provenance, and typed scientific metadata. Pickle would permit arbitrary code
execution and would not provide a stable, inspectable schema.

Plain JSON cannot preserve NumPy dtypes and shapes, byte strings, complex
values, NumPy scalar types, or floating-point details such as negative zero
without an additional typed representation.

## Decision

Geometry files use compressed `.npz` archives with format identity
`toposc_lab_geometry` and schema version 1. One UTF-8 JSON manifest is stored as
a `uint8` array. Coordinates and metadata arrays remain normal NumPy archive
members. Loading always uses `allow_pickle=False`.

The manifest stores every non-derived geometry field. Finite Python floats and
complex components use hexadecimal strings for exact binary reconstruction.
Bytes use base64. Integers use decimal strings in metadata so arbitrary Python
precision is retained. NumPy arrays and scalars are referenced as separate
archive members, preserving dtype, shape, and raw values. Mappings and tuples
use typed nodes and retain their item order.

`save_geometry` first runs the Phase-6.22 validation pipeline. Disconnected
graphs remain valid and serializable. The destination suffix is `.npz`, parent
directories are created when needed, and a fully written temporary file is
atomically moved over the destination.

`load_geometry` and `geometry_from_bytes` validate ZIP member names,
compression methods, member count, compressed size, total uncompressed size,
manifest fields, format identity, schema version, array references, dtypes,
and the reconstructed geometry. Missing, duplicate, multiply referenced, or
unreferenced members are rejected. Pickled object arrays are never loaded.

## Consequences

- Valid geometries round-trip without loss of public structure or typed
  metadata.
- Corrupt, ambiguous, oversized, and future-schema archives fail explicitly.
- Files can be inspected with standard ZIP, JSON, and NumPy tools.
- The schema can evolve through an explicit version migration rather than
  silently changing interpretation.
- Phase 6.24 canonical graph hashing does not need to depend on ZIP byte layout
  or compression timestamps.
