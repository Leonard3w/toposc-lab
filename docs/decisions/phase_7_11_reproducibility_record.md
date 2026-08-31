# Phase 7.11: Reproducibility record

## Decision

`ReproducibilityRecord` is an immutable provenance output beside, rather than inside,
the scientific `GeometryEvaluation`. A completed evaluation records the explicit random
seed (or `None` when no random process was used), model name and deeply frozen model
parameters, geometry ID, solver identity and settings, numerical evaluation settings,
and the code version when it is available.

`evaluate_geometry(...)` accepts `seed` and an optional explicit `code_version`. Without
an explicit version it uses installed `toposc-lab` package metadata; absence remains
visible as `None` with a warning. The exact-diagonalization solver records its current
NumPy backend and full-spectrum behavior. Unknown subclasses retain their qualified
solver identity without inventing settings.

The record is created after model construction, when the actual model parameters are
known, and before Hamiltonian construction. Therefore every produced scientific
evaluation, and later-stage invalid result, retains provenance. A candidate rejected by
preflight or model construction has no scientific evaluation and may have no record.

## Geometry identity boundary

The geometry ID is a SHA-256 digest of the exact version-1 pickle-free geometry archive.
It includes coordinates, oriented edges and their displacements, faces, boundary data,
rooted-tree information, dimension records, site types, and metadata. This makes it an
exact snapshot identifier suitable for repeating the recorded evaluation.

It is deliberately labeled `toposc-geometry-archive-v1-sha256` and carries a warning:
the ID is representation-sensitive. It is not the Phase-6 Weisfeiler-Lehman graph
fingerprint, not an isomorphism proof, and not a physical-deduplication key. Canonical
database identity remains a later master-plan concern.

Phase 7.11 does not add persistence, datasets, ranking, optimization, or Phase-8 search
logic.
