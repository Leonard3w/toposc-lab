# Phase 8.1: Reproducible disorder API

## Decision

Disorder execution lives in the new `toposc_lab.robustness` package, separate from
geometry construction, physical models, scientific evaluation, ensemble execution, and
robustness metrics. A `DisorderRequest` contains deeply frozen, audit-safe parameters and
a mandatory nonnegative seed. `realize_disorder(...)` alone constructs the random stream,
using the explicitly recorded `numpy.random.PCG64` algorithm, and supplies that generator
to a versioned `DisorderTransform`. Transforms must not obtain randomness elsewhere.

The neutral contract currently admits two explicit targets: immutable `Geometry`
snapshots and finite square Hamiltonian arrays. It does not define any physical disorder
distribution. In particular, onsite, hopping, pairing, removal, coordinate, and physical-
parameter perturbations remain owned by Phases 8.2-8.8. `FunctionDisorderTransform` is a
small adapter that makes the execution contract usable and testable; it is not a concrete
disorder model.

## Provenance and identity boundaries

Every `DisorderRealization` keeps its transformed state separate from an immutable
`DisorderProvenance`. The provenance records the stable transform key and version,
parameters, seed, RNG algorithm, and exact source/result snapshot references. Geometry
references deliberately reuse the Phase-7.11 exact archive hash, including physical edge
orientation. They remain representation-sensitive snapshot IDs, not canonical physical
identities or isomorphism proofs. Hamiltonian references hash the exact dtype, shape, and
C-order values under the labeled `toposc-hamiltonian-array-v1-sha256` scheme.

Hamiltonian inputs are defensively copied before transformation, results are returned as
read-only snapshots, and matrix shape must be preserved. Geometry transformations return
new immutable `Geometry` values; later concrete transforms are responsible for retaining
the physically meaningful orientation of surviving edges.

The existing `uniform_site_disorder(...)` and `uniform_edge_disorder(...)` samplers remain
available as low-level physical building blocks. Their integration belongs to Phases 8.2
and 8.3. The model-local Kitaev disorder implementations are also unchanged for backward
compatibility. `ReproducibilityRecord` and `evaluate_geometry(...)` are not extended:
evaluation provenance and disorder provenance remain distinct and can be composed by the
later ensemble layer without contaminating scientific result objects.
