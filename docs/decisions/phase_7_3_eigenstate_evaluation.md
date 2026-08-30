# Phase 7.3: Eigenstate evaluation

## Status

Accepted.

## Context

Phase 7.2 selects low-energy eigensolver columns and evaluates their eigenvalues. Phase
7.3 must add state-resolved IPR and localization without assuming a rectangular lattice,
silently inventing graph boundaries, or introducing Majorana and topology diagnostics
from later steps.

The existing lattice localization profile derives edges from array axes. General Phase 6
geometries instead have a canonical flat site order, optional embedding coordinates in any
dimension, and explicitly declared boundary sites.

## Decision

`localization_profile_on_geometry` is an observable below the evaluation layer. It maps an
eigenvector column through an explicit `BasisLayout`, combines internal components per
geometry site, and returns the existing `LocalizationProfile` result. IPR and participation
ratio use the normalized site probabilities. Boundary weight and classification use only
`geometry.boundary_sites`; graph degree and coordinate extrema never imply a boundary.

For embedded geometries, center of mass is the probability-weighted coordinate vector in
the full embedding dimension. For an abstract geometry without coordinates,
`center_of_mass` is an empty array and the higher evaluation layer records a warning.

`evaluate_eigenstates` consumes an existing `GeometryEvaluation` and evaluates exactly the
state indices in `low_energy_states`. It preserves spectral and unrelated fields, replaces
IPR and localization with complete mappings for the selected states, and removes the
Phase-7.2 pending-eigenstate warning. It records the finite-state and explicit-boundary
limitations. It does not calculate Majorana or topology diagnostics.

## Consequences

- Matrix-level eigensystems remain usable when callers provide the matching `BasisLayout`.
- Arbitrary 1D, 2D, 3D, higher-dimensional, and abstract geometries share one state path.
- Component-major BdG bases are mapped to sites without a Kitaev-specific special case.
- Boundaryless geometries receive zero boundary weight rather than a degree-based guess.
- Majorana integration remains exclusively in Phase 7.4 and topology in Phase 7.5.
