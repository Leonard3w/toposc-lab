# Phase 7.2: Basic spectral evaluation

## Status

Accepted.

## Context

`GeometryEvaluation` can store a gap, selected low-energy states, a numerical
zero-mode count, and later diagnostics. Phase 7.2 must populate only the
eigenvalue-derived fields without coupling evaluation to a geometry class, physical
model, Hamiltonian builder, or eigensolver implementation.

The existing observable API distinguishes the full spectral separation across a
reference energy from the historical positive, zero-mode-filtered excitation estimate.
The master plan requires every reported gap to state which convention it uses.

## Decision

`evaluate_spectrum` accepts a finite, non-empty one-dimensional eigenvalue array and
returns a `GeometryEvaluation`. Its `gap` is the full spectral separation across an
explicit `reference_energy`, computed by the existing `spectral_gap` observable. A state
within `zero_mode_tolerance` closes this gap. Spectra must contain states on both sides of
the reference energy.

The numerical zero-mode count uses the same reference and tolerance. It is not evidence
of Majorana character, localization, or topology. Low-energy states are selected by
distance from the reference energy and retain their eigensolver column indices. The
requested count is a minimum: every state tied at the cutoff distance is included so
degenerate multiplets are not split arbitrarily.

The returned warnings state the gap convention and the limits of the zero-mode and
low-energy classifications. Phase 7.2 does not inspect eigenvectors or populate IPR,
localization, Majorana, topology, or geometry-descriptor fields.

## Consequences

- The spectral evaluator remains geometry-, model-, and solver-independent.
- Unsorted spectra are supported without losing the original eigenstate indices.
- For a particle-hole-symmetric BdG spectrum without zero modes, the reported full gap is
  twice the smallest positive quasiparticle excitation energy.
- One-sided spectra fail explicitly instead of silently receiving an ambiguous gap.
- Eigenstate evaluation remains exclusively in Phase 7.3.
