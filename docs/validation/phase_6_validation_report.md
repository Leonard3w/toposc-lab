# Phase 6 validation report

Date: 2026-08-30

## Scope

This validation does not implement or prefigure Phase 7. It checks the completed
Phase 6 geometry infrastructure against the existing Hamiltonian, visualization,
serialization, hashing, validation, and exact-diagonalization APIs.

The repeatable entry point is:

```powershell
python examples/phase_6_validation_smoke.py
```

Its generated overview is written to the ignored path
`results/phase_6_validation/geometries.png`.

## Results

- The complete pre-validation test suite passed: 1,566 tests.
- The complete post-validation test suite and project-wide Ruff check passed.
- Strict mypy validation passed for all 50 Phase-6-relevant geometry,
  Hamiltonian, solver, core, geometry-plotting, and smoke-test source files.
- All 24 built-in registry generators produced a representative geometry.
- Every generated geometry passed structured validation.
- Every selected generator case was deterministic for identical parameters and
  seed.
- Registry provenance, coordinate shape, displacement dimension, exact
  serialization round trips, and canonical graph-hash stability were verified.
- The selected random cases happened to be connected; the general APIs still
  intentionally permit disconnected geometries.
- Representative 1D, 2D, 3D, fractal, random, and quasiperiodic geometries passed
  through tight-binding construction and exact diagonalization.
- The same representatives passed through algebraic spinless graph p-wave
  pairing, BdG construction, and exact diagonalization. Their finite spectra were
  particle-hole symmetric to absolute tolerance `1e-10`.
- The visualization was rendered and visually inspected. Coordinate-bearing
  geometries use their coordinates; the abstract random graph uses the documented
  deterministic circular fallback layout.

No verified production bug was found, so no production code was changed.

The repository does not yet have a clean whole-project strict-mypy baseline. A
diagnostic full-source run reported 76 pre-existing errors outside the new smoke
test after its own finding was corrected. They are concentrated in older
Quantum-Hall, gas, Plotly/Matplotlib, app, and model code and include missing
third-party stubs. They were not changed during this Phase 6-only validation.

## Architecture findings

### Stored edge orientation is physical input for graph p-wave pairing

Severity: important interpretation constraint, not a builder defect.

`build_spinless_p_wave_pairing` assigns the sign of the antisymmetric pairing
from each stored `GeometryEdge` orientation. A diagnostic compared an eight-site
ring whose closing edge is stored as `7 -> 0` with the same undirected graph whose
closing edge is stored as `0 -> 7`. The default canonical graph hashes are equal,
but the constant-pairing BdG spectra are not; the maximum eigenvalue difference
was approximately `0.112781` for hopping `-1`, onsite `0.25`, and pairing `0.2`.

Consequences:

- A constant graph p-wave coefficient is an algebraic smoke model, not a
  coordinate-independent physical p-wave prescription.
- Dataset deduplication for superconducting candidates must include the pairing
  orientation/gauge convention or compare the resulting attributed Hamiltonian;
  the topology-only graph hash is insufficient.
- Coordinate-aware chiral or directional pairing should be preferred when a
  physical spatial p-wave model is intended.

### Generic geometry-to-result path is matrix-level only

Severity: low, expected boundary before Phase 7.

Arbitrary geometries can use the generic Hamiltonian builders and
`ExactDiagonalizationSolver.solve(matrix)`. There is not yet a generic `BaseModel`
adapter that carries an arbitrary geometry through `solve_model` into a
`SimulationResult`. The smoke test therefore verifies the existing matrix-level
pipeline and makes no Phase 7 addition.

### Solver package export is inconsistent

Severity: low API ergonomics issue.

`ExactDiagonalizationSolver` is not exported from `toposc_lab.solvers`; callers
must import it from `toposc_lab.solvers.exact_diagonalization`. Existing project
tests already use that path, so the validation script follows the established
API rather than changing package exports.

### Production geometry plotting is two-dimensional

Severity: medium for future 3D analysis, no effect on geometry data.

`plot_geometry` projects every embedding with two or more coordinate components
onto its first two axes. A cubic lattice therefore collapses distinct z layers.
The validation script uses a local 3D Matplotlib panel to verify and display all
three coordinates without modifying the production plotting API.

## Physics limits of this smoke test

- Successful diagonalization and a nonzero finite-size minimum `abs(E)` do not
  establish a bulk gap, topological phase, Majorana mode, or thermodynamic
  stability.
- Abstract random graphs have no physical distances or directions. Their circular
  plotting layout is visual only and must not be used to derive couplings.
- Connectivity is a model/evaluation policy. Geometry validation allows
  disconnected graphs unless callers explicitly require connectivity.
- The versioned 1-WL canonical hash remains a candidate fingerprint, not an exact
  graph-isomorphism proof.

These limits should remain explicit when Phase 7 is designed, but no Phase 7 work
was performed here.
