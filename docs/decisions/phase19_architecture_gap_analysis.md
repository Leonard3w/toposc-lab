# Phase 19: architecture and gap analysis before implementation

Date: 2026-09-25. Status: repository inspection and baseline verification;
no production source changes for this stage yet.

Starting commit: `aa7500a2bcfa986e59fa17e15995e7e90aeaad5e`.
The starting worktree additionally contains the completed Phase-18 implementation
and confirmation artifacts. That commit alone does not reproduce Phase 18.
Authoritative pre-change source SHA256:
`b18bb71739b0cf864682b55c31b253442352f9f51ef7ba8d1909aa989e34a6fa`.
The existing Phase-18 source archives and a new preflight archive preserve this state.

## Architecture inspected

The inventory covers all 297 Python source files and identifies 233 test modules.
An AST inventory of public/top-level definitions is saved in
`results/phase19-preflight/architecture_inventory.txt`. This is an architecture
inventory plus targeted implementation review, not a claim to have manually
reviewed every line in all unrelated subsystems.

| Layer | Current responsibility and reuse decision |
|---|---|
| `core`, `lattices` | Model/results interfaces and older lattice interfaces; preserve compatibility. |
| `geometry/base.py` | Immutable arbitrary embedded graph and optional cell-complex representation; retain. |
| `geometry/generators` | 31 Python files including builders, helpers and registry/protocol; use existing builders. |
| `hamiltonians`, `models` | Generic edge-based hopping, antisymmetric pairing, basis-aware BdG construction; retain. |
| `solvers`, `observables`, `topology` | Exact solver, gaps/localization/Majorana diagnostics, localizers and Chern markers; retain. |
| `evaluation` | General geometry evaluation, named raw outputs, reproducibility and a separate basic score; do not substitute that score for the Phase-17/18 score. |
| `robustness` | Existing onsite and other disorder channels, ensembles, identifiers and uncertainty; reuse onsite channel. |
| `data` | Geometry/model/spectrum/observable records, validation, storage and migrations; retain. |
| `search` | Seeded generator sampling, mutation constraints, ranking, batch evaluation, persistence and old protocol-specific experiments; reuse components without rerunning old campaigns. |
| `research` | Exact-stage accounting, SQLite storage, provenance, checkpoint/resume, fixed-site search and fixed-cohort validation; extend only the geometry-dependent adapters. |
| `visualization`, `scans` | Geometry, spectra, localization and study plots plus sweeps; compose existing plots. |
| `discovery`, `generative`, `active_learning`, `ml`, `design`, `patterns` | Existing search/learning/design layers; no new algorithm or neural infrastructure needed for this stage. |
| `app`, `toposc_live`, CLI | UI/entry points over existing scientific services; no UI rewrite needed. |
| `bosons`, `gases`, `quantum_hall` | Separate scientific applications; outside this change. |

## ALREADY IMPLEMENTED

- **The requested graph representation already exists.** `Geometry` stores
  `n_sites`, arbitrary `edges`, arbitrary physical `coordinates`, embedding
  dimension, boundary sites/components, site types, faces and metadata.
  Site IDs are stable integer indices `0..N-1`; model parameters are separate.
  `GeometryEdge` stores endpoints, optional type, displacement, periodic-crossing
  marker and immutable metadata. No replacement `GraphGeometry` is warranted.
- `Geometry.site_indices`, `neighbors`, `degree`, `position`,
  `displacement_between`, `distance`, `direction`, `edge_between` and `has_edge`
  already provide the required graph/bond access. A 2D angle is simply
  `atan2(displacement[1], displacement[0])`; a parallel bond representation is unnecessary.
- `validate_geometry` reports connected components and structured validation
  issues. `geometry/serialization.py` preserves complete geometries, including
  metadata, coordinates and boundaries. Existing hashes distinguish exact
  geometry identity from graph/symmetry equivalence.
- `ChiralPWaveModel` already operates on arbitrary embedded `Geometry` objects.
  `build_tight_binding_hamiltonian`, `build_chiral_p_wave_pairing` and
  `build_bdg_hamiltonian` are the existing physics implementation.
- **Pairing convention is explicit:** for stored orientation i→j,
  Δij = Δ × (dx + i × chirality × dy)/sqrt(dx²+dy²) in a 2D embedding;
  Δji = −Δij. `Geometry.direction` supplies the normalized physical displacement.
  Hopping and pairing amplitudes do not decay with length in the current model.
  Zero-length bonds cannot supply a p-wave direction. Retain this convention.
- Hermiticity, BdG particle-hole symmetry, antisymmetric pairing and diagonal-bond
  direction are already tested in `test_chiral_p_wave.py`, `test_p_wave_pairing.py`,
  `test_bdg_hamiltonian.py`, `test_bdg_particle_hole_symmetry.py` and related tests.
- Square generators and connectivity rewiring exist. `FixedConnectivitySpace`
  already samples, validates and mutates fixed square-site graphs.
- `coordinate_cutoff_graph` and `k_nearest_neighbor_graph` already construct
  spatial connectivity. `hard_core_planar_graph` already constructs seeded
  amorphous hard-core embedded graphs with geometric constraints.
  Abstract random, random-regular and small-world generators also exist.
- Triangular, honeycomb, quasiperiodic and fractal generators already exist;
  do not reimplement or add them to this stage's experiment.
- `MutationValidityPolicy` already controls connectivity, embedding dimension,
  site/edge counts, degree bounds, coordinate bounds, minimum site separation,
  maximum bond length, boundary counts and optional edge crossings, with coded
  rejection reasons. Fixed N and edge-count bounds also control mean degree.
- `extract_geometry_descriptors` already computes graph counts, degree moments,
  clustering, components and reachable shortest paths. `compute_descriptors`
  adds degree extrema, boundary counts, length statistics and orientation anisotropy.
- Exact disorder fields, spectra, center localizers, legacy Q, success, low-energy
  states, Majorana diagnostics, spatial localizers, window projector densities and
  local Chern markers are already stored separately in the Phase-18 pipeline.
- `search/phase_9_8_evaluation.py` additionally contains explicit amorphous
  topology inputs, clipped Voronoi position areas, graph-distance bulk masks and
  boundary signatures. Reuse/factor its area calculation; do not implement a
  second Voronoi algorithm or silently assign unit area to irregular cells.
- `ResearchStore`, `ResearchEngine._stage`, provenance archives, leases,
  transaction/checksum handling, attempt budgets and recovery already exist.
  The fixed-cohort loop is `ValidationStudy`; no second checkpoint/storage engine is needed.
- `plot_geometry`, `plot_eigenvalue_spectrum`, localization plots and existing
  statistical plotting facilities provide the required rendering primitives.

## PARTIALLY IMPLEMENTED

- **Research applicability, not graph storage:** `FiniteSystemEvaluator` calls
  `_square_probe`, which rejects any coordinate set other than the unit square
  lattice. `PhysicsProtocol` deliberately freezes the old adapter ID and probe
  rule. Broadening must use an explicit new adapter/protocol version and reuse
  the evaluation body, rather than silently weakening the historical contract.
- `ValidationEvaluator` chooses spatial probes using sqrt(N), derives boundary
  distance from the sample bounding box, uses a fixed bulk inset and reconstructs
  the historical model with fixed parameters. These choices need an explicit
  physical-domain configuration for free coordinates, with old defaults preserved.
  Generic Chern topology already accepts per-position areas; the missing work is
  integrating existing area/domain helpers with this research adapter.
- The generic constraints are exposed through a mutation-oriented API requiring
  source/candidate genomes. A direct geometry-validation entry point can reuse
  the same checks. No second constraint implementation is needed.
- Descriptors are largely complete, but `compute_descriptors` unconditionally
  includes `regular_edge_distance`, which explicitly rejects nonsquare sites.
  Applicability must be explicit; undefined reference distance must not become zero.
  Boundary fraction can be derived from existing boundary count/N.
- The hard-core planar generator is a frozen Phase-9.8 ensemble with N=64,
  E=112 and a fixed box/constraints. Preserve its default seeded output.
  Generalization must factor its existing sampling/selection machinery or expose
  a separately versioned recipe, not duplicate its algorithms.
- Generic random graphs do not automatically have physical coordinates or satisfy
  length/degree/material constraints. Coordinate and edge generators must be
  composed under the common policy; an abstract graph alone is not a valid BdG input.
- `GeometryGeneratorRegistry` and `sample_random_geometries` already support seeded
  recipes and provenance. The existing sampler is fail-fast; a bounded experiment
  still needs an explicit, persisted rejection ledger and family-balanced schedule.
- `ValidationStudy.make_config` intentionally permits only the Phase-18 cohort,
  seeds and scientific modes. Extract/reuse its fixed schedule lifecycle behind
  overridable configuration/evaluation/report hooks; do not relabel Phase 18.
- Existing plotting primitives lack this particular family-comparison report.
  Add report composition and tables, not alternate geometry/spectrum renderers.

## MISSING

1. A versioned embedded-domain contract for the research adapter: physical box,
   center, boundary shell, interior probes, bulk mask and declared marker area.
   This is evaluation configuration, **not a new graph representation**.
2. End-to-end equivalence tests exercising a square reconstructed directly with
   existing `Geometry`/`GeometryEdge` through old and generalized research adapters,
   including disorder, Q and spatial/Chern/window diagnostics.
3. A common, serialized family-comparison recipe composing existing generators,
   constraints and frozen physical parameters, plus the missing constrained
   embedded random-edge recipe where no existing builder matches the contract.
4. Explicit applicability handling for square-reference descriptors and the
   small missing descriptive metadata; no new optimization objective.
5. A bounded, outcome-blind family cohort preparation step with complete accepted
   coordinates/edges and a rejection ledger, feeding the existing fixed-cohort
   execution/storage/recovery infrastructure.
6. The new exploratory experiment, reconciliation tests and family report.

## Proposed incremental design and trade-offs

**Recommended:** keep `Geometry`, all Hamiltonian/disorder/scoring functions,
serialization and exact-stage runtime. Add explicit domain configuration and a
versioned research-adapter mode; factor reusable existing constraint/generator
helpers only where needed. Reuse the fixed-cohort runtime for a frozen random
sample, without acquisition or surrogate models.

Alternative: use the general `evaluate_geometry` pipeline directly. It already
supports arbitrary geometries, but its separate basic-score contract does not
reproduce the historical Q automatically and would require additional composition.
It is useful for tests, not a reason to replace the validated research adapter.

Rejected: a new graph class and a separate BdG/evaluation/storage stack. Existing
geometry and physics already cover that responsibility; duplication would add
conversion and scientific-drift risks without solving the adapter restrictions.

Proposed first experimental configuration, to be frozen before any physics results:

- N=64, E=112, box [0,7]², connected, degree 2–4, separation ≥0.55,
  length ≤1.75, common physical boundary shell d<0.875.
  This reuses the existing hard-core ensemble and matches an 8×8 square;
  it is a new size cohort, not pooled with Phase-18's 100-site confirmation.
  The position extent and topology integration cell are distinct explicit fields:
  reuse the existing clipped Voronoi cell [-0.5,7.5]² for area weights (area 64),
  which reproduces unit areas on the square. Preserve the declared coordinate-box
  boundary-shell convention across families, and do not infer it from graph degree.
- A: one deterministic square reference, not 50 duplicate graphs mislabeled as
  independent samples. B/C/D: 50 unique valid geometries each, sampled before
  evaluating any physics. Coordinate ordering and paired disorder convention recorded.
- B: existing square connectivity sampler/rewiring. C: existing amorphous hard-core
  planar generator. D: constrained random selection of physical bonds on hard-core
  points using existing point/edge and validity machinery.
- Same model t=1, μ=2, Δ=1, chirality +1; original kappas and Q≥0.20 threshold.
  No distance-dependent coupling added. Bond-length and orientation differences
  remain explicit structural variables, never uncontrolled extra material.
- W=3/6/9 and three fresh disorder seeds per geometry plus clean reference:
  151 geometries × 10 stages = 1510 planned evaluations. Cap at 1600 attempts,
  1800 seconds, one worker/BLAS thread; measure a small technical timing check
  first and stop incomplete if the cap is insufficient. No multi-hour search.
- All outcomes reported. Example candidates selected only for descriptive plots,
  never labeled discoveries; fresh future confirmation required for any signal.

Implementation order: preserve baseline → extend adapter configuration/shared
hooks → square equivalence/PHS gate → compose families/constraints → tests →
bounded exploration → report → stop. A failed physical or fairness gate stops
progress as required by the user.

## Baseline evidence and current status

The full pre-change suite was launched before any production modifications;
its final result will be recorded in `phase19_preflight_verification.json`.
Read-only database integrity/checksum verification covers all 2510 Phase-18
records. Three fixed cases were independently recalculated: regular clean,
regular W=6 seed 181001, historical_best W=7.5 seed 181001. Hamiltonian identity,
spectra, disorder field, Q, success, spatial localizers, Chern bulk mean and
boundary-window density agree at absolute tolerance 1e-10. Historical attempts
and stored result checksums were unchanged. These are three regression
evaluations, not a new scientific ensemble or a replay of all 2510 eigensystems.

The attached full user brief is preserved in `docs/roadmap/phase19_user_plan.md`.
