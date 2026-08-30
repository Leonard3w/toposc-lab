# Toposc-Lab after Phase 6

This document describes only functionality implemented and verified in the
repository. It does not describe Phase 7 as if it already existed.

## What works now

- Immutable, model-independent finite geometries with typed sites and edges,
  coordinates or abstract graphs, explicit boundaries, faces, rooted trees,
  dimension records, metadata, utility methods, validation, exact NPZ
  serialization, and versioned canonical candidate hashing.
- Dense finite-system Hamiltonian construction and exact diagonalization with
  explicit basis layouts and standardized simulation results.
- Generic onsite/edge tight-binding terms, complex hopping, seeded site/edge
  disorder, BdG assembly, Nambu basis conversion, Hermiticity and
  particle-hole checks.
- Spinless graph p-wave, coordinate-aware chiral p-wave, onsite spinful s-wave,
  spinful d-wave, Rashba, and Zeeman Hamiltonian terms.
- Spectral, eigenstate, localization, LDOS, Majorana, symmetry, and topology
  diagnostics listed below.
- Parameter scans, reproducible study persistence, plotting/export helpers, a
  CLI Kitaev scan, and a Streamlit research/learning workspace.
- Existing non-superconducting benchmarks: SSH, graphene, Haldane, QWZ, and BHZ.
  The repository also contains tested quantum-gas, Landau-level, and integer
  quantum-Hall learning modules; they are outside the Phase 1-6 geometry demo.

The current solver is dense exact diagonalization. This is appropriate for the
small finite systems demonstrated here, not an HPC-scale sparse solver.

## Implemented superconducting and BdG models

| Model | Implemented scope |
| --- | --- |
| `KitaevChain` | Spinless 1D p-wave chain; open/periodic boundaries and seeded onsite disorder. |
| `GeometryKitaevChain` | Kitaev chain rebuilt on the model-independent `Geometry`, tight-binding, pairing, and BdG builders. |
| `KitaevLadder` | Multiple coupled Kitaev chains with leg/rung hopping and pairing and open/periodic directions. |
| `ChiralPWaveModel` | Spinless `p_x + i p_y` or opposite-chirality BdG model on an arbitrary compatible embedded geometry and selected coordinate plane. |

The generic Hamiltonian API additionally implements BdG construction and
spinless p-wave, chiral p-wave, onsite s-wave, d-wave, Rashba, Zeeman, and
disorder terms. These builders are capabilities, not additional named complete
material models. No nanowire, Josephson-junction, or self-consistent BCS model is
currently implemented.

## Implemented geometry generators

All 24 generators below are registered through the common generator protocol:

- Regular/reference geometries: `chain`, `ring`, `square`, `triangular`,
  `honeycomb`, `kagome`, `cubic`, `body_centered_cubic`, `irregular_cluster`.
- Trees and fractals: `tree`, `cayley_tree`, `sierpinski_gasket`,
  `sierpinski_carpet`, `menger_sponge`.
- Quasiperiodic/aperiodic: `fibonacci_chain`, `silver_mean_chain`,
  `ammann_beenker_patch`.
- Seeded random networks: `random_graph`, `random_regular_graph`,
  `small_world_network`, `scale_free_graph`.
- Generic construction rules: `coordinate_cutoff_graph`,
  `k_nearest_neighbor_graph`, `artificial_rule_graph`.

The Phase 6 validation generated every registry entry, checked deterministic
reproduction for equal parameters/seeds, validation, exact serialization, and
canonical-hash stability.

## Available observables

- Spectrum: positive energies, nearest-zero energy, numerical zero-mode count,
  full gap across a reference energy, and backward-compatible edge/bulk/energy
  gap estimates with explicit interpretation warnings.
- State localization: site/component probability, inverse participation ratio,
  participation ratio, boundary/edge/bulk weight, localization classification,
  center of mass, and result-aware basis conversion.
- Local density of states with result-aware wrappers.
- Majorana polarization, particle/hole weights, self-conjugacy and polarization
  norm, plus finite-size zero/split-pair diagnostics.
- Berry curvature and quantized Chern number on a periodic momentum grid.
- Numerical Hermiticity, particle-hole, BdG particle-hole, chiral, and
  time-reversal symmetry checks.
- Standardized immutable observable records and record stacking.

## Available topology diagnostics

- Tenfold-way Altland-Zirnbauer classification and numerical validation of the
  supplied symmetry operators.
- Restricted 1D class-D/BDI Pfaffian invariant at `k=0` and `k=pi`.
- Real-space winding invariant for compatible 1D chiral classes.
- Bott index for finite 2D class-A/C/D Hamiltonians.
- Position-resolved local Chern marker with an explicit bulk mask.
- Two-dimensional spectral localizer and local Chern index.
- Dimension/symmetry-aware topology dispatch and a unified result representation
  that allows diagnostics from different methods to be compared consistently.

A scattering-matrix invariant was investigated in Phase 5.8 but is not
implemented and is therefore not listed as an available calculation.

## Supported embedding dimensions

- `Geometry` accepts no coordinates or any positive coordinate dimension; no
  maximum embedding dimension is hard-coded.
- Explicit repository tests cover dimensions 1, 2, 3, 4, 7, and 11.
- Fixed 3D families are `cubic`, `body_centered_cubic`, and `menger_sponge`.
- `coordinate_cutoff_graph`, `k_nearest_neighbor_graph`, and
  `artificial_rule_graph` construct arbitrary `d >= 1` embeddings.
- Abstract networks and trees may intentionally have no coordinates.
- Embedding dimension is kept separate from lattice, topological, and Hausdorff
  dimension records. A ring, for example, is a 1D lattice embedded in 2D.

Physics routines can impose stricter requirements than the geometry container:
chiral/d-wave directions require a compatible coordinate plane, and current
real-space Chern/Bott/localizer implementations are specifically two-dimensional.

## Reproducible end-to-end demo

Run:

```powershell
python examples/phase_6_capabilities_demo.py
```

The script uses only existing APIs and writes:

- `results/phase_6_capabilities/geometry_families.png`
- `results/phase_6_capabilities/physics_diagnostics.png`
- `results/phase_6_capabilities/numerical_summary.json`

Representative output from the verified run:

| Quantity | Value |
| --- | ---: |
| Open eight-site Kitaev nearest-zero `abs(E)` | `2.089026e-4` |
| Kitaev full finite spectral gap at zero | `4.178052e-4` |
| Next Kitaev excitation | `1.477546` |
| Kitaev split-pair isolation ratio | `7072.89` |
| Nearest-zero-state boundary weight | `0.925926` |
| Majorana polarization norm | `0.999883` |
| 1D Pfaffian invariant | `-1` (topological) |
| BdG particle-hole residual | `0.0` |
| QWZ Bott estimate/index | `1.000000000000001 / 1` |
| QWZ bulk local-Chern estimate/index | `0.997892 / 1` |
| QWZ spectral-localizer index/gap | `1 / 0.692391` |

The selected finite-size Kitaev eigenstate has near-zero global
self-conjugacy even though its polarization norm is near one. This is expected
for an eigensolver-selected member of a split particle-hole pair and illustrates
why energy, localization, Majorana polarization, and an independent topology
invariant must be interpreted together.

These example gaps are finite-system numbers, not thermodynamic bulk-gap proofs.

## What is not implemented yet

Phase 7 and later remain planned work. In particular, the repository does not
yet provide:

- a `GeometryEvaluation` data structure;
- one unified `evaluate_geometry(...)` pipeline that automatically combines
  spectra, states, Majorana diagnostics, topology, descriptors, validity, and
  reproducibility;
- automatic scalar or multi-objective scoring of arbitrary geometry candidates;
- candidate mutation/crossover and evolutionary geometry search;
- dataset generation, ML surrogates, active learning, generative geometry AI,
  or an autonomous discovery engine;
- HPC-scale geometry searches or advanced multidimensional discovery.

The individual solvers and diagnostics already work, as demonstrated above;
their future automated orchestration must not be confused with a current
capability.

## README-ready summary

> **Toposc-Lab after Phase 6** is a tested finite-system toolkit for building,
> solving, and analyzing superconducting and topological Hamiltonians on
> model-independent geometries. It provides generic tight-binding and BdG
> builders, four implemented superconducting model classes, dense exact
> diagonalization, spectral/localization/LDOS/Majorana observables, real- and
> momentum-space topology diagnostics, and 24 deterministic or seeded geometry
> generators spanning abstract graphs and 1D, 2D, 3D, fractal, quasiperiodic,
> random, and arbitrary-dimensional embeddings. Geometry validation, exact
> serialization, reproducible provenance, visualization, and canonical candidate
> hashing are included. Automated geometry evaluation, optimization, datasets,
> machine learning, and autonomous discovery remain Phase 7+ work.
