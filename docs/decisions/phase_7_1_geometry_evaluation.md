# Phase 7.1: `GeometryEvaluation` result structure

## Status

Accepted.

## Context

Phase 6 established generic geometry construction and a matrix-level path to exact
diagonalization. The project already has separate result types for localization,
Majorana diagnostics, and topology, but no aggregate that can represent the scientific
evaluation of one geometry. Phase 7.1 introduces that aggregate only; it does not add an
evaluation pipeline or calculate any new quantity.

## Decision

`GeometryEvaluation` lives in the new `toposc_lab.evaluation` package, above geometry,
models, solvers, observables, and topology in the dependency direction. It is a frozen,
validated result container with explicit missing values and no model-building or numerical
logic.

Low-energy states are recorded as a mapping from eigensolver column index to eigenenergy.
IPR values, localization profiles, and Majorana diagnostics use the same indices, making
state correspondence explicit without embedding solver output or eigenvectors in the
aggregate. Localization, Majorana, and topology fields reuse their existing scientific
result types rather than flattening or duplicating them.

Topology is a tuple because more than one independent diagnostic may be appropriate. At
most one result per topology method is permitted. Geometry descriptors are restricted to
named scalar values; their extraction remains Phase 7.6. Warnings stay separate from all
scientific quantities and future scores.

## Consequences

- Geometry and physical-model objects remain unaware of evaluation concerns.
- The container can represent partial evaluations without inventing placeholder numbers.
- Later Phase 7 steps can populate individual fields without changing their ownership.
- The meaning of `gap` remains the responsibility of the future spectral evaluator; the
  data structure does not silently choose a bulk, excitation, or full spectral-gap
  convention.
- Phase 7.1 does not introduce `evaluate_geometry`, scoring, reproducibility metadata, or
  geometry descriptor computation.
