# Phase 7.8: Multi-objective evaluation

## Decision

Multi-objective evaluation is an immutable, named vector of raw scientific values. It is
derived from `GeometryEvaluation` through explicit `ObjectiveSpec` records and remains
separate from both `GeometryEvaluation` and the Phase 7.7 engineering score.

Each specification declares:

- a unique Python-style name;
- the scientific quantity;
- a caller-selected `maximize` or `minimize` preference;
- exactly one selector when the quantity is state-, topology-, or descriptor-specific.

The supported raw quantities are the finite-system gap, numerical zero-mode count,
state-resolved IPR, state-resolved boundary weight, state-resolved Majorana
self-conjugacy, a method-resolved topology classification, and an individual geometry
descriptor. State indices remain eigensolver column indices. Topology is selected by
`TopologyMethod`, rather than collapsed across methods. Geometry descriptor values retain
their boolean, integer, or floating-point type.

Requested missing or unresolved quantities raise an error. Values are not normalized,
weighted, sign-flipped, or combined, and the direction does not transform them. This
keeps the vector auditable and prevents dimensionful gaps, state diagnostics, topology
classifications, and structural descriptors from acquiring an implicit common scale.

## Scientific boundary

An objective direction is an optimization preference, not a statement that increasing or
decreasing a quantity is universally desirable. In particular, numerical zero modes do
not prove Majorana character, state observables can depend on a selected basis within a
degenerate subspace, topology classifications retain method-specific assumptions, and
geometry descriptors alone do not imply physical performance.

This phase does not define scalar aggregation, Pareto dominance, candidate ranking,
invalid-candidate detection, or a unified evaluation pipeline. Those concerns remain in
their later master-plan steps.
