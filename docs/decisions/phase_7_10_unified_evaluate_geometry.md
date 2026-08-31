# Phase 7.10: Unified `evaluate_geometry(...)` pipeline

## Decision

The unified pipeline is an explicit orchestrator, not a new physical model. A
`GeometryModelAdapter` supplies the only authorized mapping from an arbitrary `Geometry`
to a `BaseModel`, declares model-specific geometry requirements, and can optionally
resolve the model's `NambuBasis`. The pipeline never chooses a Hamiltonian, pairing
convention, physical dimension, symmetry class, or topology method from graph structure.

`evaluate_geometry(...)` executes these stages in order:

1. model-independent and explicitly configured model-geometry preflight validation;
2. model construction and basis acquisition;
3. one Hamiltonian construction and pre-solver numerical validation;
4. exact diagonalization and post-solver residual validation;
5. spectral and state-resolved evaluation;
6. optional basis-explicit Majorana diagnostics;
7. geometry descriptor extraction;
8. optional method-specific topology hook followed by applicability-aware integration;
9. final candidate validation under the complete policy.

Topology work receives a `GeometryEvaluationContext` containing the separate geometry,
model, Hamiltonian, `SimulationResult`, and current scientific evaluation. A hook and its
`TopologyDispatchDecision` must be supplied together. This keeps method-specific inputs
and physical applicability outside the generic orchestrator.

## Failure and result semantics

The immutable `GeometryEvaluationRun` keeps numerical output (`SimulationResult`),
scientific output (`GeometryEvaluation`), validity (`CandidateValidityReport`), and an
optional serializable `CandidateStageFailure` in separate fields. Invalid preflight or
numerical checks stop later stages. Ordinary exceptions raised during candidate-specific
model, Hamiltonian, solver, evaluation, or topology work are converted to the matching
stage failure; `KeyboardInterrupt` and other non-`Exception` control signals are not
captured. API configuration errors are raised before candidate execution.

Majorana and topology omissions remain explicit evaluation warnings. They are not
automatically invalid unless the final validity configuration requires topology evidence.
No warning string is parsed to make a validity decision.

This phase does not compute a scalar score, rank objectives, store candidates, infer
topological applicability, or add seeds, geometry IDs, solver settings, code versions, or
other Phase 7.11 reproducibility records.
