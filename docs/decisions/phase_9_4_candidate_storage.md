# Phase 9.4: Store every evaluated candidate

## Status

Accepted.

## Decision

Phase 9.4 persists one completed Phase-9.3 `BatchEvaluationResult` as one ordered,
versioned candidate-ledger archive. The writer neither samples nor evaluates anything.
It stores every member in exact request order, including invalid
`GeometryEvaluationRun` values and evaluator-callback failures. Existing target files
are rejected rather than silently replaced.

Each candidate record contains:

- the exact Phase-6.23 geometry archive and its Phase-7.11 snapshot ID;
- the deeply immutable, type-sensitive model-parameter snapshot and its existing exact
  parameter-set ID;
- the explicit evaluation seed, including `None` when no seed was supplied;
- a terminal outcome distinguishing valid evaluation, invalid evaluation, and callback
  failure;
- for evaluation runs, the complete validity report, optional pipeline-stage failure,
  optional reproducibility record, and flags recording whether numerical simulation and
  scientific evaluation payloads existed;
- for callback failures, the Phase-9.3 failure stage, exception type name, and message.

The writer does not use the Phase-6 graph hash, deduplicate candidates, infer physical
equivalence, canonicalize edge orientation, add coordinates or boundaries, or classify
scientific merit. Exact geometry IDs remain representation-sensitive snapshot IDs.

## Archive and integrity contract

The public writer produces an atomic `.zip` archive with a strict UTF-8 JSON manifest
and one existing pickle-free `.npz` geometry archive per candidate. Member names,
ordering metadata, schema version, identifier schemes, typed parameter values, and
terminal-outcome structure are explicit. Archive timestamps are fixed, so equivalent
inputs produce identical bytes. The loader applies compressed-container size and member
limits, rejects duplicate or unexpected members, never enables NumPy pickle loading,
revalidates every geometry, and recomputes exact geometry and model-parameter IDs.

The archive has no random-number generator and derives no seeds. Its order is inherited
unchanged from Phase 9.3.

## Phase boundary

This ledger retains candidate inputs and execution/audit outcomes. It deliberately does
not define persistent schemas for Hamiltonians, eigenvalues, eigenvectors, observables,
topology results, robustness arrays, ranking values, datasets, duplicates, or data
splits. The original in-memory `BatchEvaluationResult` still contains the complete
scientific outputs needed by later Phase-9 processing. General numerical-result and
dataset persistence remains owned by Phase 11.

No score, threshold, aggregation, ranking, top-candidate selection, visualization,
baseline statistic, experiment budget, or chiral-p-wave-specific policy is introduced.
Consequently invalid and unsuccessful candidates remain visible without turning this
infrastructure step into the Phase-9.5 ranking or the Phase-9.8 experiment protocol.
