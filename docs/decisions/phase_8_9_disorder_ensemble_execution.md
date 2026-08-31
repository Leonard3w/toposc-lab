# Phase 8.9: Disorder ensemble execution

## Decision

A disorder ensemble is deterministic orchestration over a nonempty ordered tuple of
explicit, unique, nonnegative seeds. `DisorderEnsembleRequest` records those seeds
directly. Phase 8.9 does not derive child seeds, create a new random stream, read global
NumPy state, or hide seed relationships behind a process-dependent scheduler.

`execute_disorder_ensemble(...)` executes members sequentially in request order. A
caller-supplied `realization_factory(seed)` owns the concrete Phase-8.2--8.8 disorder
definition and remains subject to the Phase-8.1 PCG64/provenance contract. Successful
members must share the exact source snapshot, transform key and version, transform
parameters, and RNG algorithm. Only their explicit seed and resulting snapshot may
differ. This prevents one nominal ensemble from silently mixing sources or disorder
definitions.

## Separated member results

`DisorderOutcome` is a read-only structural interface exposing only `state` and
`provenance`. It admits the common `DisorderRealization` and specialized immutable
outcomes such as `NodeRemovalRealization`, so node-reindex information is not discarded.

Each `DisorderEnsembleMember` stores the request index and seed, the disorder outcome,
an optional unchanged `GeometryEvaluationRun`, and an optional ensemble-owned execution
failure in distinct fields. When supplied, an `evaluation_factory(disorder, seed)` owns
the explicit scientific interpretation and Phase-7 pipeline invocation. The ensemble
does not choose a model adapter, Hamiltonian, topology method, or validity policy.

A returned invalid `GeometryEvaluationRun` remains a completed scientific/validity
result and is not relabeled as an ensemble failure. Ordinary exceptions raised by a
realization or evaluation callback are recorded with stage, exception type, and message;
exception objects and tracebacks are not retained, and later seeds continue. API contract
violations such as a wrong return type, mismatched provenance seed, or mixed disorder
definitions raise immediately. `KeyboardInterrupt` and other `BaseException` control
signals are not caught.

`DisorderEnsembleResult` records whether evaluations were requested and the versioned,
ordered member tuple. It intentionally exposes no aggregate success count, fraction,
mean, variance, confidence interval, robustness score, ranking, or statistical
interpretation. Those belong to Phase 8.10 and later steps.

This phase does not add parallel scheduling, statistical uncertainty, finite-size
analysis, robustness reports, automated search, or datasets.
