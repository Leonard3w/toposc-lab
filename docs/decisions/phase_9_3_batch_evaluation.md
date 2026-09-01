# Phase 9.3: Batch evaluation

## Status

Accepted.

## Decision

Batch evaluation lives in `toposc_lab.search.batch_evaluation` as deterministic
orchestration over a nonempty ordered tuple of explicit candidates. A
`BatchEvaluationCandidate` contains one immutable `Geometry`, one deeply immutable
Phase-9.2-compatible `ModelParameterSet`, and an optional explicit evaluation seed.
The batch creates no random stream, derives no child seeds, and changes none of these
inputs.

Geometry and model-parameter sampling remain separate. Phase 9.3 does not zip, form a
Cartesian product, resample, or otherwise choose how Phase-9.1 and Phase-9.2 outputs are
paired. A caller constructs the requested candidate sequence explicitly.

`execute_evaluation_batch(...)` receives a caller-owned
`evaluator(candidate) -> GeometryEvaluationRun`. The evaluator owns parameter-model
validation, `GeometryModelAdapter`, solver and evaluation settings, optional topology
hooks, and the call to the established `evaluate_geometry(...)` pipeline. The batch
does not select or infer a physical model, Hamiltonian, basis, physical dimension,
topology method, validity policy, or scientific threshold.

Candidates execute sequentially in request order. Each `BatchEvaluationMember` retains
the exact requested candidate and contains exactly one of:

- the unchanged `GeometryEvaluationRun`, whether scientifically valid or invalid; or
- a structured `BatchEvaluationFailure` for an ordinary exception raised by the
  evaluator callback.

Callback failures record stage, exception type name, and message without retaining an
exception object or traceback. Later candidates continue. `KeyboardInterrupt` and other
`BaseException` control signals are not caught. A callback returning the wrong type is
an API-contract error and raises immediately rather than being mislabeled as candidate
invalidity.

## Correspondence checks

When an evaluation has a Phase-7.11 `ReproducibilityRecord`, the batch verifies that its
exact geometry snapshot ID, type-sensitive model-parameter snapshot, and evaluation seed
match the requested candidate. A mismatch is an evaluator contract violation and raises
immediately. Evaluations rejected before a reproducibility record can be created remain
valid batch outputs; their `GeometryEvaluationRun.validity` and optional stage failure
already express why the Phase-7 pipeline stopped.

The geometry ID is used only to bind an evaluation to its exact representation-sensitive
input. It is not a canonical physical identity. The Phase-6 graph hash is not used, and
Phase 9.3 performs no deduplication.

## Scientific and roadmap boundaries

`BatchEvaluationResult` exposes only the versioned ordered member tuple. It has no
aggregate count, success fraction, statistic, score, ranking, selection, visualization,
or interpretation. It writes no files or datasets and performs no candidate persistence;
Phase 9.4 remains the owner of storing every evaluated candidate.

The batch does not hard-code the first chiral-p-wave experiment or weaken the research
charter. A later frozen protocol must provide model applicability, resource matching,
shared parameter policy, topology evidence, disorder roles, and seed separation. Phase
9.3 merely executes the explicitly supplied evaluator once per explicitly supplied
candidate and preserves every in-memory outcome in order.
