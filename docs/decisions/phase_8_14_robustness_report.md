# Phase 8.14: Robustness report object

## Decision

Phase 8.14 adds an immutable composition layer over the already-computed Phase-8
objects. It performs no new disorder realization, evaluation, success classification,
uncertainty estimation, finite-size fit, or geometry-family inference.

`RobustnessReportEntry` binds one unchanged `DisorderEnsembleResult` to its
`RobustnessFractionMetric` and `RobustnessUncertaintyEstimate`. The ensemble request and
seed order must match the metric exactly, the metric's execution-failure indices must
equal the ensemble's actual operational failures, and the uncertainty object must refer
to that exact metric object. Completed ensemble members are not reclassified because the report does
not retain or recreate the caller's success predicate.

`RobustnessReport` has a stable caller-supplied key, description, ordered entries, and a
versioned report contract. It supports two explicit shapes:

- a single-size report with exactly one entry and no scaling claim;
- a cross-size report with one entry for every Phase-8.12 point in exact order, together
  with both the matching scaling result and Phase-8.13 geometry family.

Cross-size object links use identity, not merely value equality, and the same entry or
ensemble object cannot stand for multiple system sizes. Where completed disorder
realizations provide provenance, entries must share the transform key and version, RNG
algorithm, target, source/result snapshot schemes, and parameter-key schema. Full
parameter values and source snapshot IDs remain in their entries: legitimate
size-specific records such as `n_sites` and `geometry_id` must not be forced equal or
silently removed.

Entries and reports intentionally use identity rather than structural dataclass equality.
An ensemble may contain NumPy-backed scientific results, while excluding the ensemble
from equality would make distinct provenance compare equal. Contract validation instead
compares only safe, explicitly selected relationship fields.

The report derives per-entry success counts, denominators, fractions, Wilson intervals,
and operational failure counts. It deliberately exposes no pooled fraction across
different system sizes. Deterministic warnings state the success-criterion boundary,
surface execution failures, distinguish a single finite system from cross-size evidence,
and flag an out-of-range infinite-size intercept without clipping it.

## Separation and interpretation boundaries

The report retains references rather than flattening source data. Disorder realizations
and their provenance, optional `GeometryEvaluationRun` objects, `SimulationResult`,
`GeometryEvaluation`, candidate validity records, stage failures, and Phase-7.11
reproducibility records remain in their existing typed layers beneath each ensemble
member. Their warnings are not copied into one untyped report list.

The report validates internal relationships that its inputs make observable. It cannot
reconstruct the success predicate for completed members, prove that cross-size ensembles
share a physically identical disorder protocol, or prove that every evaluation used the
family geometry associated by the caller. The Phase-6 graph fingerprint remains only an
isomorphism candidate, and the exact geometry ID remains a representation-sensitive
snapshot identifier.

Neither a high success fraction nor a clean fitted trend establishes topology,
protected boundary modes, or a thermodynamic phase. Phase 8.14 does not introduce a new
score, ranking, persistence format, automated search, candidate generation, dataset, or
Phase-9 behavior. It uses no RNG.
