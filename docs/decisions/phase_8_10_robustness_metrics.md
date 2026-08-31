# Phase 8.10: Robustness metrics

## Decision

Phase 8.10 implements the empirical success fraction

`R_robust = N_successful / N_total`

over a completed `DisorderEnsembleResult`. `N_total` is always the number of explicitly
requested seeds. Realization and evaluation-callback failures remain in the denominator
and are unsuccessful. They are never silently excluded, because doing so would bias the
reported fraction toward members that happened to execute successfully.

`RobustnessSuccessCriterion` requires a stable key, human-readable description, and an
explicit caller-supplied predicate over one `DisorderEnsembleMember`. No default
criterion is inferred. In particular, the metric layer does not parse warnings, guess a
topology method, treat graph connectivity as physical success, select a gap threshold,
or automatically equate Phase-7 validity with topological robustness. A caller may use
typed fields such as `member.evaluation.is_valid` or a named topology result, but that
scientific policy must be visible in its criterion.

The predicate is evaluated only for members without an ensemble execution failure and
must return an actual boolean. Predicate exceptions and invalid result types are API or
criterion errors and propagate instead of being converted into additional ensemble
samples.

## Metric result boundary

`RobustnessFractionMetric` records the criterion identity and description, original
ensemble request, one ordered boolean decision per seed, and the indices of execution
failures. It derives the successful count, total count, fraction, and successful,
unsuccessful, and execution-failure seed tuples transparently. Operational failures stay
distinguishable while remaining part of the unsuccessful population.

This immutable value is one metric result, not the Phase-8.14 robustness report. It does
not copy or merge `SimulationResult`, `GeometryEvaluation`, validity reports, disorder
provenance, or evaluation reproducibility records; those remain in the ensemble members.

No standard error, variance, confidence interval, bootstrap, hypothesis test,
finite-size scaling, cross-size aggregation, ranking, automated search, or dataset
generation is introduced in this phase. Statistical uncertainty begins at Phase 8.11.
