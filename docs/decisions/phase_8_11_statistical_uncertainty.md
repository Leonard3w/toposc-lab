# Phase 8.11: Statistical uncertainty

## Decision

Phase 8.11 treats the ordered boolean decisions in one `RobustnessFractionMetric` as
Bernoulli observations and quantifies sampling uncertainty without rerunning or
resampling the disorder ensemble. `RobustnessUncertaintyEstimate` retains the complete
metric, an explicit confidence level, the named interval method, and an uncertainty
contract version. This preserves the relationship to the original seeds, success
criterion, execution failures, and ensemble request without copying scientific results
or disorder provenance into a statistical object.

The estimate exposes the observed robustness fraction and the plug-in binomial standard
error

`sqrt(R_robust * (1 - R_robust) / N_total)`.

Its two-sided confidence bounds use the Wilson score interval. Wilson bounds remain in
`[0, 1]` and do not collapse to a zero-width interval merely because a finite ensemble
observed zero or all successes. The confidence level is explicit, finite, and strictly
between zero and one. The method is recorded as `wilson_score`; the initial statistical
contract is version 1.

## Interpretation boundary

The calculation uses all requested seeds exactly as Phase 8.10 classified them.
Realization and evaluation-callback failures therefore remain unsuccessful observations
in the denominator and remain inspectable through the retained metric. If such failures
are present, the interval describes the recorded composite success indicator; it must
not be silently reinterpreted as uncertainty from physical disorder alone.

A confidence interval has its usual repeated-sampling interpretation only when the
seeded realizations can be treated as representative independent draws from one common
disorder distribution and the success criterion was fixed independently of the observed
outcomes. The API records the numerical method but does not prove these assumptions,
infer topology, choose a scientific success criterion, or turn the interval into a
probability that a particular geometry is topological.

No global or local RNG is used in this phase. Bootstrap resampling, Bayesian intervals,
hypothesis tests, finite-size scaling, cross-size family handling, robustness reports,
ranking, search, and dataset generation remain outside Phase 8.11.
