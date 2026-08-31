# Phase 8.12: Finite-size scaling

## Decision

Phase 8.12 adds a numerical finite-size fit over existing Phase-8.11 robustness
uncertainty estimates. It does not execute disorder ensembles again and does not mutate
or merge their scientific, validity, disorder, or reproducibility records.

`FiniteSizeScalingSpec` gives the size variable a stable key and explicit description
and requires a positive leading correction exponent `p`. Each
`FiniteSizeRobustnessPoint` pairs one caller-supplied positive system size `s` with an
unchanged `RobustnessUncertaintyEstimate`. No size is inferred from site count,
coordinates, graph structure, embedding dimension, metadata, the Phase-6 graph hash, or
the Phase-7.11 geometry snapshot ID.

`FiniteSizeScalingResult` fits the explicit leading-correction model

`R(s) = R_infinity + a * s**(-p)`

by unweighted ordinary least squares. It requires at least three strictly increasing
sizes, one unchanged robustness success criterion, and one confidence-level/method/
version contract across all points. The immutable result retains the ordered input
points and exposes the scaling coordinates, observed and fitted fractions, raw
infinite-size intercept, finite-size coefficient, residuals, residual sum of squares,
and coefficient of determination. `R_squared` is `None` for constant observations,
where the total variation is zero.

The fitted intercept is deliberately not clipped to `[0, 1]`. An out-of-range value is
reported by `infinite_size_intercept_in_unit_interval == False` and is evidence that the
chosen leading-power extrapolation is not physically admissible for those data, not a
value to repair silently.

## Scientific boundary

The numerical fit does not establish that the supplied size variable is physically
appropriate, that a leading power correction is valid, or that the points form one
comparable geometry family. It also does not account for correlations between sizes or
weight the fit by binomial uncertainty. Reusing seeds across sizes may create correlated
observations; this phase neither forbids nor models that choice.

Cross-size geometry-family identity, construction rules, and physical comparability are
Phase 8.13 responsibilities. Even a clean extrapolation over several arbitrary finite
graphs is not evidence of thermodynamic topological behavior. This phase does not infer
topology, create a robustness report, rank candidates, start a search, or generate a
dataset. It uses no RNG and performs no bootstrap resampling.
