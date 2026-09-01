# Phase 9.5: Candidate ranking

## Status

Accepted.

## Decision

Candidate ranking lives in `toposc_lab.search.candidate_ranking` and consumes one
completed in-memory Phase-9.3 `BatchEvaluationResult`. It does not load the reduced
Phase-9.4 candidate ledger because that audit archive intentionally excludes numerical
scientific payloads.

The caller must provide a nonempty ordered `CandidateRankingConfig`. Every
`CandidateRankingCriterion` declares a unique name, the existing Phase-7.8
`ObjectiveDirection`, and an explicit boolean, integer, or real value kind. The caller
also supplies a `value_factory(member)` that returns exactly one finite typed value for
every configured criterion. There are no default criteria, weights, normalizations,
thresholds, score components, or physical-model assumptions.

The value factory can reuse existing representations without the ranking layer
duplicating them. For example, it may expose the value of a caller-configured Phase-7.7
`BasicScalarScore`, values from a Phase-7.8 `MultiObjectiveEvaluation`, raw evaluation
quantities, resource controls, or later robustness quantities. Comparability remains the
caller's responsibility and requires identical criteria and value-construction
conventions.

## Ordering and failure contract

Only members containing a valid `GeometryEvaluationRun` are supplied to the value
factory. Invalid evaluation runs and Phase-9.3 callback failures remain present in the
batch-order `entries` tuple with an explicit `UnrankedCandidateReason`, no artificial
score, and no rank. They are not silently discarded and are not interpreted as
scientific low performers.

Rankable candidates are ordered lexicographically by the declared criterion order.
Each criterion independently uses its explicit maximize or minimize direction. Exact
ties receive the same one-based competition rank, so a tie sequence is `1, 1, 3`.
Original candidate order is the deterministic final tie-break for the ordered view; it
does not break the shared rank. The implementation creates no RNG and reads no global
random state.

The value factory is called exactly once per valid candidate in original batch order.
Missing or additional keys, type mismatches, non-finite real values, wrong callback
return types, and callback exceptions are API or ranking-policy failures. They abort the
whole call without returning a partial ranking and are never converted into candidate
invalidity.

`CandidateRankingResult.entries` retains every exact Phase-9.3 member in batch order.
`ranked_entries` is a second immutable view containing every and only ranked entry in
configured order. No candidate is copied, resampled, reevaluated, mutated, or persisted.

## Scientific and roadmap boundaries

Ranking is an engineering ordering, not a physical observable, topological invariant,
or scientific discovery. A high rank does not satisfy the research charter's topology,
spectral-protection, boundary-state, robustness, resource-matching, finite-size, or
independent-seed evidence gates. Geometry structure, coordinate dimension, a numerical
zero mode, boundary localization, Majorana self-conjugacy, or the Phase-7 finite-system
gap is not independently promoted to a topology claim.

Phase 9.5 introduces no experiment-specific policy. Exact criteria, weighting or
lexicographic priority, eligibility thresholds, reference strata, protection quantity,
and seed roles must be frozen before Phase 9.8 as required by the research charter.

This phase performs no top-candidate selection or visualization, candidate persistence,
baseline statistics, full random-search experiment, dataset generation, evolutionary
search, mutation, ML, GNN, active learning, reinforcement learning, or generative work.
Those remain owned by their later numbered master-plan steps.
