# Phase 9.7: Baseline statistics

## Status

Accepted.

## Decision

Baseline statistics live in `toposc_lab.search.baseline_statistics`. The module consumes
a nonempty ordered collection of explicitly named `SearchBaselineTrial` objects. Each
trial retains one completed Phase-9.5 `CandidateRankingResult`; Phase 9.7 does not
sample geometries or parameters, execute evaluations, create seeds, rank candidates,
or claim that the inputs came from a random process.

The intended Phase-9 use is to summarize repeated random-search attempts after a later
experiment protocol has fixed their construction. The API remains general because no
Phase-9.8 search execution or concrete scientific policy is introduced here.

`BaselineSuccessCriterion` requires a stable key, a human-readable description, and a
caller-owned predicate over one ranked `CandidateRankingEntry`. The predicate defines
what `strong candidate` means. The library supplies no score component, weight,
threshold, topology rule, robustness rule, or default scientific success condition.

The predicate is evaluated exactly once for every ranked candidate in original
Phase-9.3 batch order. Evaluation does not stop after the first success. Invalid
evaluation runs and Phase-9.3 callback failures are assigned unsuccessful decisions,
remain in the candidate denominator, and retain separate index and count fields. They
are never passed to the predicate and are not reinterpreted as scientifically weak
valid candidates.

## Trial and denominator contract

A search trial is successful when at least one of its candidates satisfies the explicit
criterion. `SearchBaselineTrialOutcome` retains every candidate decision in source
batch order, the successful candidate indices, invalid-candidate indices, and callback-
failure indices.

All trials in one `SearchBaselineStatistics` result must have unique trial keys, the
same Phase-9.5 ranking configuration, and the same number of requested candidates.
This prevents silent mixing of different criterion orders or per-trial search budgets
without choosing a concrete ranking policy or budget. Compatibility is checked before
the first predicate call.

The primary baseline quantity is

`successful search trials / all requested search trials`.

The result also reports the raw candidate success fraction and separate invalid and
callback-failure counts. The candidate fraction is descriptive: candidates within and
across trials need not be independent, so Phase 9.7 does not attach a candidate-level
binomial interval to it.

The trial success fraction receives a two-sided Wilson score confidence interval and
plug-in binomial standard error at a caller-supplied confidence level. This uses the same
mathematical convention as Phase 8.11 but a distinct typed object because search trials
are not disorder realizations. No fake disorder request, seed, or robustness metric is
constructed.

Predicate exceptions, non-boolean decisions, malformed inputs, duplicate trial keys,
mixed ranking configurations, unequal trial budgets, or invalid confidence levels are
API or protocol errors. They abort the call without returning partial statistics.

## Statistical and scientific boundaries

The Wilson interval has its repeated-trial interpretation only if the supplied trials
can be treated as representative independent attempts under one frozen search
protocol. The API records trial order and result identity but does not prove
independence, random-search provenance, seed separation, absence of adaptive tuning,
or resource matching.

A high hit rate is an engineering baseline, not evidence of topology, a bulk or
mobility gap, protected boundary physics, Majorana character, robustness, novelty,
causal geometry advantage, or discovery. Operational failures stay in the denominator
and must be inspected separately. No hypothesis test, p-value, cross-method comparison,
finite-size inference, reference-family advantage, or thermodynamic conclusion is
computed.

Phase 9.7 introduces no concrete success threshold, confidence level, search budget,
trial count, seed list, seed split, model parameters, reference family, or ranking
policy. These must be fixed in the separate versioned protocol required before Phase
9.8. This phase performs no reproducible random-search experiment, persistence,
visualization, dataset generation, evolutionary search, mutation, ML, GNN, active
learning, reinforcement learning, or generative work.
