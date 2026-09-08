# Phase 10.17 — Novelty score

## Status and scope

Accepted as the Phase-10.17 architecture for measuring how sparse each exact
population member is under one caller-declared geometry-distance contract.

This phase adds no built-in geometry or physical distance, fitness adjustment,
selection objective, elite rule, family balancing, offspring behavior, persistent
archive, checkpoint, resume behavior, machine-learning behavior, or dataset
functionality.

## Research decision

Classical novelty search characterizes behavior in a domain-specific space and
measures sparseness as the mean distance to the `k` nearest neighbors. Its comparison
set includes the current population and a permanent archive of previously novel
behaviors ([Lehman and Stanley, 2011](https://doi.org/10.1162/EVCO_a_00025)). The
original conference formulation likewise searches for behavioral novelty instead of
objective progress ([Lehman and Stanley, 2008](https://doi.org/10.1162/isal.2008.11.0.382)).

Toposc-Lab does not yet have a representation-neutral answer to what behavior means
for a physical geometry. Graph-edit distance, scalar descriptor distance, coordinate
RMSD, objective-space distance, spectral distance, or a weighted hybrid would each
encode different invariances and scientific priorities. The pre-Phase-9 charter also
requires novelty analysis to consider graph isomorphism, coordinate symmetries,
relabeling, orientation, construction rules, and physical deformations; neither the
Phase-6 graph fingerprint nor Phase-7 exact snapshot ID satisfies that requirement.

Phase 10.17 therefore implements the classical `k`-nearest mean calculation but no
default distance. A caller supplies the domain-specific function and a versioned
identifier frozen by the experiment protocol. The implementation checks each returned
value for real, finite, nonnegative form, but it does not certify symmetry, triangle
inequality, physical equivalence, or scientific validity of the supplied rule.

## Minimal neutral contract

`evaluate_population_novelty(...)` accepts one exact ordered initial or later
generation population, a positive `neighbor_count`, a caller-supplied distance
callback, its nonempty versioned identifier, and an optional
`NoveltyReferenceCohort`. For every candidate it evaluates distances in this order:

1. every other current-population slot in population order; then
2. every reference-cohort genome in declared cohort order.

Only the candidate's own slot is excluded. Another slot containing the same exact
genome remains a comparison and can contribute zero distance under an appropriate
caller contract. Reference duplicates also remain distinct observations. The
library performs no hash-based deduplication or equivalence inference.

Comparisons are ordered for nearest-neighbor selection by distance, then population
before reference cohort, then source index. The tie order is audit information and
does not change the mean when tied distances are equal. The score is computed with
`math.fsum` divided by exactly the declared `neighbor_count`.

The full requested `k` must exist for every candidate. A one-member population with
no references, or any comparison set smaller than `k`, is rejected. The implementation
does not silently reduce `k`, change the denominator, invent a self-distance, or
assign a missing score.

## Reference-cohort and archive boundary

`NoveltyReferenceCohort` is an explicitly named immutable tuple supplied to one
evaluation. It can represent a predeclared historical comparison set, but Phase 10.17
does not select members for it, apply an admission threshold, update it, serialize it,
or carry it between calls. Persistent checkpoint and later resume state remain Phases
10.18 and 10.19.

## Failure and audit behavior

Every `PopulationNoveltyMember` retains its exact population member, exact geometry
ID, complete ordered comparison ledger, deterministic nearest-neighbor subset, and
raw score. Every comparison retains its source kind, source index, exact genome,
exact geometry ID, and validated distance. `PopulationNoveltyReport` recomputes the
source layout, nearest-neighbor subset, and mean and rejects altered records.

Ordinary distance-callback exceptions become `NoveltyDistanceEvaluationError`
instances bound to the distance identifier, generation, candidate member, comparison
source, and comparison index. `BaseException` subclasses are not swallowed. Invalid
callback values become `InvalidNoveltyDistanceError`. No comparison is retried and no
partial report is returned.

## Search and scientific boundary

The output is a separate engineering-search signal. It is not automatically inserted
into scalar fitness or a multi-objective vector and does not alter Phase-10.12
selection, Phase-10.13 elitism, Phase-10.16 family occupancy, or the generation loop.
An experiment that later rewards novelty must predeclare how it uses this raw score
without retroactively tuning the policy from outcomes.

A high novelty score proves only sparsity under the declared comparison set and
distance. It does not prove physical distinctness, literature priority, topology,
spectral protection, Majorana character, robustness, resource fairness, or a causal
geometric mechanism and satisfies none of the charter's scientific evidence gates.
