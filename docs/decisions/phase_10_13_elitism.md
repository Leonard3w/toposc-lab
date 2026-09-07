# Phase 10.13 — Elitism

## Status and scope

Accepted as the Phase-10.13 architecture for declaring which complete leading
fitness tiers must be retained by a future generation-composition step.

This phase adds no crossover, mutation execution, offspring construction, next
population, generation loop, diversity or novelty policy, checkpoint,
machine-learning behavior, or dataset functionality.

## Minimal neutral contract

`identify_population_elites(...)` accepts one immutable
`PopulationFitnessResult` and an `ElitismConfig` containing a positive
`minimum_elite_count`. It returns exact references to existing fitness members;
it does not copy genomes or construct survivors.

Only Phase-10.11 members with available fitness are eligible. Invalid
evaluations, evaluator failures, and fitness-construction failures retain their
original source-ledger positions and diagnostics but cannot enter an elite
tier. An empty eligible pool or a requested minimum larger than that pool is
rejected.

## Complete cutoff tiers

The configured value is a minimum rather than an exact count. Elitism includes
complete ordered fitness tiers until at least that count is reached. If the
cutoff intersects a tie or Pareto front, the entire tier is retained and the
actual elite count exceeds the configured minimum.

This rule avoids random or source-order truncation of candidates that the
declared fitness contract cannot distinguish. It also makes the capacity cost
explicit for the later generation-loop design instead of hiding a preference
inside Phase 10.13.

## Scalar and multi-objective ordering

For `ScalarFitnessDefinition`, exact equal values form a tier. Tiers are
ordered by the stored direction: descending under `maximize` and ascending
under `minimize`. No numerical tolerance is introduced. Members within a tier
retain source-population order.

For `MultiObjectiveFitnessDefinition`, tiers are successive nondominated
Pareto fronts using every objective's stored direction. Removing one front and
recomputing nondominance produces the next. No scalarization, normalization,
weighting, lexicographic priority, crowding distance, diversity heuristic, or
random tie-break is introduced. Members within each front retain source order.

## Audit and generation boundary

Each `EliteTier` stores its zero-based tier index and exact member references.
`PopulationElitismResult` retains the source, configuration, selected complete
tiers, actual count, cutoff-expansion state, version, and limitations. Its
constructor recomputes the deterministic tier contract and rejects missing,
reordered, or altered tiers.

The result is a retention requirement only. It does not merge elites with the
Phase-10.12 selection output, reserve next-generation slots, clone genomes, or
claim that survival has already occurred. Those composition responsibilities
belong to the later generation-loop phase after a scientifically meaningful
crossover decision.

## Scientific boundary

Elite status is an engineering search label. It does not satisfy the research
charter's topology, spectral-protection, boundary, robustness,
reference-matching, finite-size, or leakage gates and is not evidence for a
scientific discovery claim.
