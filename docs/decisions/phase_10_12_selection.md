# Phase 10.12 — Selection

## Status and scope

Accepted as the Phase-10.12 architecture for selecting an ordered set of
references to fitness-bearing population members.

This phase adds no elitism, mutation, crossover, offspring construction, next
generation, generation loop, diversity or novelty rule, fitness sharing,
archive, checkpoint, machine-learning behavior, or dataset functionality.

## Minimal neutral contract

`select_population_members(...)` accepts one immutable
`PopulationFitnessResult`, one `TournamentSelectionConfig`, and one explicit
nonnegative seed. It returns references to existing `PopulationFitnessMember`
objects; it neither copies genomes nor constructs a new population.

The baseline mechanism is finite tournament selection. `selection_count`
declares the number of output slots and `tournament_size` declares the number
of contestants in each slot. Contestants are sampled without replacement
inside one tournament. Tournaments are independent, so the same source member
may be selected in more than one output slot. This is selection with
replacement across tournaments, not population reproduction.

## Eligibility and failure accounting

Only members with `available` Phase-10.11 fitness are eligible. Invalid
evaluations, evaluator failures, and fitness-construction failures retain their
original positions and diagnostics in the source ledger but are never silently
assigned a score or entered into a tournament. Selection rejects an empty
eligible pool and a tournament size larger than that pool.

## Comparison rule

For `ScalarFitnessDefinition`, the stored scalar direction is authoritative:
the largest value wins under `maximize` and the smallest under `minimize`.
Exact scalar ties remain ties; this phase introduces no numerical tolerance.

For `MultiObjectiveFitnessDefinition`, a contestant is preferred only by
Pareto dominance using every objective's stored direction. All nondominated
contestants in the tournament form the winner set. No scalarization,
normalization, objective weighting, lexicographic priority, or crowding metric
is introduced.

An exact scalar tie or a multi-objective nondominated set is resolved uniformly
through the explicit random stream. Consequently, incomparable objective
vectors are not misrepresented as an ordered scientific ranking.

## Reproducibility and audit ledger

The versioned implementation uses one local `numpy.random.PCG64` generator.
For every output slot it first draws the ordered contestant set and then draws
one winner position by calling `Generator.integers` with the size of the best
set. For a unique best contestant that call deterministically returns zero and
NumPy may leave the generator state unchanged. The exact ordered record
sequence, RNG identifier, and implementation version therefore jointly define
reproducibility; no stronger cross-version random-stream claim is made.

Every `TournamentSelectionRecord` stores its selection index, ordered exact
contestant references, and exact winner reference. `PopulationSelectionResult`
retains the source, configuration, seed, RNG identifier, version, complete
record sequence, and explicit limitations. Its constructor replays the
versioned algorithm and rejects altered contestant sets, winners, or order.

## Elitism and scientific boundary

Tournament selection does not guarantee that the globally best available
member appears in the output. Guaranteed survival is an explicit Phase-10.13
elitism decision and is not anticipated here.

A selection outcome is an engineering search decision. It does not satisfy the
research charter's topology, spectral-protection, boundary, robustness,
reference-matching, finite-size, or leakage gates and is not evidence for a
scientific discovery claim.
