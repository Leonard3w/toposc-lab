# Phase 10.11 — Fitness evaluation

## Status and scope

Accepted as the Phase-10.11 architecture for producing an ordered,
failure-aware fitness ledger from an `InitialPopulation`.

This phase adds no ranking, selection, elitism, mutation sampling, crossover,
generation loop, diversity rule, novelty score, checkpoint, machine-learning
behavior, or dataset functionality.

## No new scientific formula

Phase 10.11 does not invent a new fitness equation. It reuses the existing
Phase-7 evaluation products through one of two explicit definitions:

- `ScalarFitnessDefinition` constructs the existing `BasicScalarScore` from a
  predeclared positive weight map, scalar optimization direction, and gap
  scale; or
- `MultiObjectiveFitnessDefinition` constructs the existing
  `MultiObjectiveEvaluation` from a predeclared ordered set of `ObjectiveSpec`
  records.

The scalar form remains an auditable engineering convenience and not a physical
observable or topological invariant. The multi-objective form preserves raw
quantities and their caller-selected directions without normalization or
aggregation. The two forms cannot be mixed inside one result.

## Evaluation boundary

`evaluate_population_fitness(...)` accepts an immutable initial population, one
fitness definition, and a caller-supplied evaluator. The evaluator receives
each `InitialPopulationMember` exactly once in population order and must return
a `GeometryEvaluationRun`.

The orchestration owns no model selection, physical parameters, disorder
protocol, seed generation, solver configuration, or topology policy. Those
choices belong to the evaluator and must be retained by the existing
reproducibility record. When that record is present, its exact geometry ID must
match the member genome before fitness construction.

## Outcome and failure accounting

Every source member produces exactly one `PopulationFitnessMember` with one of
four explicit states:

1. `available` — a valid evaluation and completed fitness payload;
2. `invalid_evaluation` — the evaluation pipeline returned a structured invalid
   run;
3. `evaluation_failure` — the evaluator callback raised an ordinary exception;
4. `fitness_construction_failure` — a valid run lacked or rejected a requested
   fitness quantity.

Ordinary per-member failures are serialized without exception objects or
tracebacks, and later members continue. Keyboard interrupts and other
`BaseException` subclasses are not swallowed. Wrong callback return types,
geometry-identity mismatches, malformed definitions, and inconsistent result
records are API-contract violations and raise immediately.

Invalid or failed members remain in the population-order ledger with no fitness
payload. They are never silently removed or assigned zero, negative infinity,
or another artificial worst value. The result exposes available members only
as a convenience view without renumbering them.

## Comparability and selection boundary

All available payloads are cross-checked against the one stored definition.
Scalar components, normalized weights, and gap scale must agree; multi-objective
specifications and order must agree. Direct fitness comparison additionally
requires the same underlying evaluation protocol.

Directions recorded by the scalar or multi-objective definitions do not sort
or select candidates in this phase. How available, invalid, and failed members
participate in reproduction is an explicit Phase-10.12 selection decision.

## Scientific boundary

A high scalar or favorable objective vector is a search signal only. It does
not satisfy the research charter's topology, spectral-protection, boundary,
robustness, reference-matching, finite-size, or leakage gates, and it does not
constitute a scientific discovery claim.
