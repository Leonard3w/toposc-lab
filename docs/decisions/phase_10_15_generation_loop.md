# Phase 10.15 — Generation loop

## Status and scope

Accepted as the Phase-10.15 architecture for composing, validating, and
evaluating fixed-size ordered geometry populations across explicit generation
transitions.

This phase adds no automatic parent pairing, mutation or crossover
probability, repair, retry, diversity preservation, novelty score, archive,
checkpoint, resume behavior, machine-learning behavior, or dataset
functionality.

## Population and evaluation contract

The existing `InitialPopulation` remains the generation-zero contract.
`GenerationPopulation` and `GenerationPopulationMember` represent generations
one and later without pretending that another initial population is a later
generation. Every member stores its generation index, member index, exact
genome, validation-source genome, and successful Phase-10.9 validity report.
The population constructor recomputes that report under the one retained
policy.

`evaluate_population_fitness(...)` now accepts either population kind while
preserving its Phase-10.11 behavior. The evaluator therefore sees the actual
generation and member indices. Every generation receives exactly one ordered,
failure-aware fitness ledger under the same exact fitness-definition object.
Ordinary evaluation and fitness-construction failures remain in that ledger;
they are not converted to artificial scores.

The population size, embedding dimension, and exact validity-policy object are
constant across a successful loop. Generation and member indices are
consecutive. Population size is not silently reduced when an evaluation fails.

## Transition order

For each transition, `run_generation_loop(...)` performs these steps in order:

1. identify complete elite tiers through Phase 10.13;
2. reserve the first target slots for all members in those tiers;
3. if capacity remains, run Phase-10.12 tournament selection;
4. call the supplied offspring producer exactly once for the remaining count;
5. validate every proposal independently through Phase 10.9;
6. compose the exact fixed-size target population; and
7. evaluate that population through Phase 10.11.

Elite genomes retain exact object identity and their established tier/source
order. Their transition source is the same retained genome. Complete-tier
elitism can fill the entire capacity; in that case selection and offspring
production are not run merely to consume randomness or create unused records.

## Neutral offspring boundary

Phase 10.15 cannot infer a universal reproduction policy from the available
primitives. The mutation APIs require explicit caller-chosen edits, and the
Phase-10.14 crossover is deliberately restricted to compatible fixed
scaffolds. Consequently, an `OffspringProducer` receives one immutable
`GenerationReproductionRequest` containing:

- the exact source fitness ledger and tournament-selection result;
- target generation and required offspring count;
- the retained validity policy; and
- one explicit reproduction seed.

It must return exactly that many ordered `OffspringProposal` records. Each
proposal identifies its exact selected-parent slots, the selected parent used
as its Phase-10.9 validation source, an operator identifier, and any explicit
operator seed. The producer may compose already defined mutation or compatible
crossover operations, but the loop does not choose parents, pair them, assign
operator probabilities, reinterpret an incompatible crossover, or inspect
fitness to alter a candidate.

The producer identifier passed to the loop names and versions the external
composition policy. The in-memory Phase-10.15 ledger is not a persistence or
resume format; those contracts remain Phase 10.18 and 10.19.

## Invalid proposals and failures

The loop recomputes proposal validity instead of trusting the producer. If one
or more proposals are invalid, `InvalidGenerationOffspringError` retains the
complete ordered proposed batch, every report, the exact reproduction request,
the selection ledger, and its seeds. No proposal is filtered, repaired,
replaced, or retried, and no partial target population is returned.

Wrong producer return types, wrong counts, out-of-range parent slots, malformed
provenance, and altered result ledgers are API-contract errors. Ordinary target
evaluation failures continue to use the existing Phase-10.11 member-failure
records.

## Reproducibility and audit ledger

The loop uses one local `numpy.random.PCG64` stream initialized from the
explicit nonnegative root seed. It draws exactly two raw words per requested
transition in generation order: first the tournament-selection seed and then
the reproduction seed. Both words are drawn even when complete elites fill a
generation, so the versioned root-seed schedule is independent of a fitness
tie at one transition.

`GenerationTransition` retains the exact source fitness, elitism result,
optional selection result, derived seeds, producer identifier, offspring
records, target population, and target fitness. `GenerationLoopResult` retains
generation zero and the complete transition chain. Constructors cross-check
object identity, population capacity, ordering, validity, fitness-definition
continuity, config continuity, and the root-seed schedule.

## Diversity and scientific boundary

Duplicate genomes and repeated selected parents remain in exact order. No
deduplication, crowding, fitness sharing, family balancing, or novelty pressure
is introduced; those are explicit Phase-10.16 and Phase-10.17 decisions.

A generated, elite, selected, or fit member is an engineering search object.
It does not satisfy the research charter's topology, spectral-protection,
boundary, robustness, reference-matching, finite-size, leakage, or causal
mechanism gates and is not evidence for a scientific discovery claim.
