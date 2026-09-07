# Phase 10.16 — Diversity preservation

## Status and scope

Accepted as the Phase-10.16 architecture for preventing a successful
generation loop from silently collapsing beyond explicit search-family
occupancy limits.

This phase adds no inferred geometry taxonomy, graph or physical distance,
fitness sharing, crowding distance, diversity or novelty score, automatic
parent pairing, repair, replacement, retry, archive, checkpoint, resume
behavior, machine-learning behavior, or dataset functionality.

## Research decision

Population collapse is a known finite-population evolutionary-search failure
mode. Classical niching work compares crowding and fitness sharing and shows
that their behavior depends on a chosen genotypic or phenotypic sharing
definition ([Deb and Goldberg, 1989](https://www.cse.unr.edu/~sushil/class/gas/papers/NichingSpeciationDeb.pdf)).
NSGA-II preserves spread in multi-objective search through an objective-space
crowding-distance rule ([Deb et al., 2002](https://doi.org/10.1109/4235.996017)).

Neither distance is representation-neutral for Toposc-Lab. A graph-edit
distance, coordinate distance, Weisfeiler-Leman fingerprint, objective-space
distance, or weighted combination would encode a new scientific judgment
about what makes two physical geometries similar. Selecting such a metric in
Phase 10.16 would also anticipate the explicit Phase-10.17 novelty decision.

The version-1 mechanism therefore uses no distance and changes no fitness.
It enforces only caller-declared categorical family occupancy constraints.
This is a deliberately transparent hard gate rather than a claim that hard
family caps are universally optimal evolutionary-search behavior.

## Explicit search-family contract

`GeometrySearchFamily` is a technical `(family_key, family_version)` label.
It is distinct from the Phase-8.13 `GeometryFamilySpec`, whose purpose is to
bind cross-size construction provenance to a finite-size scaling result.

A caller supplies one versioned classifier identifier and a
`DiversityFamilyClassifier` that maps each complete `GeometryGenome` to one
`GeometrySearchFamily`. The classifier is called once for every member in
population order. Its definition must be frozen with the experiment protocol;
the library does not inspect outcomes and then invent or revise categories.

The classifier may use an externally justified construction rule, motif
taxonomy, or other predeclared domain contract. The library does not silently
fall back to:

- `metadata["family"]` or generation metadata;
- the generator key;
- exact geometry snapshot identity;
- the Phase-6 canonical graph fingerprint;
- site count, edge count, coordinates, or embedding dimension;
- fitness, topology, robustness, visual appearance, or a learned cluster.

The Phase-7 exact geometry ID is used only to reject inconsistent labels when
the same exact snapshot occurs more than once in one population or across a
generation history. It does not determine the label, prove physical
equivalence, or merge distinct snapshots.

## Occupancy policy

`DiversityPreservationPolicy` requires at least one positive absolute bound:

- `minimum_distinct_families`; and/or
- `maximum_members_per_family`.

Absolute counts are sufficient because Phase 10.15 preserves one fixed
population size. They avoid a hidden rounding convention for fractional
capacities. Numeric choices remain experiment-level decisions and have no
library default.

`assess_population_diversity(...)` returns a complete immutable
`PopulationDiversityReport` containing the exact population, policy,
classifier identifier, ordered member-to-family ledger, exact snapshot IDs,
stable first-occurrence family counts, and all policy violations. It defines
no entropy, pairwise distance, diversity score, novelty score, ranking, or
fitness adjustment.

## Generation-loop integration

`GenerationLoopConfig.diversity` is optional so the Phase-10.15 contract
remains backward compatible. When configured, the caller must also supply the
classifier and its identifier to `run_generation_loop(...)`.

The initial population is classified and checked before its first fitness
evaluation. Every target population is classified after structural mutation
validity and fixed-size composition but before target fitness evaluation. A
successful `GenerationLoopResult` retains the initial report and one exact
target report per transition. Version 2 of the generation-loop record denotes
this optional extension; its two-raw-word PCG64 seed schedule is unchanged.

Each `GenerationReproductionRequest` exposes the valid source diversity
report. An experiment-specific producer can therefore account for current
family occupancy while creating its exact required proposal count. The core
loop still does not choose a family, reweight tournaments, pair parents,
resample operators, or request surplus candidates.

Phase-10.13 complete elite tiers remain intact and occupy the first target
slots. Diversity does not split a scalar tie or Pareto front and does not
override fitness ordering. The producer must fill remaining capacity so that
the final population meets the frozen family policy.

## Failure and audit boundary

A violating initial or target population raises `DiversityPreservationError`
with the complete `PopulationDiversityReport`. The loop performs no filtering,
repair, replacement, retry, or partial evaluation. An invalid initial report
stops before fitness and reproduction. An invalid target report stops before
target fitness, so an inadmissible family composition cannot influence later
selection.

Ordinary classifier exceptions become `DiversityClassificationError` records
bound to the exact classifier identifier, generation, member index, exception
type, and message. `BaseException` subclasses are not swallowed. Wrong return
types and inconsistent repeated-snapshot labels are API-contract errors.

The result is an in-memory audit ledger, not persistence or resume state.
Checkpointing and resume remain Phase 10.18 and Phase 10.19.

## Scientific boundary

Family occupancy is an engineering search constraint. A declared family label
does not prove common physical origin, equivalence, novelty, literature
priority, topology, protection, boundary character, robustness, resource
fairness, or a causal geometric mechanism. It does not satisfy any scientific
evidence gate in the pre-Phase-9 research charter and cannot by itself support
a discovery claim.
