# Phase 10.20 — Benchmark vs random search

## Architecture review and scope

Phase 10.19 is the accepted starting point. Geometry genomes, validity checks,
population fitness, selection, elitism, and the generation loop already provide
the needed execution and audit contracts. The Phase-9 baseline statistics accept
ranked candidate batches, not the complete generation histories used here.
Converting histories into those batches would obscure repeated evaluations and
generation ordering. This step therefore adds a small paired driver and immutable
comparison views over the existing fitness ledgers; it does not add another
evolution engine, physical evaluator, optimizer, dataset, or persistence schema.

Random search is a meaningful adaptive-search baseline: see
[Bergstra and Bengio, JMLR 2012](https://jmlr.org/papers/v13/bergstra12a.html).
That paper concerns hyperparameter optimization, not evidence of an advantage in
topological-superconductor geometry search. The project's scientific boundaries
remain those in [the research charter](pre_phase_9_research_charter.md).

## Frozen public contract

`SearchBenchmarkProtocol` retains the versioned protocol and candidate-space
identifiers, sampler/evaluator/producer identifiers, code version, population
size N, generation-loop configuration G, distinct ordered trial seeds, fitness
definition, validity policy, and explicit success criterion. It optionally retains
the evolution diversity classifier identifier. It is constructed before execution.

`run_search_benchmark(protocol, sampler=..., evaluator=...,
offspring_producer=..., family_classifier=...)` executes every declared trial:

1. Draw N initial genomes through `sampler(seed)`. Both arms receive this exact
   initial population, but evaluate it separately. No initial fitness is reused.
2. Run the existing evolution loop, unchanged, for G transitions.
3. Evaluate the random arm's initial population, then G independently sampled
   populations of N members. Random draws receive only their own seed, no search
   outcomes. No selection, elites, or diversity-occupancy quotas are applied to
   the random arm; those are evolution strategy choices, not candidate validity.
4. Retain complete histories, strict success decisions, and the evolution result.
   Report best-so-far and hit progress at every individual evaluation attempt.

Every supplied sampler must produce a policy-valid candidate by construction from
the declared distribution. The common validity policy and embedding dimension are
enforced. More specific candidate-space/resource controls (for example, a fixed
backbone or boundary policy) must be enforced in the caller's sampler and producer.
An identifier is a declaration, not a proof of arbitrary callback behavior.

The evaluator receives `(genome, paired_seed)` and is the same callable for both
arms. Available evaluation provenance must match that seed and the declared code
version. Model name and parameters, solver name and settings, and evaluation
settings must be identical across both arms and all trials. This v1 benchmark is
for a fixed physical evaluation, not joint model-parameter optimization. A physical
study must additionally freeze applicable topology hooks, resource strata,
disorder protocols, and independent validation/confirmation roles.

## Budget and randomness

Each completed arm has exactly B = N × (G + 1) evaluator attempts. This counts
generation zero, every elite reevaluation, duplicate snapshots, invalid numerical
evaluations, callback exceptions, and fitness-construction failures. Nothing is
deduplicated or assigned an artificial zero score. The budget is NOT a count of
unique geometries, guaranteed successful eigensolves, wall-clock seconds, or FLOPs.
Equal attempt budgets alone cannot establish equal compute cost for unequal sizes,
different solver behavior, or evaluator-internal ensembles. Generation overhead
also differs: the common initial genomes are sampled only once.

For each trial root seed, version 1 consumes local PCG64 raw words in this order:
one evolution root word, B sampling words, and B paired evaluation words. The first
N sampling words generate the shared start; the remainder generate random-only
candidates. Both arms use the same evaluation word at absolute attempt index
`generation_index * N + member_index`. Evolution internally keeps its existing
two-words-per-transition schedule. No global NumPy RNG state is consumed. Seeds
and versioned code determine the schedule; callback-private RNG/counter state does
not. Callbacks must be deterministic from explicit inputs, with no hidden history,
adaptive restarts, benchmark feedback, or held-out seed access.

## Interpretation and failures

The explicit criterion is evaluated once per available member, after both searches,
and must return a strict Python bool. It may use retained raw scalar or vector
quantities. Unavailable members are unsuccessful without invoking the criterion.
Every `BenchmarkArmResult` provides attempt count, available count, cumulative hit
count, first-hit attempt (one-based), and best scalar score. None means unavailable
or no hit, never zero. Failure categories and raw evaluations remain in the original
fitness ledgers. Minimize/maximize direction is respected. Multi-objective histories
have no automatic scalar, hypervolume reference point, or scalar quality difference.

`scalar_quality_difference` is final best evolution minus best random for maximize,
reversed for minimize; positive favors evolution. It is None if either arm lacks a
scalar. `paired_success_differences` retains one -1/0/+1 hit difference per run.
Both-success and neither-success are distinguishable from the underlying ledgers.
Independent runs, NOT the correlated candidates within them, are replication units.
No automatic significance test, confidence interval, seed pooling, superiority
label, or physical discovery claim is made. A publication needs predeclared
run-level uncertainty/decision rules and adequate independent replication.

Ordinary evaluator failures are retained by the existing fitness API. Sampler,
offspring, selection, diversity, criterion, and provenance contract violations
propagate: no retries, partial-success benchmark, or fabricated replacement run.
Invalid random draws retain the rejected genome, complete validity report, trial
seed, zero-based absolute attempt index, and sampling seed in
`InvalidBenchmarkSampleError`, including invalid shared-start candidates.
An all-unavailable generation can prevent the existing loop from continuing. Such
an aborted campaign is not a complete equal-budget comparison; retain the failure
and fix/version the protocol before a new campaign. Do not discard unfavorable
seeds. Benchmark-suite checkpoint/resume and archive formats are not introduced;
the returned individual evolution results remain compatible with existing search
checkpoint APIs.

## Demonstration and completion boundary

`examples/phase_10_search_demo.py` fixes six abstract sites, a connected ring plus
two uniformly sampled optional chords (36 possibilities), eight equal-strength
edges, a real normal-state hopping Hamiltonian, and a normalized finite-spectrum
gap objective. Evolution replaces one chord, retaining the backbone and resources.
The 0.65 demo hit threshold, five seeds, N=6, G=4, and operator policy are explicit
engineering choices, not library defaults or the class-D research protocol.

It uses real numerical evaluations but is NOT a superconducting/BdG/topological
experiment. It writes no files and never accesses the deleted Phase-9.8 outputs.
Do not retune its threshold/seeds after observing its output to manufacture a win.
The five small pairs illustrate machinery, not statistical significance.

Completion of Phase 10 means tested evolutionary-search and benchmarking
infrastructure, including an executable engineering comparison. It does not mean
the scientific research charter's geometry-advantage hypothesis has been proved.
Phase 11 and later remain separate, unimplemented by this change.

## Verification

Verified on Python 3.14.7 with `PYTHONDONTWRITEBYTECODE=1` and explicit local
pytest temporary directories: 56 focused benchmark/demo tests, 402 Phase-10
regression tests, and 2,522 full-suite tests passed. Focused Ruff passed; the
repository-wide baseline remains 147 findings (101 fixable), unchanged. Relevant
strict mypy with `--python-version 3.14` passed for 17 source/example files.
`git diff --check` passed. Protected worktree artifacts were not modified, and
neither `geometry_demo.npz` nor any deleted Phase-9.8 results were regenerated.
