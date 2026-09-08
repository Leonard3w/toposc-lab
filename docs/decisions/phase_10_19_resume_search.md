# Phase 10.19 — Resume search

## Status and architecture

Phase 10.19 continues a completed-generation checkpoint to its already recorded
target. The accepted Phase-10.18 checkpoint contains the complete prefix, root
seed, derived seed history, configurations, external policy identifiers, runtime
versions, and evaluation/failure/diversity/novelty records. No new storage schema
or persistent record type is required. Phase 10.20 benchmarking is not included.

Fresh and resumed searches must use the same transition implementation. The
existing transition body now resides in `_continue_generation_loop`; the fresh
entry point first constructs and evaluates generation zero, while the resume entry
point uses the retained final fitness and diversity reports. Both entry points
then invoke that shared body. GenerationLoopResult version 2 and checkpoint
schema version 1 remain unchanged because their fields, seed schedule, and
search decisions are unchanged.

## Minimal public contract

`resume_search(checkpoint, ...)` accepts a `SearchCheckpoint` plus explicitly
rebound evaluator and offspring producer callbacks, their identifiers, and the
declared code version. If the stored config enables diversity, it also requires
the family classifier and its identifier. A checkpoint is loaded separately with
the existing `load_search_checkpoint` API; resume performs no implicit file I/O.

Resume always targets `checkpoint.requested_generation_count`, interpreted as
the total number of transitions from generation zero. It does not accept a new
target, seed, validity policy, fitness definition, tournament config, elite
config, or diversity policy. Those objects come from the stored experiment.

The return value is another `SearchCheckpoint`, containing the full original
history plus the newly completed transitions. The original prefix's objects
retain identity in memory; existing fitness and failure records are not retried
or replaced. The requested target and provenance remain unchanged. A checkpoint
already at its target returns unchanged after validation, with no search or
checkpoint callbacks.

## Compatibility gate

All gates run before any user callback:

- required callbacks must be callable;
- evaluator, producer, classifier, and code identifiers must equal their stored
  values after the existing nonempty-label normalization;
- the complete recorded runtime inventory must equal the current Python,
  NumPy, SciPy, and package version inventory;
- the full checkpoint graph must pass the existing Phase-10.18 codec and
  constructor validation, including versions, identity relationships, seeds,
  evaluation correspondence, fitness, and diversity/novelty invariants.

`IncompatibleSearchCheckpointError.mismatches` retains the detected identifier
and runtime mismatches. Unsupported or malformed stored state uses the existing
`SearchCheckpointError`. There is no permissive override that silently continues
an altered experiment. The runtime inventory helper is shared with checkpoint
creation, so recording and checking use the same fields and missing-package
convention.

Identifiers are caller declarations, not proofs of callback implementation
identity. The explicit code version covers the experiment's external policies;
resume does not discover Git state or serialize executable code. Determinism
requires callbacks whose complete state is determined by their explicit inputs.
Closures with private counters or RNGs are not restored. Matching package version
strings also does not identify BLAS builds, thread settings, hardware, or every
floating-point behavior; bitwise equivalence across machines is not promised.

The validation roundtrip reconstructs a temporary graph and discards it, retaining
the supplied prefix objects for execution. Existing constructors recompute
deterministic consistency checks with local random generators. They do not rerun
the numerical evaluator, producer, classifier, or novelty distance callback.

## Seed, history, and failure continuity

For a stored prefix of `g` transitions and a target of `T`, the shared engine
derives the existing `T` pairs from the root PCG64 seed and consumes only pairs
`g` through `T - 1`. It uses the original two-raw-word schedule, including seed
pairs for fully elite generations. Generation indices remain absolute.

The source fitness is the stored final fitness ledger. Its failed members remain
failed and retain their indices; selection still uses the established available
fitness policy. Every new target population is evaluated exactly once in member
order. Reproduction validation, complete elite tiers, and invalid-offspring
errors retain their existing definitions and no retry is introduced.

The complete stored diversity history is supplied to the same consistency checks
for new generations. A repeated exact snapshot cannot change family merely
because execution crossed a checkpoint boundary. Stored populations are not
reclassified. Existing Novelty reports and reference cohorts are preserved with
their population identities; resume does not invent a distance callback or
automatically compute or insert new novelty scores.

## Further checkpoints and repeated interruption

The optional `ResumeCheckpointCallback` receives a full `SearchCheckpoint` after
each newly completed generation. Unlike the lower-level fresh-loop observer,
its payload already carries the original target, external provenance, and stored
Novelty reports. The starting boundary is not emitted again.

Callback errors propagate before another generation starts. The caller may use
the atomic Phase-10.18 writer to retain the latest completed boundary. Repeated
save/load/resume cycles preserve the same target and seed positions.

```python
checkpoint = load_search_checkpoint(checkpoint_path)

def save_completed(snapshot):
    save_search_checkpoint(checkpoint_path, snapshot)

completed = resume_search(
    checkpoint,
    evaluator=evaluator,
    evaluator_identifier="experiment.evaluator.v1",
    offspring_producer=producer,
    producer_identifier="experiment.producer.v1",
    code_version=declared_code_version,
    checkpoint_callback=save_completed,
)
result = completed.result
```

Experiments with a stored diversity policy additionally provide its classifier
and matching identifier. Neither this function nor the example grows a novelty
archive or changes the experiment protocol.

## Verification and scientific boundary

Tests compare saved complete checkpoints from uninterrupted and interrupted,
disk-loaded, resumed runs byte-for-byte in the same test runtime. They cover
generation-zero and later interruptions, repeated interruptions, real seeded
edge mutations, scalar and multi-objective fitness, full elitism, existing
failures, diversity-history consistency, Novelty preservation, compatibility
rejection, and callback execution boundaries.

Continuation preserves a declared search protocol; it does not satisfy additional
scientific evidence gates. Validation seeds, resource fairness, model applicability,
and the pre-Phase-9 charter remain binding. A scientific efficiency comparison
against random search remains Phase 10.20.
