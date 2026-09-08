# Phase 10.18 — Checkpointing

## Status and scope

Phase 10.18 adds versioned, data-only persistence of completed generation
prefixes. A checkpoint can be loaded for inspection and validation. Continuing
the search is Phase 10.19; benchmarking remains Phase 10.20.

## Architecture review

The Phase-6 geometry archive already preserves oriented edges, arbitrary
embedding dimension, abstract graphs, boundaries, faces, construction metadata,
and exact snapshot identity. Phase-9 candidate storage retains candidate outcomes
but does not represent the Phase-10 population/selection/elite reference graph.
In particular, GenerationLoopResult constructors require shared *object identity*
for populations, genomes, policies, fitness definitions, and selected members.
Independent serialization of each record would break those invariants.

The checkpoint therefore stores the complete existing result graph through an
explicit object table. Each registered dataclass appears once by object identity,
with references to previously encoded dependencies. Equal but separately
constructed genomes are not merged. GeometryGenome nodes reuse the Phase-6
geometry archive and reconstruct through the Phase-10.1 conversion API.

The closed registry in `_checkpoint_codec.py` lists only existing result records
and enums needed by the loop, evaluation, diversity, and novelty contracts. A file
cannot choose import paths, constructors outside that registry, or callbacks.
Unsupported values in user metadata fail explicitly before destination writes.
This is an internal checkpoint codec, not a general object serializer or dataset
schema. Changes to its accepted record schema require an explicit checkpoint
version decision.

## Minimal checkpoint contract

`SearchCheckpoint` contains:

- the complete `GenerationLoopResult` prefix, starting with generation zero;
- the originally requested total transition count, separate from completed count;
- an explicit, versioned evaluator identifier and caller-declared code version;
- recorded Python, NumPy, SciPy, and package versions;
- optional existing Novelty reports in unique completed-generation order, referring
  to the exact populations in the retained result.

The loop result already retains the root seed, PCG64 identifier and version,
two derived seeds per transition, operator seeds, producer identifier, validity
policy, selection/elitism configuration, complete numerical evaluations,
reproducibility records, fitness and failure ledgers, and optional diversity
reports and classifier identifier. These are saved without rerunning numerical
evaluation, classifiers, distance callbacks, reproduction, or selection callbacks.
Constructor validation does replay the existing deterministic selection and seed
checks using local random generators; it does not alter the search stream.

The completed count and root seed locate the next pair in the established
two-raw-word PCG64 schedule. No additional random stream or new seed policy is
introduced. There is no hidden mutable loop RNG state at a completed boundary.

Callback identifiers refer to external experiment policies. The checkpoint does
not capture closures, solver instances, private callback state, or arbitrary
external random generators. Exact future continuation requires deterministic
callbacks under their recorded inputs and a compatible environment. Loading alone
does not assert runtime compatibility or prove that a caller supplied the correct
policy implementation. That execution gate belongs to Phase 10.19.

## Completed-generation callback

`run_generation_loop(..., checkpoint_callback=...)` optionally calls an observer
after generation-zero fitness and after every completed target fitness ledger.
Its argument is a validated `GenerationLoopResult` prefix. The prefix config's
generation count is the number of completed transitions; the original selection,
elitism, and diversity config objects remain shared. The final returned result
still uses the caller's original config. Its record version and seed schedule are
unchanged because the observer adds no result fields or search decisions.

The observer runs before another transition begins. Its exceptions propagate
immediately, with no retry. A failure during a later evaluation or reproduction
leaves the last successfully saved boundary available. There is no snapshot of a
partially constructed or partially evaluated generation. A callback must treat
the supplied result as read-only; some existing nested scientific metadata
containers retain their preexisting mutability.

Example integration with already defined experiment callbacks:

```python
def checkpoint_completed(prefix):
    snapshot = create_search_checkpoint(
        prefix,
        requested_generation_count=config.generation_count,
        evaluator_identifier="experiment.evaluator.v1",
        code_version=declared_code_version,
    )
    save_search_checkpoint(checkpoint_path, snapshot)

result = run_generation_loop(
    initial_population,
    definition=definition,
    evaluator=evaluator,
    config=config,
    seed=root_seed,
    offspring_producer=producer,
    producer_identifier="experiment.producer.v1",
    checkpoint_callback=checkpoint_completed,
)
snapshot = load_search_checkpoint(checkpoint_path)
```

An experiment using diversity must additionally supply its already required
classifier arguments. Novelty remains separately evaluated and is attached only
when the caller explicitly supplies existing reports. Reference cohorts are saved
as report inputs, not grown into a novelty archive.

## Format, validation, and atomic writes

Version 1 is a ZIP containing exactly `manifest.json` and `payload.json`, with a
format tag, integer schema version, and SHA-256 digest over payload bytes. The
payload contains the dependency-ordered object table and root reference. Arrays
retain dtype, shape, and raw bytes encoded as base64; object arrays are rejected.
Typed mapping keys, tuples, bytes, complex values, and supported NumPy scalars are
retained. No pickle is used. Fixed ZIP timestamps make repeated writes of the
same supported snapshot byte-identical in the same runtime; cross-runtime
compression identity is not promised.

Loading checks size limits before decompression, exact container members,
duplicate JSON keys, checksum, supported record/enum names, backward reference
bounds, field sets, genome schema, and reconstructed derived fields and versions.
Array byte lengths are checked against shape before reshaping. Nested geometry
archive expansion has an aggregate bound. Constructors revalidate existing
population, fitness, seed, selection, elitism, diversity, and novelty invariants.
No archive members are extracted to filesystem paths. The checksum detects
accidental corruption, not authenticity against a party able to rewrite the file
and recompute its digest.

Saving first serializes and reconstructs the snapshot in memory to validate it,
then writes a temporary sibling, flushes and fsyncs it, and atomically replaces
the explicit destination using `os.replace`. Failure before replacement leaves
the previous checkpoint intact and removes the temporary file. The destination
directory must already exist. Filesystem power-loss guarantees beyond file fsync
and atomic replacement are platform-dependent.

Version 1 writes the complete prefix each time, including spectra and states.
Storage and validation costs therefore grow with history; incremental or
distributed checkpointing is not introduced here. Existing scientific failure
records remain distinct and are never converted to fitness or discarded.

## Scientific boundary

Persistence preserves evidence and provenance; it does not add scientific
validity. The pre-Phase-9 charter's independent validation seeds, fair resource
comparisons, physical applicability, and discovery-claim gates remain binding.
