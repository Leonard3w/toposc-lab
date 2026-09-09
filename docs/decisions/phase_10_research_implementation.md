# Implementation of TOPOSC-P10-EVO-RS-001

## Scope and immutable scientific inputs

The separately committed protocol is
`71e159f9eeb67de17b551cb820c4f3599830ca3b`, at
[pre_phase_10_research_protocol_v1.md](pre_phase_10_research_protocol_v1.md).
This implementation makes that post-Phase-10 research experiment executable.
It does not change the frozen protocol or introduce Phase-11 dataset machinery.
The user starts preflight and full execution explicitly, after committing the
implementation. No full research experiment was run during implementation.

`phase_10_research.py` binds the existing Phase-9.8 model, complete topology
grids, gates and scientific summary to the paired search. Its generator seeds
go directly through the registry. It reconstructs the generator's existing
Delaunay pool through the new `hard_core_planar_edge_pool` helper. This public
helper delegates to the original algorithm; both generator outputs retain their
version-1 behavior.

The edge-swap neighborhood is enumerated without evaluating physics. The
parent's complete physical contract and Delaunay membership are checked first.
With coordinates, boundary, site count, edge count and pool fixed, only degree
and connectedness can change. Checking those properties for every proposed
swap is equivalent to rechecking the unchanged spatial constraints each time.
The selected child receives the full physical validation again, plus the
existing generation-loop validity checks. No legal neighbor means a recorded
no-op, with an evaluation slot consumed as required by the protocol.

## Scientific values and backward compatibility

`DerivedEvaluationRun` extends the existing pipeline record with an immutable
mapping of explicitly derived scientific quantities and a derivation identifier.
It retains the original simulation, scientific evaluation, validity and
reproducibility objects. It does not place physical fitness in geometry metadata
or overwrite the finite spectral gap or geometry descriptors.

`LexicographicFitnessDefinition` declares ordered quantity names, directions and
the required derivation identifier. `LexicographicFitness` retains finite raw
values without a weighted scalar or normalization. Selection uses the resulting
lexicographic order and seeded tie resolution. Elitism preserves the complete
best equality tier in source order. Available ledger members are checked against
their retained derived values, including on reconstruction.

Scalar and Pareto definitions retain their existing semantics. The closed
checkpoint codec additionally recognizes these new record types and paired
trial/arm records. The checkpoint envelope and existing record fields remain
unchanged, so existing version-1 scalar/Pareto files remain readable. Older
readers fail explicitly on the unknown new record types. The new lexicographic
definition has its own version field; corrupt values and incompatible versions
are rejected through the normal constructor/codec checks.

The benchmark API is version 2. Its default `evaluation_mode="seeded"` retains
the original PCG64 schedule and integer-seed evaluator contract. The explicit
`"deterministic"` mode draws no evaluation words, supplies `None`, and audits
that provenance contains `None`. Both modes count initial populations, elites,
duplicates and ordinary callback failures. An optional generation-checkpoint
callback delegates to the existing loop; the generic benchmark itself has no
scientific significance-test or campaign-resume policy.

## Persistence, progress and resume

`phase_10_campaign.py` supplies the experiment-specific wrapper. Before writing
results, it verifies Python 3.14, disabled bytecode, thread environment, committed
code, protocol ancestry and unchanged protocol text. Results use a new
`results/phase_10_research...` directory. An OS-owned file lock prevents two
writers and releases automatically on process exit.

`_research_storage.py` reuses the closed, data-only search codec for separately
versioned, checksum-protected campaign artifacts. It does not deserialize pickle
or import classes named by a file. Publication writes and fsyncs a local temporary
file, then creates the destination exclusively. Completed artifacts are never
replaced. Since [storage amendment A1](phase_10_research_storage_amendment_a1.md),
generation checkpoints are also immutable: the campaign publishes one exclusive
`checkpoint_generation_NNNN.zip` per completed generation. The existing writer
has an opt-in `overwrite=False` mode; its default replacement API and version-1
ZIP payload remain compatible. Replay validates both legacy `checkpoint.zip`
and all retained generation files, including their generation/name agreement.

Input records precede physics evaluation; outcomes follow immediately. Each
unit's attempts live in separate `execution_XXXX` directories. Completed panels
and pairs are sealed with file inventories and hashes. Resume verifies them
before reuse. An incomplete pair is repeated from the same root seed; every
previously stored input, operation and outcome must recur identically. Known
failures cannot become successes through selective reevaluation. The two
evaluations of the shared deterministic start are also compared directly.
Replay prefixes and their checkpoints remain available for audit.

Storage and monitor failures use the explicit `ResearchAbort` control-flow
exception, crossing the fitness evaluator's ordinary-exception boundary. They
stop the campaign rather than being mislabeled as an unsuccessful physical
candidate. The CLI converts that control-flow abort to a readable execution
error. Ordinary numerical/evaluator exceptions remain in the fitness ledger.

`_research_runtime.py` reuses the existing process-memory reader and adds one
thread-safe JSONL writer with a ten-second heartbeat. Timing, CPU and RAM are
observations only. ETA updates from actual evaluation progress. Measurements
from the current execution and additional recorded starts due to replay are
reported separately; the event journal retains earlier execution timing.
Generation of shared starts, independent random candidates and references has
separate timing categories. Physics timing stops before outcome publication;
ledger/checkpoint publication is measured as storage. These observed categories
do not claim to partition all wall time (monitoring, report rendering and replay
loading remain in elapsed time). Summaries also expose elite counts per
transition and no-legal-swap counts from the sealed operation ledger, never
counting superseded replay attempts as new offspring.

`_research_report.py` consumes only retained outcomes. It implements the one
frozen exact paired test, Wilson intervals, best-so-far/first-hit data, descriptive
threshold/ranking sensitivity and the predeclared candidate nominations. Static
figures are generated from stored geometry and values. Interrupted report
generation may finish only if already published bytes agree. A completed
campaign is verified and returned without additional physics.

## Test boundary

The focused tests cover ordering, direction, complete ties, derivation errors,
seed provenance, failure-aware budgets, closed-codec/checkpoint round trips,
exact partial resume, corruption and non-overwrite, failure replay, seed
collisions, reference failure and explicit CLI modes. A tiny clearly labelled
nonphysical fixture exercises the complete 32-pair campaign and completed resume;
it does not evaluate research-role seeds with the physical model.

The real physical adapter and legal/no-op operator are exercised only with the
reserved preflight geometry seed `10_799_902`. The exact paired-test formula is
checked against SciPy for all discordant count combinations through 32, plus
the no-discordance case. These are implementation checks, not research findings.

Relevant strict mypy targets the new modules and the changed search/CLI modules
with `--strict --python-version 3.14 --follow-imports=silent`. The explicit
targets are fully checked; unrelated imported legacy modules are not promoted
to new strict-mypy work. Repository-wide Ruff retains the known baseline of
147 findings (101 fixable), including two pre-existing export/import ordering
findings in `search/__init__.py`. No unrelated lint cleanup is included.

Final verification on 2026-09-09 used Python 3.14.7 with bytecode disabled:
the 189-test search regression selection passed; after the final reporting
adjustments, all 16 campaign contract tests and all 2,538 repository tests
passed. Strict mypy passed for the 12 explicit source targets, all new Python
files passed Ruff, and `git diff --check` passed. The CLI help was checked
through the direct Python entry point because this PC's virtual environment
does not contain the optional installed `toposc.exe` launcher. Protected
bytecode hashes stayed unchanged; `geometry_demo.npz` and the research results
directory remained absent. Only temporary, reproducible test artifacts were
removed after verification. No commit, push or actual campaign was performed.

User instructions and commands are in
[phase_10_usage_de.md](../phase_10_usage_de.md#unseren-forschungslauf-starten).
