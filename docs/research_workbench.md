# Autonomous research workbench

Phase 17 adds a persistent research framework under `toposc_lab.research` and an
**Autonomous Research** section in the existing TOPOSC native application. The
application configures and inspects experiments through `ResearchService`;
detached workers execute the same `ResearchEngine` available from the CLI.

The first implementation searches connectivity on a fixed square set of sites.
The default is 100 sites. Its exact adapter is a versioned extension of the
established finite localizer calculations. It preserves the model, basis,
disorder convention, numerical tolerance, and success threshold. It does not
establish a thermodynamic phase, a bulk gap, or separated Majorana modes.

## Start and inspect an experiment

Launch the existing application with `toposc-live`, then open **Autonomous
Research → New Experiment**. The form exposes common settings; the editable
configuration JSON exposes the full serialized protocol, constraints, mutation
weights, surrogate settings, and validator selection.

1. Choose a new output directory, experiment name, seed, and search method.
2. Set the site count through `space.side` (`N = side²`) and the exact budget.
3. Review the physics protocol and disorder widths/seeds in the configuration.
4. Save the configuration or start the run.
5. Inspect **Live Dashboard**, **MAP-Elites**, **Search Progress**, **Candidate
   Explorer**, **Baselines**, and **Checkpoints / Research Log**. Select an
   archive cell to inspect its stored candidate. **Final Report** shows the
   generated report after completion or safe stop.

Pause, resume, checkpoint, and safe stop use durable control requests. The
current exact stage finishes before a running worker pauses or stops. Closing
the application leaves its detached research worker running. Reopening an
experiment reconnects to persisted state. Duplicate an experiment to change its
configuration; the original run retains its recorded configuration and lineage.

The CLI uses the same service and engine:

```powershell
.venv/Scripts/python.exe -B -m toposc_lab.research create results/research/example --config config.json
.venv/Scripts/python.exe -B -m toposc_lab.research run results/research/example
.venv/Scripts/python.exe -B -m toposc_lab.research inspect results/research/example
.venv/Scripts/python.exe -B -m toposc_lab.research pause results/research/example
.venv/Scripts/python.exe -B -m toposc_lab.research resume results/research/example
.venv/Scripts/python.exe -B -m toposc_lab.research checkpoint results/research/example
.venv/Scripts/python.exe -B -m toposc_lab.research stop results/research/example
```

`run` is foreground execution; `resume` launches a detached worker. Use a
separate terminal to request controls during a foreground run. `run --max-cycles
1` bounds one invocation without replacing the experiment's total cycle budget.
An archived or terminal experiment is retained for inspection; duplicate it to
start a new experiment.

## Configuration and exact budgets

`ExperimentConfig` expands physical and surrogate defaults when it loads or
saves JSON. The top-level `objective` is the single objective setting. It is
injected into the selected physical protocol; a conflicting
`physics.objective` is rejected. Unknown plugins and changed scientific
definitions fail validation before a run is created.

Selected defaults:

| Setting | Default |
| --- | --- |
| Algorithm | `surrogate_map_elites` |
| Sites | `space.side = 10`, giving 100 sites |
| Exact attempt budget | 280 |
| Candidate proposal budget | 10,000 |
| Search cycles | 10 |
| Proposal pool / selected batch | 40 / 4 |
| Archive descriptors | coordination variance, clustering coefficient |
| Archive shape | 12 × 12 |
| Exploitation / uncertainty / novelty | 60% / 20% / 20% |
| Checkpoint frequency | 250 exact attempts |
| Surrogate | graph-descriptor bootstrap ensemble, 8 members |
| Exact workers per experiment | 1 |
| BLAS threads | 1 |
| Disorder widths | 0.0, 0.2, 0.4 |
| Disorder seeds at every width | 17001, 17002, 17003, 17004 |
| Clean confirmation | enabled |

Every clean, independent confirmation, or single-seed disorder stage is one
expensive exact attempt. A stage includes its BdG spectrum and three localizer
calculations. Default complete validation costs `2 + 3 × 4 = 14` attempts per
candidate, including each baseline. Cheap proposals and surrogate predictions
do not consume this budget. An interrupted attempt remains charged; retrying
it consumes another attempt. Compare methods at equal exact attempt budgets,
including baselines, failures, and retries.

The geometry defaults require a connected graph, degree between 2 and 6, bonds
no longer than √2, no geometric crossings, and the exact regular-grid bond
count. `bond_tolerance` is a fraction of that count. Nonzero tolerance enables
approximately fixed budgets. Standalone bond addition/removal will be rejected
when it violates the selected bond count; compound swaps and rewiring remain
available. Duplicates and nearby graph neighborhoods are filtered before exact
selection. Symmetry-equivalent geometry identities are used for search
diversity, not for reusing exact physics results.

To generate a small configuration for a bounded engineering smoke run:

```python
import json
from pathlib import Path
from toposc_lab.research.config import ExperimentConfig

config = ExperimentConfig(
    name="Small workbench smoke",
    cycles=1,
    exact_budget=32,
    candidate_budget=200,
    pool_size=12,
    batch_size=4,
    checkpoint_every=4,
    physics={"disorder_widths": [0.2], "disorder_seeds": [17001, 17002]},
)
config.space["side"] = 4
config.validate_plugins()
Path("config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
```

This smoke configuration retains the exact scientific thresholds. A smoke run
tests the system; it is not a sufficient algorithm benchmark or hypothesis test.
The full 100-site scientific experiment is a separately launched run.

## Scientific evidence contract

The adapter ID is `phase17.fixed-sites-chiral-p-wave.v1`.

- The existing `ChiralPWaveModel` uses hopping 1, chemical potential 2, pairing
  1, chirality +1, and component-major Nambu order. Energy units are hopping.
- Sites occupy the unit square grid anchored at `(0, 0)`, with fixed exterior
  boundary labels and open boundaries. The probe is the bounding-box center:
  `(2.5, 2.5)` for 36 sites and `(4.5, 4.5)` for 100 sites.
- Exact diagonalization uses `numpy.linalg.eigh`. Localizers use κ values
  `(0.1, 0.2, 0.3)` and tolerance `1e-10`. Eligibility requires invertible
  localizers, a consistent nonzero local index, and particle-hole residuals
  within tolerance. Quality is the minimum localizer gap when eligible, zero
  otherwise. The frozen success criterion is quality ≥ 0.20.
- Disorder reuses the established scalar uniform onsite transform, sampled in
  `[-width/2, width/2]`. Particle and hole offsets have opposite signs. Identical
  width/seed protocols across candidates give matched site disorder draws.
- Every exact payload retains the spectrum, Hamiltonian identity, localizer
  values, residuals, actual onsite offsets, boundary weights, and low-energy
  Majorana diagnostics. It also embeds a validated existing `DatasetRecord`.
  Launch provenance includes source archive hashes, Git state, package/runtime
  versions, experiment ID, protocol, seeds, and tolerances.

The default objective, `robustness_success_fraction`, is the equally weighted
mean of empirical success fractions at the declared disorder widths. Each width
retains its own seed outcomes, successes, sample counts, quality statistics, and
95% Wilson interval through the existing robustness metric implementation.
Widths reuse seeds, so these groups are correlated; the adapter does not claim
an independent pooled confidence interval. `robustness_quality_mean` and
`clean_quality` are explicit alternative objectives. All raw metrics remain
available alongside the selected objective and the surrogate acquisition data.

No `W_c` value is invented. A finite list of successful disorder samples does
not locate a critical disorder strength. `minimum_abs_energy` is a finite full
spectrum gap; a separately established bulk gap is unavailable. Majorana
polarization, self-conjugacy, boundary weight, and splitting remain diagnostic
quantities and never produce `MAJORANA_VALIDATED` in this adapter. A fixed-site
wiring has no declared size-extension map, so finite-size validation is
explicitly unavailable even when requested.

Validation is stored by stage and state:

| State or field | Meaning in the reference adapter |
| --- | --- |
| `EXACT_EVALUATED` | Exact finite spectrum available |
| `TOPOLOGY_VALIDATED` | Eligible finite center-probe localizer evidence |
| `ROBUSTNESS_VALIDATED` | Clean and every requested sampled disorder realization pass the frozen success criterion, with requested confirmation passed |
| Majorana status | `diagnostics_only`; no certificate |
| Finite-size status | `unavailable`; no candidate-preserving extension |
| `FAILED` | Exact stage or requested reproducibility check failed |
| `score = null` | Full requested exact evidence or confirmation unavailable/failed |

`ROBUSTNESS_VALIDATED` is scoped to those sampled finite-system tests. It does
not certify unseen disorder seeds or widths, Majorana robustness, or a phase.
Missing stages do not receive a success fraction. Completed execution failures
count as unsuccessful requested ensemble members; their identities remain
visible. Incomplete/failed candidates cannot obtain a complete objective score
or enter the exact elite archive.

## Architecture and persistence

| Module | Responsibility |
| --- | --- |
| `config.py` | JSON configuration, default expansion, plugin validation |
| `space.py` | Fixed sites, graph constraints, mutations, identity/distance |
| `descriptors.py` | Registered structural measurements |
| `strategies.py` | Random, evolutionary, MAP-Elites and surrogate MAP-Elites search |
| `surrogate.py` | Exact-label bootstrap models, OOD measures, saved predictions and calibration |
| `physics.py` | Versioned exact protocol, stage plan, objectives and evidence validators |
| `engine.py` | Budgeted loop, stage commits, controls, recovery, archive and model updates |
| `storage.py` | SQLite journal, checksummed JSON records, atomic exports |
| `provenance.py` | Source archive and compatible-resume checks |
| `reporting.py` | Diagnostics, checkpoint reports and final report |
| `service.py` | Shared UI/CLI operations, detached launch, consistent snapshots |
| `toposc_live/research_page.py` | Existing native application's workbench views |

Each experiment has a persistent `EXP-...` identity and its own directory:

```text
experiment/
  research.sqlite3
  config.json
  state.json
  manifest.json
  source.zip
  candidates/
  checkpoints/
  models/
  plots/
  reports/
  logs/
  final_report.md
```

SQLite is authoritative. Its typed object records hold configuration, state,
candidates/geometry/descriptors/lineage, predictions, exact stages, validation,
archive/search state, models, checkpoints, errors, and reports. Separate tables
record attempts, chronological events, and pending controls. Transactions use
WAL and full synchronization. The engine holds one writer lease; UI snapshots
use one read transaction to avoid mixing different commits. Atomic readable
files are exports rather than the only copy of results.

Before an exact call, the engine commits an attempt. After it returns, the
result and completion status are committed together. Pending candidates and
search random-generator state are persisted. On compatible recovery, already
committed stages are reused without another exact call; an uncommitted stage
uses its original seed for a bounded retry. The interrupted attempt remains in
the budget ledger. Timing for a crashed process can be a lower bound.

Search/model state is reloaded only after acquiring the writer lease. Archive
parent ordering is deterministic across canonical JSON serialization. Pause/stop
state changes and consumption of the corresponding control requests commit
together. A `FINALIZING` state makes report publication resumable. Missing or
corrupt checkpoint exports are rebuilt from checked database records; database
payload corruption itself stops the run. On a completed run, **Checkpoint now**
can repair missing report/state exports without running any physics.

Resume requires matching source and runtime, intact archived source, and the
stored configuration. Recover from the archived source when the working tree
has changed, or duplicate the configuration for a new experiment. Editing a
running experiment's scientific settings is not a supported parameter-update
path. Every new run logs its complete configuration; `ResearchService.duplicate`
additionally records the source experiment and explicit configuration changes.
Wall-clock limits and manual controls are cooperative between exact stages;
they cannot interrupt a dense eigensolver midway through a matrix operation.

The initial worker count is one per experiment for deterministic stage
accounting and bounded native-library concurrency. Independent experiments may
run concurrently. Larger matrices require explicit time/memory planning;
changing the site count does not establish finite-size convergence.

## Extending the workbench

Extensions are Python registrations loaded before configuration validation and
worker construction. They belong in version-controlled source, so worker
startup imports the same definitions and source snapshots cover their code.
There is no runtime execution of arbitrary code from configuration JSON.

### Search strategies, mutations, and descriptors

Implement the shared `SearchStrategy` surface: `initialize`, `propose`,
`observe`, `checkpoint`, `resume`, and `summarize`. Register it with
`register_strategy(name, strategy_type)`. Preserve deterministic checkpointed
random state, exact-result eligibility, and the engine's separate exact budget.
The UI obtains algorithm names from the strategy registry.
Set `uses_surrogate = True` on a strategy that requests the common surrogate
training/screening lifecycle. Keep plugin imports in a shared startup module
(for example the research package initializer), so both the UI and a fresh
detached worker load the same version-controlled registrations.

Use `register_mutation(name, function)` for an operation receiving the space,
edge set, and seeded generator. Return a proposed edge set; the common space
constraint/duplicate validation still applies. Configure nonnegative operation
weights in `search.mutation_rates`.

For example, a dimensionless boundary coordination descriptor can reuse the
fixed boundary labels:

```python
from toposc_lab.research.descriptors import register_descriptor

register_descriptor(
    "mean_boundary_coordination",
    lambda geometry: sum(geometry.degree(i) for i in geometry.boundary_sites)
    / len(geometry.boundary_sites),
)
```

Select its name in `search.behavior_descriptors`, and provide corresponding
finite `descriptor_bounds` and positive `archive_bins`. A new geometry space
registers a compatible constructor in `SPACE_REGISTRY`; its proposed geometries
must also satisfy its selected exact adapter. A registry entry alone is not
scientific validation of a new geometry family.

### Objectives and validators

`register_objective(ObjectiveDefinition(...))` supplies a named description and
a function of the exact summary. An objective is evaluated only when the
requested exact plan is complete and the requested confirmation passes.
`register_validator(ValidatorDefinition(...))` supplies an explicit evidence
gate, its candidate state, and a structured result. Keep raw quantities,
thresholds, scope, and absence reasons inspectable. Registrations reject
accidental replacement of existing definitions.

A new validator that needs expensive calculations also needs declared exact
stages. It must not hide simulations inside a scoring or summary function.
Majorana or finite-size certification requires an appropriate scientifically
declared protocol, not merely a validator returning `passed=True`.

### Physical models and validation protocols

Register `PhysicsAdapterDefinition` using `register_physics_adapter`. Its
factories construct a serializable dataclass protocol and an evaluator. The
protocol exposes `adapter_id`; the evaluator implements:

```python
plan() -> list[dict]                    # stable keys, explicit seeds and parameters
evaluate(geometry, stage) -> dict       # exactly one budgeted expensive stage
summarize(results_by_stage_key) -> dict # pure evidence aggregation, no simulation
```

The engine resolves these factories through `PHYSICS_REGISTRY` and passes the
captured reproducibility metadata. Choose a new versioned adapter ID whenever
the Hamiltonian, basis, probe, physical units, tolerance, success definition, or
validation contract changes. Benchmark new adapters against exact references
and add dataset/protocol/recovery tests. Existing adapters remain frozen.

### Surrogates

Register compatible surrogate constructors in `SURROGATE_REGISTRY`. Implement
fit, prediction, observation, checkpoint, resume, and summary behavior. Train
only on complete exact labels; preserve the distinction between predicted and
exact scores. Persist model identity, training IDs/fingerprint, uncertainty,
and out-of-distribution assessment.

The reference surrogate reuses the established bootstrap gradient-boosting
stack over graph descriptors. The existing minimal GNN did not pass its
adoption gate and is not promoted to the default. Calibration compares saved
predictions with later unseen exact labels. Bootstrap uncertainty and empirical
two-sigma coverage are diagnostics, not a claim of a calibrated predictive
interval. Too few labels means training remains unavailable until sufficient
exact evidence exists.

## Verification and interpretation

The targeted adapter/configuration and scientific regression suite passed 156
tests on 2026-09-15. It covers the existing chiral-p-wave model, localizers,
Majorana diagnostics, onsite disorder, robustness metrics/uncertainty/reports,
legacy discovery validation, and the workbench adapter/configuration:

- The 36-site reference reproduces the established clean spectrum, quality,
  topology indices, and onsite disorder spectrum within `1e-10`.
- The 100-site regular reference has center indices `[1, 1, 1]` and clean
  quality `0.3759853813347358`, passing the finite quality threshold. This is a
  clean finite reference check, not a disorder or phase certificate.
- Partial/failed ensembles cannot create a complete score. Confirmation
  disagreement blocks scoring. Matched seeds yield identical site offsets
  across connectivity changes.
- Exact dataset records validate and retain the launch manifest's source/Git,
  environment, experiment identity, protocol, and chosen objective.
- Ruff and isolated strict Mypy checks pass for the physical adapter and
  configuration modules.

End-to-end run, UI, interruption/recovery, and full-suite gate results are
recorded in the Phase-17 implementation report and execution state. Reports
distinguish observed data, validated finite evidence, suggestive relationships,
and inconclusive hypotheses. A small engineering smoke run cannot establish
that an algorithm or connectivity family is scientifically superior.
