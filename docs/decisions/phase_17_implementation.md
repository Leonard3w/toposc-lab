# Phase 17 — Autonomous Research Workbench

Date: 2026-09-16. **Engineering completion gate PASS** within the scientific
adapter scope documented below. No large scientific experiment launched.

## Scope and architecture

The existing native TOPOSC Live and Streamlit interfaces now include **Autonomous
Research**. Both use the same experiment service, detached worker and transactional
SQLite journal. Existing discovery, physics, topology and disorder implementations
remain unchanged. The new exact adapter reuses their validated conventions in a
separate versioned fixed-site protocol.

Implemented:

- Configurable, uniquely identified experiments; JSON save/load/duplicate; frozen
  physics and compute settings; source archives and package/runtime provenance.
- Random, evolutionary, MAP-Elites and surrogate-assisted MAP-Elites through a
  common registered search interface. Behavior descriptors and mutation weights
  are configurable; duplicate and near-duplicate rejection precede exact selection.
- Seven graph mutation operators, fixed sites, connectedness, degree/locality,
  crossing and bond-budget constraints. Default: 100 sites, 180 bonds, open edges.
- Graph descriptors including coordination, triangles/4-cycles, clustering,
  boundary/bulk degrees, paths, Laplacian spectrum, bond lengths and anisotropy.
- Existing bootstrap gradient-boosting infrastructure; exact-only training,
  multiple available finite-observable targets, OOD signals, uncertainty and
  prequential calibration. Default selection allocation: 60/20/20.
- Individual clean/confirmation/disorder attempts, raw data, existing DatasetRecord
  validation, finite topology and boundary diagnostics, per-width robustness
  fractions and Wilson intervals. Failed or incomplete physics cannot become elites.
- Regular and random-rewired matched controls. All physics stages, including
  controls, failures and interrupted attempts, consume the same exact budget.
- Live metrics, controls, clickable archive, progress plots, candidate comparison,
  baselines, checkpoints/log and structured final reports inside the existing UI.
- Bounded cycles, proposals, attempts and wall time; cooperative pause/stop;
  atomic checkpoint exports; safe restart; source/runtime/config guards.
- Extensible registries for strategies, geometry, descriptors, mutations,
  surrogates, objectives, validators and physical adapters; documented examples.

See [the workbench guide](../research_workbench.md) and
[the default configuration](../../examples/research_experiment001.json).

## Reliability decisions and corrections

SQLite is authoritative, with FULL synchronization, short read transactions,
payload checksums and an OS-released single-writer lease. Existing Windows atomic
replacement/backoff helpers publish readable exports. Every attempt is committed
before numerical work; completed results and attempt completion commit together.

The recovery tests exposed and fixed canonical-JSON archive ordering changing
parent selection. Search uses a stable cell order. An engine reloads search/model
state after acquiring its lease, so an object created before another worker's
progress cannot overwrite that progress. Control consumption and pause/stop state
commit together. FINALIZING permits interrupted report publication to resume;
completed exports can be repaired without further physics. Corrupt database
payloads stop the run instead of silently substituting older scientific evidence.

One exact worker per experiment is supported initially. BLAS thread configuration
is applied before import in detached/CLI workers and execution settings are recorded
and checked at resume. Python API callers should set thread environment variables
before importing NumPy. Controls take effect between numerical stages or bounded
proposal pools; a dense eigensolver cannot be paused midway through a call.

## Verification

- Focused backend/search/physics/UI gate: **103 passed** before final dashboard
  presentation changes. Final full-regression result is recorded below.
- Scientific adapter/configuration and legacy reference checks: **156 passed**.
  The 36-site spectrum/quality and onsite-disorder calculations agree with the
  established adapter within `1e-10`; the 100-site clean reference is exercised.
- Ruff: PASS for the new subsystem, UI modules, native integration, tests and drivers.
- Strict isolated Mypy: backend 13 modules PASS (`--follow-imports silent`);
  UI 2 modules PASS (`--follow-imports skip`), Python 3.14. A combined UI check with
  imported Qt/application types reports eight cast/inference diagnostics; the
  isolated check is the recorded scope, not a claim of repository-wide type cleanliness.
- Native screenshots reviewed at normal window size: configuration, dashboard,
  archive, progress, candidate, baselines, checkpoint and report. Native UI tests
  and Streamlit AppTests use real persisted exact results as well as controlled fixtures.

### Final end-to-end smoke

`scripts/phase_17_smoke.py --output results/phase17-smoke-final`: **PASS**.

The affordable run uses 16 sites, 3 search cycles, two matched controls, four
selected candidates per cycle, one disorder width (0.2), two matched disorder
seeds and independent clean confirmation. It requests pause after a completed
cycle, hard-exits a fresh process after a numerical calculation, detects the
interruption, resumes through the UI's detached launch API, and compares with an
uninterrupted exact reference.

| Check | Result |
| --- | --- |
| Complete candidates | 14 per run: 2 controls + 12 search candidates |
| Exact attempts | 57 resumed; 56 uninterrupted |
| Hard crash | exit 73, incomplete attempt remains charged |
| Search/results after recovery | identical candidate IDs, connectivity, acquisitions, raw metrics and states |
| Completed replay | zero additional exact calls |
| Surrogate retraining | 2 model versions, 8 prequential labels |
| Active allocation | 5 exploitation, 2 uncertainty, 1 novelty selections |
| Archive | 6 / 144 cells |
| Checkpoints | 13 in resumed run |
| Proposals | 84, with invalid/duplicate records retained |
| Best robustness score | 0; no fully robustness-validated candidate |

Zero prediction error here reflects a constant zero target, not evidence that
the surrogate improves scientific search. The initial hypothesis remains
**inconclusive**. Total non-test physics attempts across this final smoke pair:
113. The earlier smoke pair also used 113 and is retained separately; no 100-site
research campaign was launched.

Read-only `scripts/phase_17_review.py` audit: **PASS** — 56 exact-stage dataset
records, 14 candidate summaries, all 6 archive cells and 13 checkpoint envelopes
verified, with zero added calculations. Final source SHA-256:
`2d9b2f48dd870f86c8911db9050b26130a5c0f9a40a7e33d47f3ded5ba7f610c`.

Artifacts: `results/phase17-smoke-final/`, `results/phase17-artifact-review.json`,
`results/phase17-ui-review/`, `results/phase17-*.log`. Results are git-ignored and
must be preserved separately. The final report is
`results/phase17-smoke-final/resumed/final_report.md`.

Direct inspection: [smoke final report](../../results/phase17-smoke-final/resumed/final_report.md),
[artifact audit](../../results/phase17-artifact-review.json),
[native dashboard](../../results/phase17-ui-review/real-dashboard-default-size.png),
[candidate explorer](../../results/phase17-ui-review/real-candidate.png).

## Scientific limits and recommended next settings

The objective is the existing finite-quality success fraction averaged over
declared widths, not an invented critical disorder strength. Model parameters,
pairing/basis, tolerances, kappas and the Q >= 0.20 success criterion remain fixed.
Center-probe topology and boundary Majorana diagnostics do not certify a
thermodynamic phase or chiral Majorana edge modes. A bulk-only gap, true W_c,
candidate-preserving finite-size validation and matched amorphous/quasiperiodic/
fractal controls remain unavailable. Their absence is explicit, not a passed gate.

Initial geometry uses square site counts (side squared), not arbitrary N.
Long-run memory/storage scaling and multi-worker exact concurrency are not
scientifically or operationally benchmarked at large budgets. The first full
study must predeclare multiple search seeds, matched exact budgets, disorder
coverage and independent held-out validation before making algorithm comparisons.

For the first inspected 100-site pilot: keep the serialized default fixed model,
180-edge budget, degree 2–6, maximum bond length sqrt(2), crossing prohibition,
single worker/BLAS thread, 60/20/20 allocation and four disorder seeds. The default
280-attempt limit includes controls and validation: 14 attempts per fully
evaluated candidate. This is a bounded pilot, not evidence sufficient to answer
Experiment 001. Launch separately after reviewing the implementation and smoke.

`TOPOSC_LAB_MASTER_PLAN_V3.md` was absent from the repository search. Architecture
decisions therefore used current code, `docs/roadmap/EXECUTION_STATE.md`, recorded
Phase-16 decisions and the user's supplied Phase-17 specification.

## Full-regression gate

**PASS: 3,077 tests in 904.23 seconds**, 2026-09-16. No skipped/disabled failing
tests. Full log: `results/phase17-full-tests.log`.

Command (all four BLAS thread environment variables set to 1):

```powershell
.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider --basetemp=results/pytest-phase17-full
```

Completion checks passed: configuration, all four strategies, configurable
descriptors/mutations, duplicates/near-duplicates, active/uncertainty/novelty
selection, exact validation, persistence/checkpoints/crash recovery/pause-resume,
integrated UI/dashboard/archive/explorer, matched baselines, checkpoint/final
reports, provenance, automated regression and successful end-to-end smoke.

This closes the Phase-17 **workbench implementation**, not the scientific
Experiment-001 hypothesis. Stop here; review the saved artifacts before separately
authorizing and launching the larger study.
