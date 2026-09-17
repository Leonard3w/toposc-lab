# Phase 17.1 — Expanded Connectivity Long-Search Preparation

Engineering gate: **PASS**, 2026-09-16.
Scientific Experiment 002: **not created or launched**.

## Result and scope

The existing Phase-17 workbench now supports explicitly configured length-2
overpasses and seeded multi-scale rewiring. The user resolved the original
geometry conflict by permitting unconnected crossings at fixed sites only.
Positive-length overlap and crossings between fixed sites remain forbidden.
The initial conflict and decision are recorded in
`phase_17_1_geometry_preflight.md`.

The versioned physics adapter, Hamiltonian couplings, objective definitions,
success threshold, exact-stage ledger, recovery engine and provenance checks
remain unchanged. Earlier experiment configurations retain the original search
space unless they explicitly select the new convention. Archived experiments
are untouched; source-matching resume remains strict.

## Changed modules

| Files | Change |
| --- | --- |
| `src/toposc_lab/research/space.py` | Opt-in overpasses, explicit crossing/overlap validation, cached geometric conflict checks, bounded composition and seeded initialization |
| `src/toposc_lab/research/descriptors.py` | Regular-edge distance, long-bond fraction and descriptive bond-length aliases |
| `src/toposc_lab/research/strategies.py` | Optional fresh-lineage probability and initialization metadata; existing engine lifecycle reused |
| `src/toposc_lab/research/reporting.py` | Stored structural values and best-searched versus baseline comparisons including per-width evidence |
| `src/toposc_live/research_page.py`, `widgets.py` | Visible structural/lineage summaries and opt-in purple long-bond rendering in research views |
| `src/toposc_lab/app/research_page.py` | Matching Streamlit summaries and long-bond rendering |
| `examples/research_experiment002_long_connectivity.json` | Launch-ready, bounded long-search configuration |
| `scripts/phase_17_1_sample.py` | Reproducible geometry-only reachability audit |
| `scripts/phase_17_1_smoke.py`, `phase_17_smoke.py` | Expanded configuration through the existing hard-crash smoke harness |
| `tests/test_research_expanded_connectivity.py`, `test_research_ui.py` | Focused geometry, descriptor, search, physics and UI verification |
| Workbench guide, decision reports, execution state | Configuration, semantics, evidence and launch instructions |

## Geometry and mutations

New config keys:

```json
{
  "space": {
    "max_bond_length": 2.0,
    "site_crossings": "unconnected",
    "initialization_rewires": [5, 10, 20, 30, 40]
  },
  "search": {
    "initialization_probability": 0.2,
    "mutation_rates": {"multi_rewire_local": 0.5, "multi_rewire_explore": 0.5}
  }
}
```

This is an excerpt, not a complete config. The example additionally fixes 100
sites, 180 edges, degrees 2–6, connectedness and `forbid_crossings = true`.
Explicit endpoint pairs alone define couplings. An overpassed site does not
become an endpoint. Legacy pool and crossing semantics remain the default.

Local composition requests 1–4 valid elementary edits; exploration requests
5–20. An atomic edit can remove multiple conflicting edges and restore the
budget before acceptance, allowing length-2 edges without retaining overlapping
unit edges. At most 40 trials per requested edit keep proposal cost bounded.
Failed trials leave the current valid intermediate intact. Partial composition
can return fewer edits, and no-ops use existing duplicate rejection. Net added
and removed edges are persisted. Seeding records its requested scale, and the
final distance is always measured rather than inferred from operation count.

All final proposals still pass the common space validator, duplicate and
near-duplicate filtering. Search RNG state and seeds follow the existing
checkpoint protocol. The engine has no special-case Phase-17.1 execution path.

## Structural descriptors and reachable diversity

- `regular_edge_distance`: symmetric edge difference divided by twice the
  regular edge count; fraction of regular edges replaced at equal budgets.
- `long_bond_fraction`: fraction of edges longer than sqrt(2).
- `mean_bond_length`, `max_bond_length_observed`: aliases of existing measures.
- `bond_length_variance`: existing definition retained.

The exact-label surrogate feature pipeline includes these descriptors.
Predictions remain acquisition aids; failed/incomplete/rejected evidence is
not introduced as a training label. Descriptor persistence is checked against
recomputation from stored geometry.

Geometry-only audit: seed 17102, 20 examples per initialization scale, each
followed by ten exploration compositions: **1,100 valid 100-site geometries**,
zero exact calls (`results/phase17_1_geometry_audit.json`).

| Requested initialization edits | Observed initial regular-edge distance |
| --- | --- |
| 5 | 0.0222–0.0500 |
| 10 | 0.0500–0.0944 |
| 20 | 0.0944–0.1833 |
| 30 | 0.1444–0.2167 |
| 40 | 0.2000–0.2778 |

Across initialization and subsequent mutations, regular-edge distance ranges
from 0.0222 to 0.5667, long-bond fraction from 0.0056 to 0.2111, and mean bond
length from 1.0125 to 1.3492. These dependent examples demonstrate reachability;
they are neither independent scientific samples nor estimates of exact extrema.

Three actual 100-site examples from that audit (sample 0 in each indicated band):

| Initialization band / subsequent compositions | Regular-edge distance | Long-bond fraction | Mean bond length |
| --- | --- | --- | --- |
| 5 / 0 | 0.027778 | 0.011111 | 1.018015 |
| 20 / 0 | 0.127778 | 0.038889 | 1.075708 |
| 40 / 10 | 0.438889 | 0.116667 | 1.250135 |

The 20 × 20 archive uses distance bounds `[0, 1]` and long-bond bounds `[0, 0.5]`.
The latter follows because each long edge occupies two of the grid's 180 unit
axis segments and overlap is forbidden. Thus no more than 90 of 180 edges can
be long. The audit demonstrates useful variation on both axes; the upper bounds
need not be attainable under the full constraints.

## Experiment 002 preparation

- Objective: existing `robustness_quality_mean`, retaining raw success fractions.
- Widths: 0, 0.4, 0.8, 1.2; matched seeds 17001–17004 at each width.
- Independent clean confirmation and matched regular/random-rewired baselines.
- 50 cycles; pool 200; selected batch 6; proposal cap 100,000.
- Acquisition allocation 50% exploitation, 25% uncertainty, 25% novelty.
- 6,000 charged exact attempts; one exact worker; one BLAS thread.
- Explicit cooperative wall-clock cap: 86,400 seconds.

At 18 stages per candidate, 300 searched candidates plus two baselines need
5,436 attempts before retries. The cap leaves 564 attempts of headroom; actual
completion can be lower. No physical parameters outside the requested disorder
protocol change. Distance-dependent hopping/pairing is not implemented and
would require a new physical model/adapter version.

## Verification

- Focused workbench suite: **126 passed in 80.16 s** before the final visual
  highlighting adjustment (`results/phase17_1_focused.log`).
- Follow-up UI suite: **12 passed in 11.67 s**.
- Final complete regression: **3,099 passed in 949.85 s (15:49)**, including the
  final long-bond rendering checks (`results/phase17_1_full_final.log`). The
  earlier partial full-suite invocation was stopped for that visual adjustment;
  the final suite was restarted and completed against the final runtime source.
- Ruff: all modified Python files pass (`results/phase17_1_ruff.log`).
- Strict scoped Mypy: research backend, 13 files, passes with silent imports;
  UI, 3 files, passes with skipped imports. Both use Python-3.14 configuration,
  matching the Phase-17 scoped checks. This is not a repository-wide typing claim.
- Final source paired smoke: **PASS**, 49 total charged attempts; 25 resumed
  (including one interrupted attempt), 24 uninterrupted. Six complete candidates
  in each experiment, four of them searched. Pause, hard exit 73, detached resume,
  identical recovered scientific sequence and zero-call terminal replay pass.
- Read-only final artifact audit: **PASS**, 24 stage datasets, six summaries,
  five archive cells and eleven checkpoints; zero additional exact calls.
- Native Candidate Explorer rendered and visually inspected with actual stored
  long-bond geometry. Purple distinguishes long bonds from adjacent short bonds;
  text explains the unconnected-overpass convention.

Final smoke: `results/phase17_1_smoke_final/smoke_gate.json`.
Audit: `results/phase17_1_artifact_review_final.json`.
UI: `results/phase17_1_ui_review/candidate-explorer-final.png`.
Runtime source SHA-256:
`30e78b58e8e8d6a8a953d144fd83b32421aef53cdd2b4536235088940d6cc488`.
Base Git commit: `f568a6645cf59ba92f9bb7eab47e59942cde9cbf`; the source archive
includes the uncommitted implementation changes.

One exact-evaluated smoke candidate replaces 58.33% of regular edges, contains
20.83% long bonds and has mean bond length 1.363663. All six tiny-system objective
scores are zero. This smoke verifies engineering behavior and does not establish
successful topology or superiority. An earlier pre-highlighting paired smoke
also used 49 attempts and is retained separately; total non-test smoke spending
for this phase is 98 attempts. No 100-site exact search was launched.

## Limitations and separate launch

Finite sampled disorder does not establish `W_c`, a phase, a bulk gap, validated
chiral Majorana modes, finite-size convergence or universal robustness. Reports
use observed finite-system comparisons and do not claim algorithmic superiority.
Crossings represent a declared connectivity model, not a fabrication proposal.
Only one exact worker is supported, with cooperative controls between stages.

After review, launch separately from the repository root:

```powershell
.venv/Scripts/python.exe -B -m toposc_lab.research create results/research/experiment002-long-connectivity --config examples/research_experiment002_long_connectivity.json
.venv/Scripts/python.exe -B -m toposc_lab.research run results/research/experiment002-long-connectivity
```

**The 6,000-attempt Experiment 002 has not been created or launched.**
