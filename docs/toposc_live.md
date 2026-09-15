# TOPOSC LIVE native research frontend

`toposc-live` is a separate PySide6/Qt desktop application with a Campaign
Launcher, Live Campaign monitor, and Candidate Leaderboard / Best Candidate
page. The existing Streamlit `toposc-ui` entry point and source are unchanged.
At the original tooling checkpoint, Phase 16B and Discovery Campaign 1A had
not started. A subsequent user-run Campaign 1A exposed Windows file contention;
see the [storage fix and seed-17101 recovery instructions](decisions/windows_discovery_storage_contention.md).
The read-only **Campaign Group** view now summarizes compatible independent
seeds together; see [group usage and definitions](campaign_groups.md) and the
[Campaign 1A aggregate](decisions/campaign_1a_group_summary.md).

## Install and start on Windows

From the repository root, using the existing environment:

```powershell
uv pip install --python .venv/Scripts/python.exe -e ".[live]"
.venv/Scripts/toposc-live.exe --root results
```

For environments with pip, `python -m pip install -e ".[live]"` is equivalent.
PySide6 and psutil are optional `live` dependencies; core/headless installations
do not acquire them. The editable installation supplies the entry point.

```powershell
.venv/Scripts/python.exe -B -m toposc_live --root results
.venv/Scripts/toposc-live.exe --root results --open results/phase15-gate-final/random
```

Repeat `--root` to scan additional locations. `--state-dir PATH` overrides the
application metadata location, normally `%LOCALAPPDATA%/toposc-live` on Windows
and `~/.local/state/toposc-live` elsewhere. No browser or localhost service is
used. `--help` works without Qt. Future executable packaging should retain an
explicit Python scientific-engine environment: the current launcher uses
`sys.executable`; a bundled GUI executable is not a Python CLI interpreter.
An installer or standalone frozen executable is not supplied in v1.

## Architecture and files

```text
Qt pages ← CampaignSnapshot / CandidateSnapshot ← read-only CampaignReader
                                                     ↑
                                    checksummed persistent engine journals
                                                     ↑
ConfigurationAdapter → ProcessService → detached supervisor → official discovery CLI
      DiscoveryConfig                                              ↓
                                                         scientific engine
```

All new application source lives under `src/toposc_live/`, outside
`src/toposc_lab/`. Scientific implementation files are unchanged.

| Module | Responsibility |
|---|---|
| `__init__.py`, `__main__.py` | Lightweight package and optional desktop entry point |
| `configuration.py` | Official DiscoveryConfig mapping, validation, review text, resume preflight |
| `processes.py` | Detached launch, application metadata, PID identity checks, bounded log tail |
| `worker.py` | GUI-independent supervisor, official CLI execution and persisted exit status |
| `reader.py` | Campaign discovery, version-aware read-only adapter and artifact cache |
| `models.py` | Qt-independent snapshots, missing-value and conservative scientific labels |
| `refresh.py` | Timer-driven background I/O and queued snapshot delivery |
| `widgets.py` | Native geometry and exact-quality plots |
| `pages.py` | Launcher, live monitor and sortable/filterable candidate table |
| `window.py` | Navigation, recent paths, manual open and lifecycle actions |
| `groups.py` | Compatible group definitions, descriptive aggregation, JSON/CSV exports |
| `group_page.py` | Group creation, per-seed summary, progress curves and group leaderboard |

Other changes: `pyproject.toml` adds the entry point and optional dependency
group; `README.md` links this guide; `docs/roadmap/EXECUTION_STATE.md` records the
tooling addition without changing scientific gates. Four focused test modules
are `tests/test_live_{configuration,processes,reader,gui}.py`.

The GUI never calls an exact evaluator, eigensolver, generator, trainer,
scoring function, topology classifier, robustness calculation or Majorana
classifier. Configuration uses the official `DiscoveryConfig` constructor.
Geometry loading uses the public data-only geometry archive decoder.
No duplicate scientific configuration schema or scientific parameters are
introduced. The small launch request format stores process orchestration
metadata, not a second scientific contract.

## Campaign Launcher

1. Choose a name and a **new** output location. Existing directories are rejected
   for new launches. Open an existing campaign to review Resume instead.
2. Set the generator identifier, first seed, number of independent seeds,
   cycles, exact candidates per cycle, and pool size.
3. Optionally select **Run one cycle, then checkpoint and exit**.
4. Choose **Validate and review configuration**. Inspect the full official
   configuration, scope, output paths, seed list, threshold and exact budget.
5. Choose **Start Campaign**. Monitoring becomes available immediately; the
   manifest and numerical records appear as the engine persists them.

Changing any input invalidates the review. Validation and launch both preserve
the official configuration. Unsupported generators or combinations fail closed
with the engine's explanation. The v1 engine accepts `auto`, `patch`, `random`,
`evolution`, and `coverage`; only `phase14.planar-wiring.v1` is supported for
launch. The GUI's core reader and display logic accept arbitrary stored
generator names. Model, thresholds, disorder defaults, tolerances and basis
fields are displayed for review and are not editable in this v1 form.

The exact attempt cap comes directly from `DiscoveryConfig.exact_attempt_cap`:
cycles × batch size × (2 + disorder samples) + retry reserve. There is no
independent budget override in the current engine. The launcher displays both
per-seed and aggregate caps, including unused reserve. One seed uses the chosen
directory; multiple seeds use `seed-N` subdirectories with one independent
process each. Multi-seed launch is not transactional: a later launch failure
reports which earlier processes were launched; those continue independently.
No multi-seed aggregate scientific comparison is invented.

The launcher is for the existing bounded discovery CLI, not a dedicated
Campaign-1A or Phase-16B protocol. It does not automatically launch either.

## Processes, closure, reconnect and lifecycle

`ProcessService` writes `jobs/<uuid>/request.json`, the exact `config.json`, and
application process state under the application state directory. It starts a
detached supervisor with an argument list, no shell, redirected standard
streams, and closed inherited handles. On Windows it uses `DETACHED_PROCESS |
CREATE_NEW_PROCESS_GROUP`; on POSIX it starts a new session. No terminal window
is requested. The supervisor launches:

```text
<python> -u -m toposc_lab.discovery <campaign-directory> --config <config.json>
```

The cycle-boundary option appends `--iterations 1`. Engine stdout and stderr
go to `engine.log`; supervisor errors go to `supervisor.log`. The original
runtime environment is inherited without changing numerical thread settings.
The engine remains responsible for provenance, source ZIPs, checksums, its
single-writer lease, checkpoints, all computations and recovery.

Closing the window stops only application refresh and waits for any short
application I/O operation. It has no terminate/kill/pause operation and does not
wait for the campaign lifetime. The supervisor and discovery process continue.
Reopening the application discovers jobs and reads persisted state. Process
identity uses both PID and creation time, preventing PID-reuse confusion.
Unknown liveness stays unknown; file modification age is not proof of Running,
Interrupted or Failed. External CLI campaigns can be monitored but their
process lifetime is unavailable unless launched through this application.

Supported controls are **Start**, **Open / Monitor**, and **Review Resume**.
Safe Stop and Pause are visibly unavailable because the engine has no safe
mid-run request interface. A one-cycle invocation can exit at an engine-owned
checkpoint and later resume; it is not a pause button. No hard termination is
exposed. A successful partial invocation is displayed as Interrupted, with
stored cycle count and exit code 0 available in health details.

Resume preflight validates the stored config through `DiscoveryConfig`, its
fingerprint, source fingerprint, source ZIP checksum, Python/NumPy/SciPy/platform
and thread environment, checkpoint compatibility and external exclusion list.
The official engine performs the final authoritative inventory/recovery checks.
The current CLI cannot reconstruct an API driver's external exclusion dataset,
so such campaigns are rejected for GUI resume and remain inspectable.

The engine hashes **all Python source under `src/`**, including this frontend.
Consequently, archived Phase-15 campaigns require their original frozen source
and runtime to resume. Installing this frontend does not bypass or update their
fingerprints. Likewise, updating frontend source during a new campaign can
make subsequent resume incompatible. This existing provenance contract remains
unchanged; fully independent binary distribution would need an explicit future
engine-interface/versioning change.

## Discovery and read-only refresh

The app scans configured roots to depth three for `manifest.json` or
`leaderboard.json`, adds manually opened/recent paths, and reconnects to recorded
application jobs. Symlink directories are not traversed. The sidebar lists
stored status for discovered campaigns; double-click/click opens a campaign.
Discovery repeats every 30 seconds, with a manual rescan button.

A two-second Qt timer requests background I/O with at most one outstanding
refresh. Widgets consume a completed snapshot on the GUI thread. The approach
uses Qt's documented [timer](https://doc.qt.io/qtforpython-6/PySide6/QtCore/QTimer.html)
and [thread/signal](https://doc.qt.io/qtforpython-6/PySide6/QtCore/QThread.html)
interfaces. Failed refreshes retain the last displayed snapshot and mark it
stale in the status bar. Switching campaigns clears the previous display while
the new snapshot loads.

The reader uses manifest, checkpoint, attempt ledger, cycle plans, base and
integrated candidate journals. It can show exact base outcomes before
confirmation/robustness finish; unrecorded confirmation stays unavailable.
Historical report-only campaigns can use `leaderboard.json` and candidate
reports. Unknown schema versions fail closed. Optional missing, incomplete or
malformed records produce unavailable fields or warnings instead of scientific
inferences. No dataset is rewritten or reconciled by this reader.

Each artifact envelope is checksum-verified. A bounded 1,024-entry cache keyed
by nanosecond modification time and size avoids reopening unchanged files;
changed/incomplete files are retried. Files exceeding 32 MiB are reported as
unavailable. Very large campaigns still incur directory scans and snapshot
assembly; remote/HPC, tail-indexed journals and database adapters are future
work. Refresh is tolerant of atomic concurrent writes but is not a transactional
snapshot across every file. Full committed-cycle inventory validation remains
the engine's responsibility on resume; the UI states this integrity scope.

On Windows, even a short-lived ordinary read handle can block `os.replace`.
Discovery JSON and application metadata writers now retry only appropriate
Windows access/sharing/lock failures with eight total attempts and at most
1.13 seconds of backoff. Publication remains same-directory atomic replacement;
persistent failures propagate. The reader closes its file before JSON decoding.
No custom Windows opener or change to scientific/recovery/provenance policy is
introduced. Full error policy and production recovery limits are in the
[Windows storage report](decisions/windows_discovery_storage_contention.md).

Quality curves copy exact quality and show its running maximum against the
successful exact-evaluation ledger count across all stages (clean,
confirmation, disorder), not cycle count or predicted labels. Curves need that
ledger; report-only campaigns without it show unavailable. Generator grouping
uses stored plan/report names and supplies only descriptive evidence actually
present in the selected campaign. No cross-campaign benchmark is synthesized.

Candidate selection updates geometry and diagnostics. Sorting is numeric for
quality, rank, seed and cycle; filtering covers table fields. OOD flags are
compact in the table with full evidence in details/tooltips. Geometry uses
stored coordinates and endpoints, with stored boundary sites highlighted;
embeddings above two dimensions are explicitly labeled as first-two-coordinate
projections. Geometry identity checks precede archive rendering.

`diagnostics_only` and `majorana_claim=false` are preserved explicitly. Missing
energy, robustness, OOD, topology or confirmation data are unavailable. Neither
zero energy nor a nonzero localizer index implies a Majorana claim in the GUI.
Progress separates successful exact evaluations from charged attempts and
remaining attempt budget; failed/unfinalized attempts and repeated-stage retries
appear in health. Events are summaries of persisted ledger/plan/candidate/commit
records, not an invented timestamped event bus. Log tail is supplementary.

## Extension boundaries

New supported generators use the existing generator identifier adapter and
official engine validation; no new widget branch is needed to display their
stored names. New models/protocols or scientific configuration choices require
an officially supported engine adapter first. New campaign schema adapters
should normalize into the snapshot contracts in `models.py`, preserving unknown
fields as unavailable. New tabs can subscribe to snapshots alongside the three
existing pages in `window.py`; they should not traverse raw scientific files.

Remote execution, richer comparisons, parameter-space studies, 3D interaction,
new scientific models, Phase-16B and later workflows are intentionally not
implemented. The process service, reader, configuration adapter and display
contracts provide separate replacement points for those future additions.

## Validation

Focused tests cover configuration mapping and unchanged scientific parameters,
unsupported values, resume mismatch guards, discovery, old/future schemas,
missing/corrupt/torn files, repeated cached refresh, read-only file hashes,
exact-only leaderboard parsing, geometry, generic names, conservative labels,
live states, PID identity, detached lifetime, CLI argument construction and
offscreen Qt navigation/selection/refresh/closure.

The production launch/reconnect path is also exercised with `--iterations 0`
in a temporary test directory: the official CLI creates its initial manifest
and checkpoint, exits normally, and records **zero exact attempts**. A harmless
sleeping Python process tests survival after its launcher exits. No real
discovery campaign is needed for these tests.

Focused validation: **54 passed**. Ruff and isolated strict Mypy pass for all
11 new source files (Mypy uses `--python-version 3.14 --follow-imports silent
--ignore-missing-imports`, consistent with the installed NumPy stubs).
Native Windows captures of all three pages were inspected; offscreen Qt tests
exercise rendering but the Windows offscreen platform lacks normal system font
rendering, so visual inspection used the native Windows platform.

Full repository regression: **2896 passed in 841.80 seconds**, 2026-09-14.
The Streamlit AppTest startup smoke completed with **exit code 0 and no
exceptions**. The Phase-16A preservation check matched **all 946 inventory
file hashes**, with the attempt count unchanged at **464**.

Results are recorded in the tooling section of
`docs/roadmap/EXECUTION_STATE.md` and the git-ignored
`results/toposc-live-full-tests.log`, `results/toposc-live-streamlit-smoke.log`,
and `results/toposc-live-freeze-check.json`. Native visual captures are
`results/toposc-live-{launcher,monitor,candidate}-native.png`.
No scientific implementation, Phase-16A artifact, threshold, tolerance,
generator or scoring behavior changed. Discovery Campaign 1A and Phase 16B
were not started during that original tooling task. The subsequent storage-only
fix and its validation are recorded separately in the linked Windows report.
