# TOPOSC-LAB EXECUTION STATE

current_block: G (bounded study complete; stopped at Gate 16)
current_phase: 16B falsification complete; cross-model numerical validation unavailable
next_task: research decision on structural identifiability / validated second-model adapter; no automatic Block H
last_completed_task: 16.14 independent validation and 16.16 report; 16.15 cross-model adapter unavailable
last_verified_commit: d3180d9fa1c538fd8a084ba3c80a5099a0c23c16 plus archived dirty source in results/phase16b/source.zip
last_full_test_result: 2973 passed in 675.95 s, 2026-09-15; results/phase16b-full-tests.log
last_full_test_command: .venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider --basetemp=results/pytest-phase16b-full (PYTHONDONTWRITEBYTECODE=1; OMP/OPENBLAS/MKL/BLIS_NUM_THREADS=1)
last_block_gate: MECHANISM_GATE NOT_ESTABLISHED; bounded falsification, audits, zero-call resume and full regression complete
previous_block_gate: AUTONOMY_GATE PASS for bounded monitored campaigns in validated finite wiring stratum
verified_source_sha256: d96499c9a8afa2ac020d51bb5345b925857a6ee05e02dbed3d12e91734495b92
protocol_sha256: 29212daa1b1c13407586d79eef3f90e6eb6125ebe36a84f65ab0ea37c7d3f431
campaign_driver_sha256: 0912d888085f7c6c826e11c469e373f3e72e986536e64faaa0220aed0a78687e

scientific_decisions_in_force:
- Exact numerical physics remains ground truth; predictions are never exact labels.
- Phase 13/13R AL null result and Phase-14 scoped patch advantage remain accepted; no retuning.
- Patch default only for the validated 36-site/60-edge planar wiring stratum and frozen finite-localizer quality.
- Random and Evolution remain comparison baselines; Evolution retains its established baseline role outside that stratum.
- Unsupported spaces require an exact adapter and fair benchmark; no presumed patch superiority elsewhere.
- AL/surrogate and coverage optional; no RL or learned generator justified or added.
- Success remains quality >=0.20; no Phase-15/16A cohort, ablation or simplification candidate reaches it.
- No finite-graph thermodynamic phase or separated Majorana claim; unresolved topology remains unresolved.
- Onsite-disorder ensembles are four samples at width 0.2; finite-size/family scaling explicitly unavailable.
- Model, basis, topology, numerical tolerances and scientific definitions unchanged.
- Phase16A 59-edge simplification is an explicit diagnostic extension, not a discovery-stratum change.
- No motif passes the predeclared ablation criterion; five ranked provisional questions are not design rules.
- Seeds 16101-16108 and all Phase16A parents/edits/simplifications are consumed, not independent Phase16B validation.

validation_and_artifacts:
- Block F: 57 features, 34 variable per cohort, 27 identifiable partials; 32 legacy + 64 fresh distinct analysis geometries.
- 176 fresh proposals: 64 accepted + 98 invalid + 14 duplicate; fresh separation >0.15; no label-based selection.
- 79 matched degree-preserving ablation pairs, 137 distinct edited graphs; 8 single-edge simplifications.
- 464/700 non-test attempts, all complete; no failure/interruption; 231 confirmation pairs; 241 distinct study geometries.
- No BH/BY-significant association or Holm-significant motif effect; importance cross-model rank rho=0.208.
- All 8 specific simplifications retain >=90% quality and [1,1,1]; no universal redundancy claim.
- Phase16A final full suite 2842 PASS; subsystem/reference 93 PASS; Ruff PASS; isolated Mypy 3.14 ignores missing SciPy stubs.
- Audit PASS: 946 files, 464 stored eigensystems/diagnostics, 231 confirmations and all seven Phase15 source archives.
- Exact reference reproducible; stored analysis regeneration and complete CLI resume add zero calculations.
- Phase16A report/protocol/JSON/PNG: docs/decisions/phase_16a_*; raw results/source/plans/inventory: results/phase16a/.
- Audit: results/phase16a-audit.json; full/regression/resume logs: results/phase16a-*.log; usage: docs/phase_16a_usage_de.md.
- Reproducible report/audit drivers: scripts/phase_16a_report.py and phase_16a_audit.py.
- Phase-16A graph/features 16 checks; statistics/analysis 17; intervention/exact 14; campaign recovery checks added.
- Protocol v2 corrected six/eight-seed draft inconsistency before any new exact outcomes: 64 fresh geometries, 8 seeds.
- Phase-16A budget <=700 attempts; target/control degree-preserving swaps, independent confirmations, 59-edge diagnostics.
- Earlier Phase15: AUTONOMY_GATE PASS; 1015 non-test attempts across original/final gates; 2789 tests passed then.
- Phase15 recovery reproduces all 12 records; 84 earlier/final records and 1360 checksums verified; prior source unchanged.
- Phase15 best Random quality 0.131304059; one-seed smoke comparison does not overturn Phase14 scoped result.
- Gate docs: docs/decisions/phase_15_autonomy_gate.md and .json; best candidate, protocol and PNG adjacent.
- Journals/datasets/source/reports/audit: results/phase15-gate-final/; earlier gate preserved in results/phase15-gate/.
- Logs: results/phase15-full-tests.log, phase15-regression.log; artifact review: results/phase15-final-review.json.
- Usage: docs/phase_15_usage_de.md; reproducible gate/report drivers: scripts/phase_15_gate.py and phase_15_report.py.

known_issues:
- Small seed count, imperfect matching, correlated features and compound rewiring limit causal interpretation.
- Some motifs lack a matched intervention on one seed; high-quality feasible parents are preferentially selected.
- Phase16A resume requires identical source/protocol/runtime/inputs; results/ is git-ignored and must be preserved separately.
- One fixed geometry/model stratum; no declared size-extension map or thermodynamic/Majorana certification.
- Four-sample disorder and four-label OOD reference are coarse; success threshold remains unreached.
- Resume requires matching source/config/runtime, single writer and frozen external exclusion archive; interrupted timings are lower bounds.
- Windows venv worker termination must include its process tree; gate harness corrected and real recovery verified.
- Existing default Mypy 3.11 target remains incompatible with installed NumPy stubs; isolated 3.14 checks pass.

application_tooling_toposc_live:
- Separate native PySide6 application in src/toposc_live; optional live dependencies and toposc-live entry point.
- Three pages: validated Campaign Launcher, Live Campaign, Candidate Leaderboard / Best Candidate.
- Official DiscoveryConfig and CLI; detached supervisor, persisted logs/process identity, reconnect and compatible resume.
- Closing the GUI does not stop campaign processes. Safe Stop/Pause unavailable; optional one-cycle checkpoint exit uses existing CLI.
- Read-only checksummed campaign snapshots, cached background refresh, old-report inspection, conservative scientific labels.
- Scientific source, Streamlit source, Phase16A artifacts, thresholds and tolerances unchanged.
- Focused tests: 54 passed; Ruff PASS; isolated strict Mypy 3.14 PASS on all 11 new application modules.
- End-to-end official CLI test uses zero cycles and zero exact attempts; harmless worker verifies detached lifetime.
- Native Windows visual inspection completed for all three pages; Streamlit AppTest startup PASS, exit code 0, no exceptions.
- Phase16A preservation check: 946 inventory file hashes match; 464 attempts unchanged.
- Full repository regression: 2896 passed in 841.80 s, 2026-09-14; results/toposc-live-full-tests.log.
- Documentation: docs/toposc_live.md; logs/screenshots/preservation check: results/toposc-live-*.
- Resume still requires matching source/config/runtime/exclusions; existing source hash includes frontend source under src/.
- At that original tooling checkpoint, Campaign 1A and Phase16B had not started; subsequent work is recorded below.

windows_storage_contention_fix:
- Subsequent user-run Campaign 1A encountered WinError 5 finalizing seed 17101 attempt 150; no scientific failure inferred.
- Real Windows held-read-handle fixture reproduces error 5; historical handle owner cannot be identified from traceback alone.
- Minimal filesystem helper retries replace/temporary cleanup on Windows errors 5/32/33, eight attempts, 1.13 s total backoff.
- No copy/unlink-target fallback; checksums, fsync, atomic publication, science, attempt budget and provenance guards unchanged.
- Application metadata writer shares the correction; ordinary monitor readers retain short-lived read-only access.
- Read-only audit: seed 17101 six completed cycles, 151 attempts, 25 dataset records; saved base stage complete, ledger 150 started.
- All 342 JSON envelopes, committed cycle inventories, original source ZIP and archived source fingerprint verified.
- Seeds 17102-17110 already completed 10/10 cycles and 240 attempts each; no process action taken.
- Patched-source resume intentionally rejected; original archived-source recovery instructions in docs/decisions/windows_discovery_storage_contention.md.
- No production campaign launched/resumed/stopped and no production records edited during this fix.
- All 6007 Campaign 1A files unchanged; results/windows-storage-preservation.json.
- Focused regression 98 PASS; new filesystem tests 24 PASS; full suite 2920 PASS in 448.39 s.
- Ruff PASS and isolated strict Mypy 3.14 PASS for all three changed/new runtime modules.

campaign_group_summary_tooling:
- Read-only Campaign Group view added to toposc-live; no engine/scientific computation or campaign records changed.
- Reusable saved definitions, explicit compatibility validation, technical dropout roles, per-seed statistics, combined exact leaderboard, completed-cycle curves, JSON/CSV exports.
- Campaign Seed and Exact Eval Seed are separate columns in group and single-campaign tables.
- User-specified Campaign 1A group: seeds 17102-17111 included (10 complete); seed 17101 excluded as technical_dropout, data preserved.
- 1927 proposals: 800 accepted unique pool candidates, 1096 invalid, 31 duplicate/near-duplicate; 400 exact candidate records, 2400 completed exact evaluations and charged attempts.
- Best quality 0.14005511784303276, campaign seed 17110; 0/400 candidates and 0/10 included seeds reach unchanged Q >= 0.20.
- Mean per-seed best 0.12647862026702528; median 0.1263495112749929; sample SD 0.006330455769536857.
- Seed 17111 uses the stored Windows persistence fix; archive source differences are limited to _file_io.py, discovery/storage.py and toposc_live/processes.py. Source difference is visible; resume provenance is unchanged.
- Native Windows group summary and leaderboard inspected. Independent journal arithmetic agrees; all 6636 campaign files unchanged.
- Focused regression 126 PASS; extended group/GUI checks 37 PASS. Ruff and isolated strict Mypy 3.14 PASS for 13 frontend modules.
- Full suite: 2953 PASS in 448.56 s (results/campaign-group-full.log). Final malformed-definition guard and four new cases: affected group/GUI/reader tests 64 PASS in 4.31 s (results/campaign-group-final-focused.log).
- Usage: docs/campaign_groups.md. Group definition: docs/campaign_groups/campaign_1a.json. Results: docs/decisions/campaign_1a_group_summary.md.
- No campaign launched, resumed or stopped. No Majorana, thermodynamic or saturation claim; no scientific phase advanced.

phase16b_independent_validation:
- Explicit user authorization 2026-09-15; predeclared protocol, no changed Phase16A outcomes or success labels.
- Seeds 16201-16216, 16 outcome-blind parents (8 Random/8 Patch), 95 raw proposals.
- Frozen exclusions: 32912 D4 structures from 1174 files, including existing later discovery pools and invalid proposals.
- Every selected new structure >0.15 from the snapshot and every other independent new parent family.
- H1/H2/H5 zero strict pairs; H4 one pair. Insufficient matched parents means untested, not disproven.
- H3: 15/32 deletions retained; parent-weighted fraction 0.46875, bootstrap CI [0.28125,0.65625], Holm p=0.0949097.
- Neither >0.75 support nor corrected rejection of 0.75; no confirmed motif/mechanism.
- Main-model roles 50, success hits 0, best Q=0.095223378; success threshold remains 0.20.
- 140/700 charged attempts, all complete; 52 independent exact confirmation pairs. Grid mu=2/8 references PASS.
- Every 60-edge role also evaluated at mu=1.9/2.1; parameter sensitivity only. Cross-model validation unavailable.
- No validated second-model wiring adapter or size extension; no new disorder, Majorana or thermodynamic claim.
- Audit PASS: checksums, strict matching, family separation, eigensystem residuals, diagnostics, exact repeats and stored analysis.
- Independent review PASS: 50 mapped roles, both matched swaps, 32 deletions; all 946 Phase16A files preserved.
- Complete driver replay PASS with identical 140 attempts and inventory; zero added numerical calls.
- New focused checks 16 PASS; earlier journal/exact regression included 23 PASS. Ruff and isolated strict Mypy 3.14 PASS.
- Full repository regression 2973 PASS in 675.95 s; results/phase16b-full-tests.log. No physics-convention changes.
- Report/JSON/visually inspected PNG: docs/decisions/phase_16b_validation.*; protocol adjacent; usage docs/phase_16b_usage_de.md.
- Raw data/source ZIP/inventory: results/phase16b/; preserve separately because results is git-ignored.
- No production discovery campaign launched/resumed/stopped/edited. Block H is not started; 16B seeds/families now consumed.
