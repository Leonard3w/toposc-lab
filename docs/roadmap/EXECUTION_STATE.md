# TOPOSC-LAB EXECUTION STATE

current_block: B
current_phase: 12
next_task: 12.1
last_completed_task: 11.15
last_verified_commit: dadd0da
last_full_test_result: `.venv\\Scripts\\python.exe -m pytest -q -p no:cacheprovider --basetemp=.pytest-tmp-phase11-final-full-1` -> 2703 passed in 726.55 s, 2026-09-11
last_block_gate: DATASET_FOUNDATION PASS

known_issues:
- The global pytest temp directory is inaccessible in the sandbox; use a fresh workspace-local `--basetemp` path.
- Mypy cannot parse the installed NumPy stub under the current Python 3.14 environment (`type` statement syntax); Ruff remains available.
- Uncommitted Phase-10 constructive-design work predates Block A and must be preserved.
- Full-tree Ruff currently reports 145 pre-existing findings; Phase-11 files pass Ruff.

scientific_decisions_in_force:
- exact numerical physics remains ground truth
- a low-energy state alone is not a Majorana claim
- no thermodynamic/topological phase claim from one finite graph
- existing basis, topology, and robustness conventions remain unchanged in Phase 11

notes_for_next_task:
- Block A / Phase 11 is complete; do not reimplement it.
- Start Block B at task 12.1 using the public `toposc_lab.data` API.
- Use `DatasetSplit` unchanged so geometry families cannot leak across ML splits.
- Exact dataset labels remain ground truth; predictions must retain dataset/version links.
- Gate tests must use a fresh workspace-local pytest base temp directory.
