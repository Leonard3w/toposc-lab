# TOPOSC-LAB EXECUTION STATE

current_block: C
current_phase: 13
next_task: 13.1
last_completed_task: 12.12
last_verified_commit: 68151d7
last_full_test_result: `PYTHONDONTWRITEBYTECODE=1 BLIS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv\\Scripts\\python.exe -m pytest -q -p no:cacheprovider --basetemp=.pytest-tmp-phase12-final-full-2` -> 2729 passed in 327.14 s, 2026-09-11
last_block_gate: SURROGATE_READY PASS

known_issues:
- The global pytest temp directory is inaccessible in the sandbox; use a fresh workspace-local `--basetemp` path.
- Mypy cannot parse the installed NumPy stub under the current Python 3.14 environment (`type` statement syntax); Ruff remains available.
- Full-tree Ruff currently reports 145 pre-existing findings; Phase-11/12 files pass Ruff.
- Gate-12 calibration coverage is coarse (6 test cases, empirical 0.5 for nominal 0.9); re-evaluate on larger/diverse exact data before scientific use.

scientific_decisions_in_force:
- exact numerical physics remains ground truth
- a low-energy state alone is not a Majorana claim
- no thermodynamic/topological phase claim from one finite graph
- existing basis, topology, and robustness conventions remain unchanged through Phase 12
- ML predictions prioritize exact simulations and are never scientific ground truth
- OOD is a feature-space acquisition warning, never a claim of invalid/bad physics
- the default surrogate is stump boosting; the minimal GNN lacked verified improvement

notes_for_next_task:
- Block B / Phase 12 is complete; do not reimplement it.
- Start Block C at 13.1; use `toposc_lab.ml` and the public `toposc_lab.data` split API.
- Gate seed 12120: stump-boosting RMSE 0.1300 vs trivial 0.5958; GNN rejected.
- Predictions must carry uncertainty/OOD plus exact dataset/version links downstream.
- Every acquired label/winner still requires exact-solver verification.
