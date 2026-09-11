# NEXT CODEX RUN

block: B
scope: Phase 12 — Machine-Learning Surrogate
recommended_model: GPT-5.6 Sol
recommended_reasoning: High
start_from_task: 12.1

start_command:
Read TOPOSC_LAB_MASTER_PLAN_V3.md and docs/roadmap/EXECUTION_STATE.md. Execute Block B autonomously through its block gate. Do not enter Block C.

previous_block_result:
- Gate 11: PASS
- full test suite: 2703 passed
- fixed benchmark seed 11011: PASS
- exact reference classifications: topological, trivial, topological
- known issues: global pytest temp path is inaccessible in the sandbox; use a workspace-local `--basetemp`
