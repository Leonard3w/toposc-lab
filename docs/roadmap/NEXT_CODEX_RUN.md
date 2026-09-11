# NEXT CODEX RUN

block: C
scope: Phase 13 — Active Learning
recommended_model: GPT-6 Astra
recommended_reasoning: High
start_from_task: 13.1

start_command:
Read TOPOSC_LAB_MASTER_PLAN_V3.md and docs/roadmap/EXECUTION_STATE.md. Execute Block C autonomously through Discovery Gate A. Do not enter Phase 14.

previous_block_result:
- Gate 12: PASS
- fixed exact-physics benchmark seed 12120: PASS
- selected default: stump_boosting; GNN rejected for insufficient verified improvement
- uncertainty/OOD measurable; calibration remains coarse on the six-case test split
- known issue: use a fresh workspace-local pytest `--basetemp`
