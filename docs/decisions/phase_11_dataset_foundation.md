# Phase 11 — Dataset Foundation Gate

date: 2026-09-11
gate: DATASET_FOUNDATION
result: PASS
base_commit: dadd0da
working_tree: dirty (Phase-10 user work and uncommitted Block-A implementation)

## Decision

The Phase-11 dataset foundation is accepted. Exact numerical results are
stored in a versioned typed schema under the existing `toposc_lab.data`
namespace. The storage layer does not calculate, predict, or reinterpret
physics.

## Evidence

- Tasks 11.0–11.15 completed.
- Phase-11 focused tests: 38 passed.
- Geometry/data/topology/robustness regression checkpoint: 161 passed.
- Full repository suite: 2703 passed in 726.55 s with a workspace-local pytest
  temp directory.
- Fixed benchmark seed: 11011.
- Canonical benchmark dataset SHA-256:
  `439bc23ceb1afc3f7231a53706dc2deaeec9d4f52117241fbab45ede1d886db5`.
- Exact Kitaev reference classifications: topological, trivial, topological.
- Benchmark split counts: train 1, validation 1, test 1; no family leakage.
- Dataset write/load bytes and record identities are deterministic.
- Current-schema loader rejects old schema markers; compatible pilot v0 requires
  the explicit migration API.
- Ruff and isolated strict Mypy checks pass for all new Phase-11 modules.

The full pre-existing tree has 145 Ruff findings outside the Phase-11 change
scope. They were measured but not refactored because they neither affect this
gate nor justify unrelated changes.

## Identity and duplicate policy

The exact archive SHA-256 and the relabeling-invariant 1-WL family fingerprint
remain separate. Fingerprint equality is not treated as proof: an exact
configured graph-isomorphism check must confirm it. Search-budget exhaustion
is reported as uncertain. Split grouping is conservative, so possible related
families may be over-grouped but cannot be knowingly split across train and
evaluation sets.

## Scientific impact

No BdG basis, topology definition, Majorana criterion, robustness definition,
solver tolerance, or other physics convention changed. The benchmark stores
the finite open-chain spectrum separately from the translation-invariant bulk
Pfaffian reference and records that distinction explicitly.
