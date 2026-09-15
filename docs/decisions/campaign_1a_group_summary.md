# Campaign 1A: stored multi-seed summary

The ten included independent campaign seeds are **17102–17111**. All completed
10/10 cycles, with batch size 4 and pool size 8, under the same Patch discovery
configuration and unchanged success threshold **Q >= 0.20**. The first cycle
uses the engine's existing Random warmup, which remains labeled in the records.

**Seed 17101 is excluded as `technical_dropout`** because of the diagnosed
Windows persistence failure. Its data remain preserved and inspectable. It
contributes to no scientific total, quality statistic, curve, success
denominator or group leaderboard below. This is a descriptive analysis of
stored authoritative records, not a new campaign or numerical calculation.

| Campaign Seed | Best exact quality | Last improving completed cycle |
|---|---:|---:|
| 17102 | 0.117698494734 | 4 |
| 17103 | 0.127261012235 | 2 |
| 17104 | 0.118048284079 | 4 |
| 17105 | 0.125438010315 | 5 |
| 17106 | 0.125189797215 | 5 |
| 17107 | 0.129847314619 | 8 |
| 17108 | 0.124496844592 | 4 |
| 17109 | 0.127289288947 | 3 |
| 17110 | **0.140055117843** | 6 |
| 17111 | 0.129462038091 | 7 |

## Totals and descriptive statistics

- Included complete / failed / incomplete seeds: **10 / 0 / 0**.
- Separately excluded technical dropouts: **1**, seed 17101, stored status Failed.
- Generated proposals: **1,927**; accepted unique pool candidates: **800**;
  invalid: **1,096**; duplicates/near-duplicates: **31**. These partition the
  generated proposals: 800 + 1096 + 31 = 1927.
- Exact candidate records: **400**. Successful exact evaluations across clean,
  confirmation and disorder stages: **2,400**; charged attempts: **2,400**.
- Mean per-seed best: **0.126478620267**; median: **0.126349511275**.
- Sample standard deviation of per-seed best (`n-1`): **0.006330455770**.
- Per-seed best min/max: **0.117698494734 / 0.140055117843**.
- Candidates reaching the frozen threshold: **0/400 (0%)**.
- Seeds with any candidate reaching the threshold: **0/10 (0%)**.

All ten best-so-far curves are flat in completed cycles 9 and 10. The latest
improvement in the group occurred in seed 17107 at completed cycle 8. This
describes the observed finite budget; it does not establish scientific
saturation or rule out improvements with further search. No further search was
started for this summary.

## Global best stored candidate

Campaign Seed **17110**; Exact Eval Seed **1156657121**; generator **patch**;
zero-based cycle **5** (sixth completed cycle); exact quality
**0.14005511784303276**. Stored topology status is `valid`, localizer indices
**[1,1,1]**, minimum |E| **0.0815515448025628**, and exact confirmation is true.
Stored OOD is true, reason `warm_reference`, score **1.5345723938936007**.

Stored four-sample `uniform_onsite_finite_localizer` disorder evidence has
eligible fraction **1.0**, quality mean **0.13736154216208932**, minimum
**0.13485642362764216**, SD **0.0018070576905658271**, and success fraction **0**.
Its scope is fixed geometry with scalar onsite disorder only. These are copied
diagnostics; **`diagnostics_only`, `majorana_claim=false`** remain explicit.
No Majorana or thermodynamic claim is made.

Geometry:
`geometry-archive-v1-sha256:7f502b91aaa361994c12e29a90a2a2d49bee5a359524ff7bcff32802eb47758a`.
Record:
`dataset-record-v1-sha256:d7691e5de2748bcd3896e89c479e4ccea1fabfa48573195bfb0fcbbff21f06eb`.
Stored journal: `seed-17110/cycle-0005/candidate-0002.json` under the original
Campaign 1A parent.

## Compatibility and provenance

The [saved group definition](../campaign_groups/campaign_1a.json) records all
paths, expected seed IDs and the dropout reason. Seeds 17102–17110 reside under
`results/gui-smoke-test/discovery-1a-patch/new-campaign`; seed 17111 resides at
`results/new-campaign`. All official scientific, selection and budget fields,
stored protocol identifiers, runtime environments and external exclusion lists
match. The reader validates checksums, committed inventories and exact records
without invoking any solver or modifying data.

The source fingerprint is
`d8be4164e285f4dfb2b4a5bdeacb28c0b450a61b0753616713d9261bdb32c8ba`
for seeds 17102–17110 and
`9883b54414e1ea7cfaf306e7e55f4a99f46e7adf6b03c54b151a8fdc6847978e`
for seed 17111. Comparing Python bytes in the two archived source ZIPs finds
differences only in `src/toposc_lab/_file_io.py`,
`src/toposc_lab/discovery/storage.py`, and `src/toposc_live/processes.py`, the
documented Windows storage fix. The GUI retains an explicit source-difference
warning; no source/resume policy is relaxed.

## Artifacts and validation

Usage and calculation definitions: [Campaign groups](../campaign_groups.md).
Machine-readable exports are `results/campaign-1a-group-final.json` and
`results/campaign-1a-group-final.csv`. The CSV includes summary, member and top
candidate rows with explicit row types. The `results/` tree is git-ignored and
must be preserved separately.

Focused discovery/live/storage regression: **126 passed in 63.20 s**. Additional
group/GUI checks: **37 passed in 3.65 s**, covering compatibility, corruption,
dropout exclusion, exact maxima, threshold inclusion, seed identity, exports,
read-only repeated refresh, concurrent atomic writes and stale refresh handling.
Native Windows captures of the summary and group leaderboard were inspected.
The independent direct-journal arithmetic matches every per-seed maximum and
the global success counts. Preservation audit: **all 6,636 files** across all
eleven campaign directories match their initial hashes, with no added or missing
files (`results/campaign-group-preservation.json`). No production campaign was
started, resumed, stopped or edited.
Full repository regression: **2,953 passed in 448.56 s**
(`results/campaign-group-full.log`). After a final malformed-group-definition
input guard, the affected group/GUI/reader tests pass **64/64 in 4.31 s**
(`results/campaign-group-final-focused.log`), including four new malformed-input
cases. Ruff passes; isolated strict Mypy passes for all 13 frontend modules with
`--python-version 3.14 --follow-imports silent --ignore-missing-imports`.
The full suite uses isolated fixtures, including its existing numerical tests;
it does not operate on the production campaign directories.
