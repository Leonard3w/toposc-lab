# Phase 13R decision: Discovery Gate A FAIL / NO-GO

Recovery is complete. Neither best-candidate quality nor exact-simulation savings
shows a reproducible advantage against both baselines. **Do not start Phase 14.**
Exact numerical physics remains ground truth. The original Phase-13 gate report
is preserved; this decision uses the separately frozen
[recovery protocol](phase_13r_protocol.md) and 20 new seeds, 13301–13320.

## Diagnosis and bounded changes

Diagnosis preceded source edits. Historical runs were inspected, then five
development seeds (13201–13205) were used for acquisition sensitivity and ablations.
These seeds and cached diagnostic labels were excluded from final benchmark fitting.

| Diagnostic | Finding and decision |
| --- | --- |
| Per-seed learning curves | Warm starts nearly saturate the narrow chain family; acquisition has little remaining headroom. Preserve all seed curves, including null results. |
| Exploitation/exploration | Legacy first-cycle novelty spans 0.060–0.168 versus predicted value spans 0.0017–0.0206; only 3/60 exploitation slots overlap pure top-three value. Bound bonuses in robust target-score units; keep the 25% exploration quota. |
| Parameter sensitivity | Development one-cycle mean selected quality is 0.06232/0.06209/0.05746 at novelty weights 0/0.025/0.25 with 25% exploration; 50% exploration gives 0.05396/0.05417/0.05333. This supports fixing scale, not fitting weights to benchmark seeds. |
| Uncertainty | Warm-eight raw Gaussian 90% coverage ranges 0.769–1.0; uncertainty/error rank correlation averages 0.509. OOB residual-floor ablations widen bands without best-quality benefit; remove the prototype and retain existing bootstrap uncertainty. No calibrated-coverage guarantee is claimed. |
| OOD | Supplement reference-distance detection with the training feature envelope. Exclude flagged candidates from exploitation, retaining exact exploration. OOD describes feature extrapolation, not invalid physics. |
| Batch diversity | Adjacent sizes and consecutive triples recur. Add deterministic separation using half the median training nearest-neighbor distance in standardized feature space, with explicit fallback on exhaustion. |
| Pool coverage | Every historical and recovery pool union covers all 52 non-warm-start candidates. Keep the existing candidate generator, pool of 40, and search space. |
| Warm-start sufficiency | At 4/8/12/16 labels, development MAE is 0.01162/0.00664/0.00508/0.00446, but top-five overlap averages 0/0.8/1.6/1.6. Eight labels do not resolve fine ranking; more labels alone have no demonstrated search-budget advantage. Keep eight. |

Offline development ablations use cached exact labels and consume zero new solves.
Legacy versus retained recovery: selected MAE 0.00671 versus 0.00481, minimum batch
size spacing 2.15 versus 4.00, mean final best 0.0736483 versus 0.0736898.
Scaling alone reaches 0.0737031, so diversity is retained for measured redundancy
reduction, without claiming it improves best quality over scaling alone.
Historical API defaults remain available; recovery changes are explicit config.
Physics, score, pool, warm start and baseline implementations are unchanged.

## Fair held-out result

Each arm receives **24 exact search calls per seed**, including the same eight
initial candidates: 20 seeds × 3 arms × 24 = **1,440 search simulations**.
There are 60 reserved-test solves and 180 independent winner/sensitivity solves:
**1,680 exact gate simulations**. A fresh-timestamp reproduction adds 84 and
diagnostic catalog completion adds one: **1,765 Phase-13R non-test exact solves**.
Evolutionary repeats count against its budget (mean 18.6 unique candidates versus
24 for AL/random). Cached labels are never free benchmark training data.

| Metric, mean over 20 seeds | Active Learning | Random Search | Evolutionary Search |
| --- | ---: | ---: | ---: |
| Final best score | 0.07368038 | 0.07369018 | 0.07370357 |
| Final best-score standard deviation | 0.00004856 | 0.00006118 | 0.00003480 |
| Best-so-far mean across budget | 0.07277802 | 0.07280985 | 0.07279641 |
| Exact calls to primary score ≥0.02 | 1.10 | 1.10 | 1.10 |
| Exact calls to supplementary score ≥0.073 | 5.35 | 4.75 | 5.40 |
| Seeds improving their warm-start best | 12/20 | 16/20 | 15/20 |
| Search wall time, seconds | 7.789 | 0.311 | 0.565 |
| Exact evaluator time, seconds | 0.328 | 0.308 | 0.293 |
| Dimension-cubed work proxy, millions | 15.390 | 14.713 | 11.210 |

Positive paired differences favor AL. Seed-level paired bootstrap intervals use
50,000 resamples; the exact one-sided sign-flip test enumerates 2²⁰ assignments.
Its null assumes paired-difference exchangeability. Four primary comparisons
receive Bonferroni correction; PASS additionally requires ≥16/20 wins against
each baseline, positive interval lower bounds, exact confirmation and clean checks.

| Final-quality comparison | Mean AL difference | Paired 95% CI | One-sided p | Wins/ties/losses |
| --- | ---: | --- | ---: | --- |
| AL − Random | −0.000009795 | [−0.000044882, +0.000029011] | 0.6801 | 3/6/11 |
| AL − Evolutionary | −0.000023189 | [−0.000050489, +0.000003164] | 0.9402 | 3/6/11 |

Both multiplicity-adjusted p-values are 1.0. For the primary sample-efficiency
endpoint, savings are zero against both baselines, CI [0, 0], p=1, with 100% target
success in every arm. The shared warm start already reaches this threshold.
The stricter 0.073 diagnostic cannot replace the predeclared gate endpoint.
Equal simulation counts do not imply equal runtime; AL overhead dominates here.

## Learning, OOD and exact candidate verification

Held-out RMSE falls from 0.0005554 to 0.0003626 on average, improving in 14/20
seeds. This is regression improvement without demonstrated search efficiency.
The three fixed test geometries recur across seeds; they are not 60 independent
physical cases. All seeds explore the same finite family N=6..65 at t=1,
delta=0.5, mu=1.8, so inference does not extend to new physical families.

Prequential selected MAE is 0.005108 (320 predictions): 0.003453 for 295
in-distribution and 0.024627 for 25 OOD selections. The stored random-arm/pool
intersection has MAE 0.004681 (905 prediction occurrences); the observed OOD pool
subset has MAE 0.009354 (181 occurrences). These overlaps are descriptive, not
independent or unbiased random test samples. Raw Gaussian 90% coverage is
95.31% selected, 98.23% random overlap and 81.22% OOD overlap; global coverage
therefore conceals weaker OOD behavior and is not a calibration guarantee.
All 80 cycles retain 240 exploitation / 80 exploration selections, no OOD
exploitation, no duplicate AL labels, and no diversity fallback. Mean minimum
within-batch N spacing is 4.35.

All **60 winners** reproduce through fresh exact simulations. The best AL
candidate is N=29, score **0.0737186631**, with splitting 0.0002347072,
next excitation 0.2243553, isolation gap 0.2241206, boundary weight 0.4120931,
normalized polarization 0.9981954 and particle-hole mismatch 1.78e−15.
It can already occur in the shared warm start; this is not an AL-exclusive
discovery. The bulk reference is topological, but these finite-chain metrics
alone do not establish a Majorana or thermodynamic phase claim. Exact mu±0.05
sensitivity results are retained per winner in the machine-readable report.

## Verification, artifacts and decision

Focused recovery/AL/ML tests: **25 passed**. Subsystem regression: **279 passed**.
Full suite: **2,753 passed in 388.85 s** (2026-09-11); command in `EXECUTION_STATE.md`.
Changed source/tests/audit script pass Ruff. Artifact audit passes for 160 JSON
artifacts and 80 cycles, covering budgets, exact records, source provenance,
training/test separation, absence of future-label leakage and OOD routing.
Fresh-timestamp seed 13301 reproduces identical scientific results.
The optional formatter check passes for all 18 recovery/AL/test/audit files;
the inherited Phase-13 `ml/features.py` has two formatting differences and is
preserved in the frozen source snapshot. This is separate from the passing lint check.

- [Machine-readable gate and closeout](phase_13r_recovery.json).
- [Reproduction commands](../phase_13_usage_de.md).
- Local `results/phase13r-diagnostics/`: retrospective, development, ablations and rejected-prototype archive.
- Local `results/phase13r-gate/`: exact datasets, checkpoints, audits and all per-seed/aggregate learning-curve plots.
- Local `results/phase13r-gate-repeat/`: independent seed-13301 reproduction.
- Local `results/phase13r-source-snapshot.zip`: frozen simulation source.

Verified source SHA-256:
`7919ef7e971061310e08e432843f828e143be5b085476c6884372d1c0a90fa9e`.

**Dominant remaining bottleneck:** near-saturated candidate quality in a small
one-family search space, combined with surrogate errors too large to reliably
rank the residual score differences. Pool generation is not the limiting factor.
OOD error, possible over-dispersion relative to local refinement, and expensive
retraining remain limitations. The ablations do not identify a uniquely supported
additional minimal repair. Stop recovery with **Gate A FAIL**; do not tune on
these now-observed seeds, weaken the gate, expand the campaign, or start Phase 14.
