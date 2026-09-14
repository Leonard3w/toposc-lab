# Phase 13R recovery protocol

Ground truth: completed Phase 13 and its failed Gate A, preserved unchanged.
Diagnosis preceded source edits. Retrospective artifacts and separate development
seeds 13201–13205 are diagnostic data, never final benchmark evidence.
The diagnostic catalog reuses 59 exact labels and adds one missing chain solve.
None of these labels are free training data in any benchmark arm.

## Diagnosed causes and bounded recovery

- Across historical runs the first-cycle novelty-bonus span is 0.060–0.168,
  vs value span 0.0017–0.0206. Only 3/60 exploitation slots agree with the pure
  top-three predicted values. Correct dimensional scaling rather than tuning
  weights against old seeds.
- Neighboring sizes recur within batches, including triples of consecutive N.
  Use deterministic greedy feature-space separation within a batch, with a
  training-derived neighborhood radius and explicit quota fallback on exhaustion.
- Every historical pool union covers all 52 non-warm-start candidates; retain
  pool=40, space N=6..65 and uniform initial eight. No generator expansion.
- Development MAE at warm sizes 4/8/12/16: 0.01162/0.00664/0.00508/0.00446;
  top-five overlap remains only 0/0.8/1.6/1.6. More initial labels consume the
  same acquisition budget and are not adopted merely for lower regression error.
- Raw Gaussian 90%-coverage diagnostic at warm=8 varies 0.769–1.0. Test an
  out-of-bag residual floor using training labels only. It is an uncertainty
  estimate, not a conformal coverage guarantee. Retain only if development
  evidence supports it; no calibration on final test labels.
- Explicit training-envelope OOD flags supplement nearest-neighbor flags.
  OOD stays eligible for exact exploration, excluded from exploitation.

## Held-out recovery gate (freeze before running these seeds)

- 20 independent RNG seeds **13301–13320**, disjoint from old and development
  seeds. These are algorithm repetitions in the same finite space, not 20 new
  physical systems. No interim tuning or replacement of unfavorable seeds.
- Same physical parameters, exact evaluator, quality score and constraints as
  original Gate A; 24 exact search calls per arm/seed including eight initial
  labels; original random and Phase-10 evolutionary baseline unchanged.
- Four AL cycles of four acquisitions; same 25% exploration quota and weight
  values 1/0.25/0.25. Improvements are explicitly selectable/versioned; historical
  APIs keep historical defaults for reproducibility.
- Selected recovery configuration must be frozen after development diagnostics
  and before held-out runs. Development ablations assess mechanistic effects,
  not select the best among a broad parameter search.
- Independently verify all 60 arm/seed winners through the same exact pipeline;
  retain mu +/-0.05 sensitivity and three reserved test geometries per seed.
  1680 exact gate solves = 1440 search + 60 test + 180 confirmation/sensitivity.
- Primary quality endpoint: paired final-best difference vs BOTH baselines.
  Sample efficiency separately: paired calls saved to the **unchanged 0.02**
  target, with right-censoring at budget+1 and success rates reported. The target
  is expected to be uninformative; do not replace it post hoc to obtain PASS.
  Best-so-far AUC, median and stricter 0.073 target are supplementary diagnostics
  only and cannot rescue a failed primary gate.
- For each endpoint and each baseline report paired 95% bootstrap intervals,
  exact paired sign-flip permutation p-values (20 seeds => 2^20 assignments),
  and wins/ties/losses. Correct the four one-sided endpoint/baseline tests with
  Bonferroni (p <= 0.0125). An endpoint passes only with positive 95% lower bounds,
  corrected significance and improvements in at least 80% of seeds against
  BOTH baselines. Also require exact confirmation, no leakage/failures, full
  tests. This retains the original robust-win requirement and adds multiplicity
  protection; it does not weaken Gate A.
- Report wall/evaluator times and dimension-cubed work separately from exact
  counts. No speedup claim from equal call budgets alone.
- Bounded experiments: one recovery gate, one fresh-timestamp repeat of seed
  13301 (84 additional solves), development ablations at most five configurations
  on five seeds (2100 solves), plus small diagnostic/test fixtures. No large
  campaign, no Phase 14, no post-holdout retuning.

## Final configuration freeze (before held-out execution)

`TrainingConfig(flag_extrapolation=True)`; `AcquisitionConfig(scaling="target_scale",
diversity_radius_fraction=0.5)`; all other fields unchanged. Radius = half the
median training nearest-neighbor distance in standardized feature space.
Score bonuses are bounded fractions of the robust in-distribution prediction
range; exploration scales uncertainty by that same target range.

Five development configurations were compared by offline replay on seeds
13201–13205 (cached exact records, zero new simulation calls, no benchmark claim).
Legacy / selected recovery mean selected MAE: 0.00671 / 0.00481;
mean minimum batch N separation: 2.15 / 4.00; mean best: 0.0736483 / 0.0736898.
Scaling alone gives 0.0737031 best, so diversity is justified by redundancy/error
reduction, not a claim of superior best quality on this small development sample.
OOB-only changes coverage 0.8875 to 0.9375 without changing selected candidates or
best quality. Adding OOB to recovery widens bands further (coverage 0.9625) without
best-quality gain. The experimental OOB path was removed after its label-exclusion
test passed; no uncertainty/calibration replacement is claimed. Its diagnostic
implementation is archived in `results/phase13r-diagnostics/development-source.zip`.
