# Discovery Gate A: frozen protocol v1

Frozen before running comparison seeds. No tuning on these results.

- Scope: controlled open Kitaev chain geometry family, N=6..65, t=1,
  Delta=0.5, mu=1.8. This is a finite-family search-efficiency benchmark,
  not a new phase or a general arbitrary-graph discovery claim.
- Seeds: 13101, 13102, 13103, 13104, 13105. Every arm starts from the same
  eight uniformly sampled distinct geometries; these eight exact simulations
  are charged independently to every arm. No pretrained/free physics labels.
- Total search budget: 24 exact solver calls per arm/seed. AL: eight initial
  labels plus four cycles of four acquisitions, pool 40, exploration quota 25%,
  acquisition weights value/uncertainty/novelty = 1/0.25/0.25.
  Stump-boosting bootstrap ensemble: 8 members, 40 estimators, distance weight 0.5.
- Random: uniform without replacement, same initial eight. Evolution: existing
  Phase-10 `run_generation_loop`, population eight, two generations,
  tournament size three, minimum one elite; bounded N mutation by a uniformly
  sampled nonzero step in [-6,6], 20% uniform immigrants. Re-evaluated elites
  and duplicates consume budget, as required by the existing baseline contract.
- All arms use the identical exact evaluation and scalar objective. For the
  lowest-energy particle-hole pair retain splitting, next excitation, isolation,
  mean probability on the first/last two sites, polarization and PHS mismatch.
  Quality = valid bulk-reference topology * isolation/(1+isolation)
  * boundary weight / (1+splitting/0.01). Weights and energy scales are fixed.
  Near-zero energy alone cannot produce a high score. Bulk endpoint topology
  applies only to the translation-invariant parent, separately from finite spectra.
- Fixed quality threshold for first-hit/sample efficiency: 0.02. Also report
  best-so-far AUC, final best and median, unique labels, solver dimension-cubed
  work proxy, exact-solver and total wall times, and prequential prediction error
  by OOD status. Exact calls are matched; wall time is measured, not equalized.
- Reserve N=68,70,72 as untouched test geometry families. Charge these three
  diagnostic solves separately per seed; never select/tune/train on them.
- Independently re-solve each arm's winner with fresh seed, then mu +/-0.05
  as sensitivity checks (three extra solves per arm). Re-rank stored candidates
  with splitting scale 0.005 and 0.02 without additional solves.
- PASS requires a positive lower paired 95% bootstrap confidence bound for
  final-best AL improvement against BOTH baselines, plus strictly positive
  differences in at least 4/5 seeds, independent winner agreement, no leakage,
  transparent failures/OOD, and the full suite passing. This conservative
  operational criterion tests the master's equal-budget quality option.
  A null/negative result stops Block C; no parameter hunting or Phase 14.
- Reference source identity includes Git HEAD, dirty flag and a SHA-256 of the
  source tree. A run timestamp is fixed across that run's records and retained
  with checkpoints. Reproduction compares scientific content, not wall times.

Implementation erratum after first run (no protocol/parameter change): the initial
training implementation sorted by full record ID, which includes timestamp and
source metadata. It now sorts by geometry/model candidate identity, so a changed
execution timestamp cannot change bootstrap samples. The first 420-simulation
run is retained in `results/phase13-gate-a`; its empirical result was FAIL.
The corrected frozen comparison is run once in `results/phase13-gate-a-final`.
These runs are not pooled as independent evidence. A development-seed test checks
identical scientific outcomes under changed timestamps; no gate-seed tuning.
