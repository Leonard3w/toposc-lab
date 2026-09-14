# Phase-15 frozen autonomy protocol v1

This gate tests orchestration reliability, not a new generator superiority claim.
Phase 14 remains accepted. Phase 16 is outside this execution block.

- Primary campaign: seed 15201 (separate from development fixtures), three cycles, four exact selected candidates per
  cycle, eight valid proposals per cycle, maximum 1000 raw proposals per cycle.
- The first cycle uses a common Random warm pool. Subsequent cycles use patch by
  default only in the Phase-14 36-site, 60-edge planar wiring stratum. Random and
  Evolution receive identical seed, proposal pool, and exact budgets as controls.
- Optional surrogate campaign: identical configuration with surrogate enabled;
  retraining cadence two cycles, exact completed-prefix labels only. No ML
  superiority or calibration guarantee is assumed.
- Model, geometry, basis, three localizers, tolerance and quality definition are
  unchanged from Phase 14. Success remains eligible quality >=0.20.
- Every selected candidate receives an exact clean evaluation, an independent
  exact confirmation, and four exact onsite-disorder realizations of uniform
  offsets in [-0.1,0.1], embedded with opposite particle/hole signs. Each call
  includes a BdG eigensystem and three localizer calculations. Per campaign:
  12 clean + 12 confirmation + 48 disorder = 72 evaluations. Eight retry attempts
  are reserved. Interrupted attempts remain charged against the cap of 80.
- Report mean, minimum, SD and standard error of ensemble quality, plus 95%
  Wilson interval for the frozen success fraction. Four realizations are a small
  engineering smoke ensemble, not a robust-physics certification.
- Majorana reporting retains operator/spectral PHS, low energies, site
  probabilities, IPR, boundary weight, local complex polarization and
  self-conjugacy. Existing splitting diagnostics use zero tolerance 1e-10,
  splitting tolerance 1e-3 and pair PHS tolerance 1e-8. No separated Majorana claim.
- Finite-size/family scaling is unavailable: this fixed wiring space has no
  declared candidate-specific size-extension map. Do not substitute unrelated
  geometries or parameter sensitivity for finite-size evidence.
- Duplicate distance is square-symmetry-minimized edge Jaccard, with rejection at
  distance <=0.06 against an external exclusion dataset, earlier valid pools and
  within the current pool. Raw invalids and duplicates are separately counted.
- Common OOD uses only the first completed exact batch. Before that reference
  exists, status is explicitly unknown and conservatively flagged. OOD is a
  warning, never invalid physics. Surrogate OOD cannot drive exploitation.
- Independent repeat, checkpoint-after-one-cycle resume, and process-kill/resume
  campaigns use identical seeds/config. Compare complete scientific records
  exactly after removing record ID and provenance (which include timestamps).
  Audit seed/config/source separately; predictions must never enter exact data.
- Reliability trials: reference, resumed, repeated, killed/recovered patch
  campaigns plus Random, Evolution and optional-surrogate controls: seven small
  72-evaluation campaigns, plus one interrupted attempt, plus two physics
  references (507 non-test attempts expected). No large discovery campaign.
- Automated tests additionally inject failures at all durable boundaries,
  numerical failure, corruption, duplicate and budget exhaustion conditions.

Accept AUTONOMY_GATE only with reproducible exact records, clean dataset updates,
bounded recovery, distinct candidates, preserved scientific contracts, passing
focused/regression/full suites and reviewable source/config artifacts. Improvement
is descriptive; stagnation is allowed. Report failures honestly and stop before
Phase 16. Passing permits small monitored campaigns in this stratum, not automatic
self-certification or unbounded searches.
