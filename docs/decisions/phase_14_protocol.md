# Phase 14 frozen generator protocol

User authorization accepts the Phase-13/13R negative result and opens Block D.
Historical Gate A reports remain unchanged. AL is optional, without presumed
superiority; Evolution remains the preferred baseline pending new evidence.

## Search space (14.1)

Expand from clean open chains to connected planar embedded graphs on a 6 by 6
unit grid: 36 sites, 60 edges, degree 2..6, coordinates in [0,5]^2, separation
at least 1, edge length at most sqrt(2), no crossings. The 20 perimeter edges
are fixed; interior axial edges and at most one diagonal per unit cell vary.
Boundary sites are the 20 outer-shell sites, never inferred from degree.
This is a bounded wiring/fabrication proxy, not a material fabrication claim.
Model: existing spinless chiral p-wave, t=1, delta=1, mu=2, chirality=+1,
plane=(0,1). No dimension labels, coordinate jitter or model parameters are optimized.

All arms share this support and the first four exact candidates. Random samples
cell diagonal orientations and then 40 non-perimeter edges uniformly. Evolution
wraps Phase-10 population fitness/tournament selection and edge mutation; parents
are the best eight exact archive members, tournament size three, 20% random restarts,
two edge replacements per proposal. Historical exact evaluations are reused only
for parent selection, never as new labels or free cross-arm training data.
New **patch** proposals correlate edge priorities with a random spatial patch and
diagonal/axial preference. New **coverage** proposals choose the maximum minimum
structural distance from 12 random valid proposals. Neither method learns physics.
Optional **active** uses the existing Phase-13 target-scaled acquisition and stump
ensemble, batches of four from 12 random proposals; no acquisition retuning.

## Independent geometry and physics checks (14.12--14.14)

Use Phase-10 validity policy plus explicit lattice/edge/perimeter/model compatibility.
Reject deterministically, record every reason, cap proposal attempts (1000 per pool).
Within each arm reject exact or near duplicates: minimum edge-Jaccard distance
over the eight square rigid symmetries must exceed 0.06 against the evaluated
archive and proposed batch. This eliminates site relabeling, rigid transformations
and small wiring edits in this fixed embedding. It is not general graph-isomorphism
or literature novelty detection. Report raw proposal duplicates separately from
accepted duplicates. Physics score never includes novelty.

OOD: existing feature-distance detector with training-envelope extrapolation, fit
on the same four warm geometries for cross-arm reporting. Also retain AL's own
training-dependent OOD and forbid OOD exploitation. OOD proposals remain eligible
for exact exploration, not invalidated as physics. Coverage descriptors and OOD
features contain geometry only; no future exact labels.

## Exact physics and budget (14.15)

One exact evaluation means one full existing BdG eigensystem plus the existing 2D
class-D spectral localizer at center (2.5,2.5), kappas 0.1,0.2,0.3 (including the
localizer routine's internal spectral solves). Store all eigenvalues, localizer
indices/gaps, PHS residual, four-state boundary weights and Majorana diagnostics.
Eligibility requires all localizers invertible with the same nonzero index and
PHS residual <=1e-10. Quality is eligible times minimum localizer gap; success is
eligible and quality >=0.20. This is finite localizer evidence, not a thermodynamic
phase or Majorana claim. Boundary diagnostics are descriptive, not proof of a mode.
Alternate ranking multiplies quality by boundary weight; disclose sensitivity.

Development seeds 14001,14002 only. Final seeds 14101..14120, 24 exact calls per
arm including four warm starts: 2400 search evaluations. Failures consume budget,
remain in the attempt ledger and never train. Every arm/seed winner receives a fresh
exact repeat and mu=1.9,2.1 checks (300 evaluations). Four reference checks (clean
grid at mu=2 and mu=8, each repeated) precede the gate. One independent seed-14101
rerun costs 135 evaluations. Maximum planned gate/reproduction evaluations: 2839;
development is separately reported. No cached cross-arm labels, no seed retuning.
Same n, edges, solver/grid means equal matrix work per exact call. Record actual
wall/evaluator time and cubic work proxy; all numerical threads set to one.

## Decision rules and conditional branches

Report seed-level best quality, success and first hit, mean pairwise structural
distance, novelty relative to warm start, feature coverage, invalid and duplicate
rates with denominators, OOD rate/quality, runtime and all seed curves.
Primary superiority requires positive paired mean-quality differences against BOTH
baselines, >=16/20 strict seed wins, and positive Bonferroni simultaneous bootstrap
interval lower bounds (six comparisons, 50,000 resamples, seed 14999). No promotion
from post-hoc thresholds. Diversity contributions may justify an optional coverage
tool but not a physics/discovery-efficiency superiority claim. Keep raw Pareto views.

14.3--14.8 RL: NO-GO, Gate 13 prerequisite failed. Existing geometric REINFORCE demo
is not a physics-trained generator and does not override this condition.
14.9 learned-model audit: inspect exact dataset volume, structural identities,
model/representation compatibility; 14.10--14.11 only if compatible independent
geometry diversity supports disjoint training/validation/testing. Repeated chain
sizes, timestamps, parameters or geometric-demo rewards do not provide such support.

Gate completion requires exact winner agreement <=1e-10, deterministic independent
repeat of scientific results, artifact/provenance audit, focused checks and full
tests. A null generator advantage is a valid completed result. Prefer the simplest
supported portfolio; stop before Phase 15.
