# Phase-16A frozen protocol - Block F, tasks 16.1-16.12

Protocol v2, amended before any Phase-16A physics outcomes. Phase 16B is not authorized.
Exact physics and all Phase-15 scientific definitions remain fixed. This block
ends at the Internal Mechanism Checkpoint, not MECHANISM_GATE.

## Data and budget

Use the four Phase-15 control arms (reference/Random/Evolution/surrogate), merge
symmetry-equivalent geometries with the existing D4 edge-orbit key, retain every
origin and exact raw record. Repeated recovery campaigns are audit evidence only.
Generate 64 additional valid geometries: seeds 16101-16108, eight per seed,
alternating Random and Patch by seed (four seeds each). Reject D4 edge-Jaccard
distance <=0.15 from all legacy and accepted fresh geometries. Cap raw generation
at 2000 per seed. No physics-based selection of this fresh sample. These seeds
are consumed by Phase 16A and cannot serve as Phase-16B independent validation.

Reserve a maximum of 700 non-test numerical evaluations/attempts for this block,
including confirmations, simplifications and two regular-grid physics references.
The v1 draft inconsistently specified six seeds and an eight-seed survival rule.
This pre-outcome correction makes that rule evaluable: at most 660 completed
evaluations (128 fresh, 512 interventions, 16 simplifications, two references,
two independently rerun fresh evaluations), leaving 40 attempts for interruptions.
Store each attempt before calculation and each completed result atomically.
An interrupted calculation is charged; resume requires identical source, driver,
protocol and input checksums. Four numerical thread limits are one.

## Features and analysis

Start with graph degree histograms, clustering, paths, precisely enumerated K3
and induced C4 motifs, boundary/bulk features, adjacency and combinatorial and
normalized Laplacian spectra. Coordinate distances, edge directions, bond angles,
explicit boundary-hole labels and finite ball growth are descriptive. No dimension
fit is justified on this single size. Fixed-size/edge/boundary/coordinate features
are reported as constants and excluded from statistical testing.

Analyze legacy and fresh cohorts separately. Primary outcome is the frozen
eligible minimum localizer gap; also retain ungated gaps, topology eligibility,
minimum |E| and low-energy boundary weight. Success is still quality >=0.20.
If there are no successes, the success comparison is unavailable. Exploratory
upper-quartile comparisons are explicitly separate; use the legacy upper-quartile
cutoff on the fresh cohort without redefining success.

Report raw Spearman and partial rank correlations adjusted for degree histogram
(degrees 2-6), boundary mean degree, and generator membership. Fresh analyses
also include seed indicators. Use 1999 residual permutations restricted within
seed/generator cells (legacy: origin membership), with a +1 correction. These are
exploratory tests conditional on the linear rank adjustment/exchangeability model.
BH and conservative BY adjusted p-values cover all variable features per cohort.
Use family bootstrap intervals, and fresh leave-one-seed-out sign checks. Feature
importance uses training-only standardization, fixed ridge and distance-neighbor
models, leave-one-seed-out tests with distance <=0.15 train/test purging, repeated
test-column permutations, and full baseline/prediction errors. Importance does
not imply causality; correlated-feature substitution can destabilize rankings.

Geometry families use complete pairwise symmetry-minimized edge-Jaccard distances
and single-link components at 0.10/0.15/0.20; report chaining sensitivity. Clustering
in standardized variable feature space is descriptive. Outliers are ranked by
distance from the legacy feature center, with exact quality shown separately.

## Predeclared intervention targets

1. `triangle_count`: unordered triples inducing K3.
2. `chordless_square_count`: unordered four-vertex sets inducing connected C4,
   with no diagonals/chords.
3. `boundary_bulk_edges`: edges with exactly one endpoint on the fixed perimeter.
4. `central_diagonal_count`: length-sqrt(2) edges whose midpoint lies within
   radius 1.5 of (2.5,2.5), including the radius boundary.

For each motif and each fresh seed, try candidates in descending exact quality
until two parents have matched valid interventions (up to 16 parents per motif).
Record unavailable parents. Enumerate valid double-edge swaps: two disjoint edges
are replaced by two absent edges, preserve each vertex degree, connectivity,
planarity, perimeter and every stratum constraint. Target change must be negative;
control target change must be zero. Match total diagonal-count change and
boundary/bulk-edge change except when that quantity is itself the target; minimize
remaining standardized non-target motif changes, ties selected by seed. Scales
are population SDs across all valid swap deltas for that parent (zero SD becomes
one); this calculation uses no physics labels. Both target and control edits
change two edges and preserve every individual vertex degree.
Use two predeclared perturbation seeds per parent when distinct matched pairs
exist. No physics is used to select a swap or control and no post-ablation
optimization occurs. Correlated non-target changes remain explicit confounders.

Exact-evaluate and independently confirm every new geometry. Store spectrum,
all localizer indices/signatures/gaps, PHS, four low-energy profiles, IPR,
polarization and boundary weights. Primary paired effect is target-minus-control
quality (both share the same parent); also show each change from parent. Average
perturbation replicates within parent, then report parent and seed bootstrap
intervals, seed means, topology transitions and diagnostic changes. Multiplicity
across the four targets uses exact seed-sign-flip tests and Holm correction.

A pattern survives this limited ablation only if the seed-level 95% interval
excludes zero in the hypothesized direction (beneficial motif removal should
reduce quality), at least seven of eight seed means agree, and Holm p<=0.05. Absence
of power is inconclusive, not proof of no effect. Positive correlation that loses
its benefit under controls is not promoted as a design rule. Report 3-5 ranked
candidate questions for Phase 16B even if none meets this survival criterion,
clearly distinguishing weak/falsification priorities from supported hypotheses.

## Simplification and checkpoint

For the best fresh parent per seed, delete one removable non-perimeter edge
selected deterministically by seed, without score-based selection or optimization.
Require 36 fixed sites, 59 edges, connected planar graph, degrees 2-6, unchanged
coordinates/perimeter, and length <=sqrt(2). This is an explicit diagnostic
extension, not a change to the Phase-14/15 discovery stratum. Evaluate with the
same Hamiltonian/localizer definitions and independently confirm. Report exact
quality retention >=90% plus unchanged eligibility and raw indices, without
claiming an edge is universally unnecessary. No node removal/scaling claim.

Run focused, subsystem/reference and full-suite checks. Freeze raw inputs,
features, selection plans, exact records, analysis, source archive and checksums.
Reproduce a fresh-seed reference and regenerate analysis from stored exact results.
No broad robustness ensemble or cross-model/finite-size validation is added here.
