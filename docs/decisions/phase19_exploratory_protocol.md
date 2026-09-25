# Phase 19: frozen exploratory family comparison

2026-09-25. Scope: incremental embedded-graph extension and a bounded descriptive
experiment, not a confirmation study or new optimization algorithm.

Baseline: checkpoint c666246, tag `phase18-confirmed-before-embedded-graphs`;
3140 pre-change tests passed. Three fixed historical cases recomputed to 1e-10,
all 2510 saved records checksum-verified without historical writes.

Reuse Geometry/GeometryEdge, ChiralPWaveModel, all BdG/hopping/pairing/disorder
builders, existing constraints/descriptors, ResearchStore and ValidationStudy
checkpoint/run/recovery. New adapter changes applicability, never legacy Q.

## Gate before new-family physics

Square built directly with existing Geometry/GeometryEdge must reproduce old
Hamiltonian, spectrum, Q, raw diagnostics and Chern/window values within 1e-10.
Hermiticity, PHS and pairing antisymmetry must pass. These tests passed before
constructing new family cohorts. Stop on nontrivial regression/physical failure.

## Cohort and fairness

One deterministic regular 8x8 baseline, not 50 fictitious independent copies.
50 unique valid graphs in each of rewired_square, amorphous_planar and
constrained_embedded. Fixed candidate counts, no outcome-based selection.
Master geometry seed 190000; per-proposal PCG64 seeds persisted.
Maximum 1500 external proposals per family, existing internal point budget
1000000 and complete-attempt budget 10000 per amorphous proposal. Record external
acceptances/rejections and internal rejection counters; generator failure cannot
silently relax limits or shrink the cohort.

All: 64 sites, exactly 112 edges, degree 2..4, connected, coordinates [0,7]^2,
minimum separation .55, maximum physical bond length 1.75, no straight-edge
crossings, 24..32 boundary sites. Physical boundary: distance to a coordinate-box
side <=.875; not graph degree. No periodic bonds or altered displacements.
Edges crossing at a site without a declared endpoint are prohibited by geometry
constraints. Families are algorithmic constrained ensembles, not uniform graph samples.

Square rewiring uses existing 8/16/32/64 edit sampler, revalidated by the common
policy. Amorphous family uses the existing frozen hard-core Delaunay generator.
Constrained embedded family reuses its point process and random-priority
spanning/completion algorithm, with the existing radius-cutoff edge pool and
incremental crossing rejection. Same constraints; different connectivity ensemble.
Point clouds normalized by the existing generator are checked again afterward.

Sites sorted lexicographically by x,y in every family. Same disorder seeds produce
the same iid onsite vector by index. For different coordinates this is common
random-number coupling, **not** the same spatial disorder field. Physical graph
identity includes coordinates/edges/boundary, excluding generator labels.

## Physical contract

t=1, mu=2, pairing=1, chirality +1, component-major BdG, unchanged normalized
direction pairing and no bond-length decay. Δij=-Δji. Length is constrained and
reported, not converted into a new hopping law. Localizer scales .1/.2/.3,
numerical tolerance 1e-10, same eligibility and Q>=.20 success criterion.

Center (3.5,3.5), existing square-domain spatial probe pattern, bulk inset 2.
Local Chern marker uses the existing clipped Voronoi algorithm on [-.5,7.5]^2,
area 64, matching unit square-cell areas on the baseline. This integration cell
is distinct from the coordinate-box shell definition. Store areas explicitly.
Marker remains descriptive without independently established mobility gap.
Boundary energy window |E|<=.5, whole near-degenerate groups, existing strips.
Center weight: fixed physical square [2.5,4.5]^2, including whichever sites lie
there; counts recorded so different site density cannot be mistaken for equal masks.
All interior gaps are localizer gaps, never relabeled bulk spectral gaps.

One clean reference per geometry (seed191000), W=3/6/9 with three new seeds
191001/191002/191003. 1510 planned stages; at most1600 attempts, 1800s,
one worker/BLAS thread. No multi-hour autonomous search. Technical smoke/timing
uses separate output, no scientific selection. Stop incomplete if budget expires.

## Analysis and stop

Distributions of all scalar raw diagnostics by family/W, candidate-level mean Q
and success, descriptive baseline differences, graph-feature vs physics Spearman
correlations within family/W. Same seeds across candidates are dependent; no
pooled-realization significance claims or family superiority from tiny samples.
Best mean-Q candidate in each family shown descriptively (ID tie break), plus
baseline; show clean geometry/spectrum and W=6 first-seed window localization.
Save every record, not only examples. Retain clean and disordered diagnostics.

Explicitly count success with spatially inconsistent interior topology and report
score/raw-diagnostic disagreements. Known center-score spatial limitations do
not become retrospective failures or a new score. A new serious numerical or
physical flaw stops the study and is reported before further work.

After tests, run and report: STOP. No additional search, tuning, neural methods,
MAP-Elites, new phase/Majorana claims, or candidate confirmation without new instruction.
