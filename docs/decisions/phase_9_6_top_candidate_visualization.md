# Phase 9.6: Top-candidate visualization

## Status

Accepted.

## Decision

Top-candidate visualization lives in
`toposc_lab.visualization.candidate_plots` and consumes one immutable Phase-9.5
`CandidateRankingResult`. It reads the ranking and candidate geometries without
recomputing values, reevaluating physics, mutating inputs, or writing files.

The caller must provide a positive `maximum_rank`. The visualization includes every
ranked entry satisfying `rank <= maximum_rank`. Because Phase 9.5 uses competition
ranks, this rule includes a complete tie group instead of truncating tied candidates by
an arbitrary candidate count. `maximum_rank` controls presentation scope only; it is
not a scientific success, eligibility, topology, or discovery threshold.

`TopCandidateVisualization` retains the Matplotlib figure, exactly the displayed axes,
the exact displayed `CandidateRankingEntry` objects in ranking order, the effective
column count, the requested maximum rank, and explicit scientific warnings. It has no
save method, selected-candidate artifact, dataset output, or new ranking data.

Each panel title reports the one-based rank, original Phase-9.3 candidate index, and all
Phase-9.5 criterion values in configured order. It does not rename criteria or attach a
physical interpretation. The figure heading explicitly labels the view as an
engineering visualization.

## Geometry rendering

The implementation reuses the existing model-independent `plot_geometry(...)` path.
That function receives a backward-compatible `show_edge_orientation=False` option;
Phase 9.6 enables it by default so arrowheads display the stored source-to-target
orientation of each visible `GeometryEdge`. The geometry and edge records are not
canonicalized or modified.

Stored one- and two-dimensional coordinates retain the existing plotting convention.
Geometries with more than two coordinate components are labeled as x-y projections.
Geometries without coordinates use the existing deterministic circular plotting layout
but are explicitly labeled `schematic circular layout; no coordinates`. No coordinates,
physical boundary, embedding dimension, or physical meaning are written back to an
abstract graph. Periodic seam edges remain hidden by default under the existing
geometry-plot contract because a straight segment can be misleading; callers must opt
in explicitly.

The function creates no RNG and reads no global random state. Invalid layout arguments,
wrong input types, an empty ranked view, or underlying plot errors raise rather than
producing a partial result. A partially constructed figure is closed on error.

## Scientific and roadmap boundaries

A visually prominent or top-ranked graph is an engineering candidate, not a scientific
discovery. The figure does not establish topology, a bulk or mobility gap, protected
boundary physics, Majorana character, robustness, novelty, causal geometry advantage,
or thermodynamic behavior. It does not compare resource-matched references or perform
independent-seed, finite-size, confounder, ablation, or motif-transfer validation.

No experiment-specific rank cutoff, score policy, threshold, seed split, or search
budget is introduced. Phase 9.6 performs no baseline statistics, reproducible random
search, candidate or figure persistence, dataset generation, evolutionary search,
mutation, ML, GNN, active learning, reinforcement learning, or generative work. Those
remain owned by their later numbered master-plan steps.
