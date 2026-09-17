# Phase 17.1 geometry preflight

Status: **decision resolved by user**, 2026-09-16.
The user selected length 2, with unconnected overpasses only at fixed sites.
Positive-length overlap and intersections between fixed sites remain forbidden.
Implementation and verification are recorded separately in the Phase-17.1
implementation report. No Experiment 002 was created or launched.

## Conflict in the requested geometry

On the fixed unit square grid, a displacement with length strictly greater than
sqrt(2) and at most 2 must be `(±2, 0)` or `(0, ±2)`. Every such segment passes
through another fixed site at its midpoint.

`FixedConnectivitySpace.edge_pool` in `src/toposc_lab/research/space.py`
explicitly excludes these segments using the greatest common divisor of the
integer coordinate differences. `validate` rejects them as
`edge_outside_local_pool`. This is an existing geometry restriction, not an
adapter restriction or a mutation acceptance-rate problem. Repeated sampling
cannot produce an admissible length-2 edge under this rule.

The common crossing validator also counts intersections of non-adjacent edges,
including endpoint contact. Removing only the pool restriction does not resolve
the embedding semantics: an edge incident to the midpoint can intersect the
length-2 edge. The validator skips pairs sharing graph endpoints; that must not
be used as an accidental loophole for overlapping collinear edges.

## Exhaustive edge-pool check

These are counts from enumerating the existing space's complete edge pool,
not estimates from stochastic samples. All other settings came from
`ExperimentConfig().space`.

| Sites | Length limit | Pool edges | Edges longer than sqrt(2) |
| --- | --- | --- | --- |
| 16 | sqrt(2) | 42 | 0 |
| 16 | 2 | 42 | 0 |
| 16 | sqrt(5) | 66 | 24 |
| 100 | sqrt(2) | 342 | 0 |
| 100 | 2 | 342 | 0 |
| 100 | sqrt(5) | 630 | 288 |

On both grid sizes, replacing reference edge `(0, 1)` with `(0, 2)` is rejected
with `straight_edge_crossing` and `edge_outside_local_pool`.
Pool membership at sqrt(5) does not establish the existence or diversity of
complete graphs satisfying all constraints; that still needs the requested
seeded sampling, mutation and integration tests.

## Physics inspection

The frozen adapter's `_square_probe` checks the fixed square sites, exterior
boundary and displacement conventions. It does not impose a bond-length limit.
`ChiralPWaveModel.normal_hamiltonian` uses the same scalar hopping on each edge.
`build_chiral_p_wave_pairing` uses the unit direction and the same scalar pairing
amplitude on each edge. Neither introduces distance attenuation.

No adapter incompatibility was identified in this source inspection. Exact
evaluation of an expanded admissible search space remains to be tested. Adding
distance-dependent hopping or pairing would change the physical model and
requires a separately versioned adapter.

## Decision needed

1. Retain the prohibition on passing through fixed sites and use sqrt(5) as the
   new experiment's length limit. This changes the requested numeric limit and
   the requirement to demonstrate edges in `(sqrt(2), 2]`.
2. Retain the requested limit 2 and explicitly specify new geometry semantics
   for site-passing bonds, overlaps and midpoint incident edges. This changes
   existing geometry restrictions and requires opt-in behavior plus validation
   tests; it cannot silently redefine old configurations.

The user selected alternative 2 and explicitly chose the site-only crossing rule
above. New experiments opt in through `space.site_crossings = "unconnected"`.
Absent that option, the previous pool and crossing rules remain in force.

## Budget preparation

The requested existing protocol has `2 + 4 * 4 = 18` exact stages per candidate:
clean, independent confirmation, and four seeds at each of four widths.
Fifty cycles selecting six candidates plus two baselines cost at most
`(50 * 6 + 2) * 18 = 5436` attempts without retries. A budget of 6000 therefore
leaves 564 attempts of headroom. Proposal exhaustion, wall-clock limits and
failures can reduce completion. Interrupted attempts and retries remain charged.

The existing objective `robustness_quality_mean` and requested sampled widths
can be represented without changing the physics contract. Finite disorder
samples do not identify a critical disorder strength or establish a phase,
Majorana validation, finite-size convergence or algorithmic superiority.

## Verification scope

Executed a read-only Python check of the six edge pools, two rejected length-2
swaps, and the adapter's 18-stage plan. No eigensolver or scientific run was
started. No regression pass count is claimed for Phase 17.1; the previous
Phase-17 test results remain historical evidence for that implementation.
