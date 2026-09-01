# Pre-Phase 9.8 random-search protocol, version 1

## Status

**Accepted on 2026-09-01 before Phase 9.8 implementation or execution.**

Protocol identifier: `TOPOSC-P9.8-RS-001`

This is the experiment decision required by the accepted
`pre_phase_9_research_charter.md`. It freezes the version-1 contract but does
not itself change production code or record a random-search result. Phase 9.8
remains blocked until all of the following have happened in this order:

1. this accepted document has been committed separately from Phase-9.8 code;
2. the Phase-9.8 implementation records that exact commit as its protocol
   revision;
3. the implementation prerequisites below have been completed and tested; and
4. a dry run using only designated dry-run seeds has passed without changing
   any frozen scientific choice.

The protocol revision is the full Git commit hash that first contains this
accepted file at its final path. The implementation and every result manifest
must record that hash; the file cannot contain its own future commit hash.

The only numerical physics inspected while preparing this decision was a
declared calibration on the existing, named, open square-lattice reference.
A geometry-only prototype used the reserved dry-run seeds to check that the
frozen constraints were constructible; it built no Hamiltonian and retained
no file. No search-role candidate geometry, amorphous reference realization,
search ranking, search success rate, validation seed, or confirmation seed was
evaluated. Both preparation checks are recorded below so that they cannot
silently influence a later amendment.

This acceptance freezes the protocol before any search outcome was inspected.
Any later change to a scientific value, gate, seed partition, reference,
budget, or ranking rule creates a new protocol version. It must state whether
any search, validation, or confirmation outcome was already visible and must
allocate new untouched seeds where leakage could otherwise occur.

## Intended claim boundary

Version 1 is accepted as a **finite-size engineering random-search benchmark
and scientific screening experiment**, not as a completed answer to the
charter's central discovery question.

It may establish only:

- the reproducible Phase-9 random-search hit rate under this exact protocol;
- whether a finite 64-site candidate passes the predeclared clean screening
  gates;
- separate finite-size success fractions for the six declared disorder
  channels; and
- whether later search methods beat the same frozen engineering benchmark.

It may not establish a thermodynamic phase, a bulk or mobility gap, a causal
geometry advantage, a new graph family, a general design rule, literature
priority, or a material prediction. The quasiperiodic and fractal references
available in the repository cannot be fully resource-matched at the frozen
64-site stratum. They are therefore descriptive controls in version 1, and
their presence cannot repair that claim limitation.

The experiment studies chiral Majorana boundary **signatures**. It does not
search for isolated Majorana zero modes. Individual-state Majorana
self-conjugacy is retained as a diagnostic but is not a success threshold: a
finite chiral edge spectrum normally consists of particle-hole-related
nonzero-energy states rather than two separated Kitaev-chain end modes.

## Repository capability audit

The following accepted infrastructure is reusable without changing its
scientific meaning:

- the Phase-6 generator protocol, ordered registry, explicit stochastic flag,
  and generator-owned seed request;
- Phase-6 geometry validation, serialization, exact edge orientation, and
  immutable generation provenance;
- the Phase-6 graph hash only as an isomorphism-candidate fingerprint;
- the Phase-7.11 exact geometry ID as a representation-dependent snapshot ID;
- `ChiralPWaveModel` with explicit geometry, chirality, pairing plane, and
  component-major Nambu basis;
- the Phase-7 candidate-validity and `evaluate_geometry(...)` failure
  boundaries;
- the class-D topology dispatcher and the Bott-index, local-Chern-marker, and
  spectral-localizer implementations;
- the Phase-8 explicit PCG64 disorder contracts, ordered ensembles, failure
  accounting, success fractions, Wilson intervals, and separate channels;
- the Phase-9.1 PCG64 geometry sampler and derived generator seeds;
- the Phase-9.2 parameter sampler, although version 1 deliberately fixes the
  model parameters and therefore does not call it;
- Phase-9.3 ordered failure-aware batch evaluation;
- Phase-9.4 lossless candidate-input/outcome ledger;
- Phase-9.5 caller-owned lexicographic ranking;
- Phase-9.6 visualization only after ranking; and
- Phase-9.7 repeated-trial baseline statistics.

The audit also found six prerequisites that the repository does not currently
provide:

1. **No eligible stochastic embedded generator.** All four registered
   stochastic generators (`random_graph`, `random_regular_graph`,
   `small_world_network`, and `scale_free_graph`) are abstract. The embedded
   two-dimensional generators are deterministic or consume an already-created
   point cloud. None can generate the frozen random, planar, constrained
   candidates through the common seed protocol.
2. **No charter-level geometric constraint report.** Base validation checks
   representation, duplicates, connectivity, dimensions, and metadata, but it
   does not enforce minimum separation, edge crossings, straight-line
   planarity, degree bounds, spatial extent, coupling range, or a frozen
   physical-boundary rule.
3. **No reusable topology-input builder.** The experiment must currently
   construct component-major Nambu coordinates, clipped position areas, bulk
   masks, Bott periods, and localizer probes itself. Guessing these per
   candidate would make the comparison unauditable.
4. **No typed topology-convergence bundle.** `GeometryEvaluation` stores one
   result per method and one convergence flag. It cannot retain the parameter
   grids from which that flag was justified.
5. **No particle-hole-pair boundary record or named protection proxy.** The
   current low-energy diagnostics are sufficient inputs but do not themselves
   produce the boundary-pair gate or the minimum-over-localizer-grid quantity
   defined below.
6. **No persistent scientific result payload in Phase 9.4.** The ledger
   intentionally stores exact inputs, validity, provenance, and outcome
   availability, but not spectra, topology values, robustness results, or
   ranking values. A Phase-9.8-specific, versioned summary manifest is needed;
   it must not be presented as the general Phase-11 dataset schema.

These are implementation prerequisites inside Phase 9.8, not permission to
start Phase 10, Phase 11, mutation, evolution, ML, GNN, active learning, RL, or
generation by a learned model.

## Frozen contract: software and numerical environment

An accepted run would use:

- Python 3.14;
- `PYTHONDONTWRITEBYTECODE=1`;
- the exact committed project revision containing the accepted protocol and
  Phase-9.8 implementation;
- NumPy `Generator(PCG64(seed))` wherever this protocol owns randomness;
- the existing full-spectrum `numpy.linalg.eigh` solver;
- sequential execution in stored request order; and
- no global NumPy random state, hidden retry seed, wall-clock seed, unordered
  set iteration, or parallel reduction.

The dry-run-only seeds are `9_799_900` through `9_799_909`, inclusive. Dry-run
outputs are never eligible candidates, references, validation data, or
confirmation data and must be deleted or stored under an unmistakable
`dry_run` label before the accepted experiment begins.

## Frozen contract: physical model

Every candidate and every reference uses exactly one parameter set:

| Field | Value |
| --- | ---: |
| model | `ChiralPWaveModel` |
| hopping | `1.0` |
| chemical potential | `2.0` |
| pairing | `1.0` |
| chirality | `+1` |
| pairing plane axes | `(0, 1)` |
| reference energy | `0.0` |
| zero-mode tolerance | `1.0e-10` |
| low-energy state count | `16` |
| boundary-localization display threshold | `0.8` |
| numerical tolerance | `1.0e-10` |
| require resolved topology for pipeline validity | `False` |
| require topology convergence for pipeline validity | `False` |
| pipeline-level topology convergence flag | `False` |

Hopping and pairing amplitudes are uniform over retained edges. The hopping
term is `-1.0` in the normal Hamiltonian, the onsite term is `-2.0`, and the
chiral pairing coefficient uses the normalized stored source-to-target edge
direction. No distance decay, edge-specific rescaling, model-parameter search,
or family-specific tuning is allowed.

For the implemented nearest-neighbor square convention, bulk gap closings of
the translation-invariant model occur at special chemical potentials; the
contract deliberately uses an interior value rather than a transition value.
The preparation calibration on open square lattices at sizes 8, 10, and 12
gave agreement of all three implemented real-space methods at Chern sign `+1`
for the values above. This is a software-convention check, not a candidate
outcome or a finite-size proof.

## Frozen contract: primary candidate stratum

Every random-search candidate must satisfy all of these conditions before
model construction:

| Resource or constraint | Frozen value |
| --- | ---: |
| physical sites | exactly `64` |
| stored undirected couplings | exactly `112` |
| embedding | explicit finite coordinates of shape `(64, 2)` |
| physical bounding box | exactly `[0, 7] x [0, 7]` after affine normalization |
| fixed topology cell | exactly `[-0.5, 7.5] x [-0.5, 7.5]` |
| minimum final site separation | at least `0.55` |
| maximum physical edge length | at most `1.75` |
| site degree | between `2` and `4`, inclusive |
| connected components | exactly `1` |
| self-loops or duplicate undirected edges | none |
| straight-edge crossings | none except shared endpoints |
| holes | none in the declared physical boundary model |
| outer-boundary shell thickness | `0.875` |
| allowed outer-boundary site count | `24` through `32`, inclusive |

A site is an outer-boundary site exactly when its minimum Euclidean distance
to one of the four sides of `[0, 7] x [0, 7]` is at most `0.875`. This is a
physical coordinate rule, not a degree inference. There is exactly one
`GeometryBoundaryComponent(kind="outer", component_index=0, ...)` and its
sites equal `geometry.boundary_sites`.

Every retained edge is a straight segment between its endpoint coordinates.
Its explicit displacement is target minus source. Edges are stored in
ascending undirected endpoint order, so the lower site index is the stored
source. The ordered representation is preserved through Hamiltonian building,
serialization, hashing, exact identity, disorder, and plotting.

## Frozen contract: required stochastic candidate generator

The frozen registry key is `hard_core_planar_graph`, version 1, with
`stochastic=True`. It must use only the explicit generator seed supplied by the
existing `GeometryGenerationRequest`.

For each requested generator seed it must:

1. create `Generator(PCG64(seed))`;
2. draw sequential independent two-dimensional uniform proposals inside
   `[0, 7] x [0, 7]`, rejecting a proposal when it is
   less than `0.55` from an accepted point;
3. stop with a generation error after `1_000_000` point proposals rather than
   changing the seed or relaxing a constraint;
4. affinely normalize the accepted point set to the exact bounding box, reject
   the complete attempt if the final minimum separation is below `0.55`, and
   assign site indices by lexicographic `(x, y, acceptance_index)` order;
5. construct the SciPy Delaunay candidate-edge set with fixed Qhull options
   `Qbb Qc Qz Q12`, no `QJ` jitter, lexicographically normalize the extracted
   undirected edge set, reject a Qhull failure or a triangle with absolute
   doubled area at most `1.0e-10`, and discard edges longer than `1.75`;
6. assign one PCG64 raw-word priority to each lexicographically ordered
   candidate edge;
7. build a connected acyclic base by Kruskal order over those priorities;
8. add remaining edges in the same priority order while respecting maximum
   degree 4 until exactly 112 edges are present;
9. reject the complete attempt if the minimum degree, edge count, crossing,
   connectivity, or boundary-count contract is not met; and
10. return the valid geometry or fail after `10_000` complete attempts.

There is no hidden retry seed: every retry continues the one PCG64 stream.
The returned metadata must record proposal count, complete-attempt count,
Delaunay implementation and version, all rejected-attempt reason counts, edge
priority convention, boundary convention, and the common generator
provenance. Same generator version, parameters, environment, and seed must
produce byte-identical ordered geometry serialization.

This is a constrained random planar-graph baseline, not a claim of uniform
sampling over planar graphs and not a generative-AI method.

## Frozen contract: reference families

### Resource-matched primary references

1. **Crystalline:** `square(n_x=8, n_y=8, spacing=1.0,
   boundary_x="open", boundary_y="open")`. It has exactly 64 sites, 112
   edges, a `[0, 7] x [0, 7]` box, degree range 2--4, and 28 declared boundary
   sites.
2. **Amorphous:** 32 `hard_core_planar_reference` point realizations generated
   from the dedicated reference seeds below. Their connectivity differs from
   search candidates only in the frozen edge priority: after the same
   randomized spanning-tree base, remaining admissible edges are ordered by
   `(length, source, target)` instead of random priority. This reference mode
   must have its own registry key and version, not a hidden flag passed only by
   the experiment runner. Its registry key is
   `hard_core_planar_reference`, version 1, with `stochastic=True`.

Both primary references use the same candidate constraints, model, topology
inputs, ranking quantities, and disorder exposures. The crystalline reference
is deterministic and is evaluated once per software revision, not copied 32
times and misrepresented as independent data.

### Descriptive, not resource-matched, references

3. **Quasiperiodic:** `ammann_beenker_patch(radius=4.0, spacing=1.0)` has 57
   sites, 96 edges, 24 boundary sites, and coordinate spans approximately
   `6.828427` in both axes. It remains in these native coordinates; its graph,
   site count, edge count, boundary membership, hopping, and pairing amplitudes
   must not be padded, thinned, or rescaled.
4. **Fractal:** `sierpinski_carpet(order=2, spacing=1.0)` has 64 sites, 88
   edges, coordinate spans 8, and 60 outer-or-hole boundary sites. It remains
   in its native 9-by-9 cell domain. Its hole boundaries remain physical. They
   may not be discarded to make the boundary fraction resemble the primary
   stratum.

The last two references are intentionally unmatched controls. No candidate
advantage over them may be called causal, resource-controlled, or a rejection
of the charter's null hypothesis. A future protocol seeking that claim must
define family-preserving matched constructions and fresh seeds; it may not
retrofit extra edges or change global model parameters after seeing results.

## Frozen contract: seed and role partition

All integer intervals below are inclusive and are part of the protocol:

| Role | Seeds | Use |
| --- | --- | --- |
| dry run | `9_799_900..9_799_909` | implementation checks only |
| search trial | `9_800_000..9_800_031` | one Phase-9.1 sampler seed per trial |
| amorphous reference generation | `9_801_000..9_801_031` | reference geometries only |
| validation disorder | `9_810_000..9_810_063` | frozen selected geometries only |
| final confirmation disorder | `9_820_000..9_820_127` | untouched until the confirmation trigger |

Each search trial requests exactly 32 geometry samples from the one candidate
recipe, for 32 trials and 1,024 requested candidates total. The Phase-9.1
sampler consumes its stream exactly as documented and supplies its derived raw
word as the stochastic generator seed. A Phase-9.1 generation or base-validation
failure returns no partial sample result. It therefore aborts the entire
accepted experiment as a protocol execution failure; it cannot be converted
into a shorter trial, silently replaced, or passed to Phase 9.7. Ordinary
Phase-9.3 invalid evaluations and callback failures still remain false in their
complete 32-candidate trial denominator.

Model parameters are fixed, so there is no parameter-sampler seed. Clean
Hamiltonian construction and exact diagonalization are deterministic, so
`evaluation_seed=None`; a decorative seed must not be recorded as if it drove
the calculation.

Validation and confirmation use common random numbers: a given channel and
seed is applied to every frozen candidate and eligible primary reference. Seed
reuse across different disorder channels does not combine those channels and
is explicitly visible in each transform provenance. No validation or
confirmation result may change search ranking, thresholds, candidate
replacement, or protocol choices.

## Frozen contract: primary-stratum topology inputs

The declared symmetry is two-dimensional class D with particle-hole square
`+1`, no declared time-reversal symmetry, and no declared chiral symmetry.

For a 64-site component-major spinless Nambu basis, basis coordinates are
exactly `np.tile(geometry.coordinates[:, (0, 1)], (2, 1))`. Using
`np.repeat(...)` would be a basis-ordering error.

The experiment-specific topology-input builder must provide:

- position areas from each site's Voronoi cell clipped to the fixed
  `[-0.5, 7.5] x [-0.5, 7.5]` topology cell;
- a bulk mask selecting sites at graph distance at least 2 from the explicit
  physical boundary;
- a second bulk mask at graph distance at least 3 for convergence;
- Bott coordinate-period grids `(7.6, 7.6)`, `(8.0, 8.0)`, and `(8.4, 8.4)`;
- localizer probe position `(3.5, 3.5)`;
- localizer kappa grid `0.1`, `0.2`, and `0.3`; and
- Fermi/localizer energy `0.0`.

Voronoi areas must be positive, sum to 64 within `1.0e-10`, and follow a
documented deterministic clipping and degeneracy policy. A candidate with no
site in either bulk mask is ineligible; the mask may not be relaxed.

Method tolerances are:

| Method | Numerical tolerance | Quantization tolerance |
| --- | ---: | ---: |
| Bott index | `1.0e-10` | `1.0e-6` |
| local Chern marker | `1.0e-10` | `5.0e-3` |
| spectral localizer | `1.0e-10` | not applicable |

The representative stored results use Bott periods `(8.0, 8.0)`, the
distance-2 bulk mask, and localizer kappa `0.2`. `convergence_checked=True` is
allowed only when every value in the corresponding grid is resolved and all
integer indices for that method agree. The full grid results remain in the
Phase-9.8 summary manifest; they are not discarded after setting a boolean.
The topology hook must return already unified `TopologyResult` objects whose
per-method convergence flags come from that method's retained grid. The
pipeline-level configuration remains `False`; it must not stamp one shared
claim onto specialized results before the grids have been checked.

A completed diagnostic grid that is finite but nonquantized, unresolved, or
cross-method inconsistent is a scientific screening failure: the underlying
Phase-7 evaluation remains valid and receives `clean_eligible=False`. An
exception from a topology routine, a malformed result, or a violated API or
numerical contract is instead a Phase-7 topology-stage failure and remains an
explicit invalid evaluation. The two cases must not share one reason code.

### Descriptive-reference topology inputs

The unmatched controls do not enter `R`, candidate ranking, selection, or the
version-1 disorder comparison. They retain family-native topology inputs:

- Ammann--Beenker uses component-major native coordinates, probe `(0, 0)`,
  nominal Bott periods `(8.0, 8.0)` with the same 0.95/1.00/1.05 factors,
  graph-distance-2 and graph-distance-3 masks from its explicit cut boundary,
  and one quarter of every incident complete tile area accumulated at each
  vertex as its local-marker position area. Every selected bulk vertex must
  have positive accumulated area.
- Sierpiński carpet uses component-major native coordinates, probe
  `(4.5, 4.5)`, nominal Bott periods `(9.0, 9.0)` with the same factors, and
  unit position area for each retained cell. Its explicit outer and hole
  boundaries leave no valid distance-2 and distance-3 bulk masks at order 2,
  so the local Chern marker is declared inapplicable and is not fabricated.
  Bott and localizer results are reported separately with their warnings; no
  three-method agreement or clean-eligibility classification is assigned.

These family-native diagnostics are descriptive. Their different physical
domains and method applicability are additional reasons they cannot be used
as resource-matched evidence in version 1.

## Frozen contract: clean eligibility gates

A clean evaluation is screening-eligible only if every gate below passes.
Failures remain explicit and are never repaired or retried with changed
numerics.

### Geometry gate

The complete primary candidate-stratum contract passes, base validation is
valid with `require_connected=True`, and model requirements demand edges,
boundary sites, and spatial axes 0 and 1.

### Topology gate

- Bott, local Chern, and spectral localizer are all applicable through the
  class-D two-dimensional dispatcher.
- Every convergence-grid result is resolved.
- Every result has integer magnitude 1.
- All three methods and all grid values agree on the signed integer.
- No method exceeds its quantization or numerical tolerance.

Topology is not inferred from the graph, embedding, finite spectral gap, or
boundary localization.

### Protection-proxy gate

Define

`localizer_protection_proxy = min(localizer_gap(kappa) for kappa in (0.1, 0.2, 0.3))`.

It must be at least `0.20`. It remains named a finite-size localizer protection
proxy. The Phase-7 `gap` is retained under its existing full finite-spectrum
definition and is neither this proxy nor a bulk, mobility, or topological gap.

### Boundary-signature gate

- Sort the retained states by `(abs(energy), state_index)` and use the first
  eight.
- They must form four particle-hole energy pairs with maximum
  `abs(E_i + E_j) <= 1.0e-8` under minimum-cost deterministic pairing. Enumerate
  every perfect matching of the eight ordered states, minimize the sum of
  `abs(E_i + E_j)`, and break an exact cost tie by the lexicographically sorted
  tuple of state-index pairs.
- At least four of those eight states must have explicit-boundary weight at
  least `0.80`.
- Energies, IPR, site probabilities, boundary weights, and individual-state
  Majorana diagnostics must all be retained.

There is no zero-mode-count requirement and no Majorana-self-conjugacy
threshold.

## Frozen contract: ranking and random-search success

Only valid clean evaluations enter the Phase-9.5 ranker. Ranking is
lexicographic in exactly this order:

1. `clean_eligible`, boolean, maximize;
2. `localizer_protection_proxy`, real, maximize; and
3. `minimum_boundary_weight_first_four`, real, maximize.

`minimum_boundary_weight_first_four` is the minimum explicit-boundary weight
among the first four states in the same `(abs(energy), state_index)` order.
The localizer gap remains available and nonnegative even when its integer
index is unresolved, so scientifically ineligible but numerically completed
candidates need no invented sentinel value.

The final implicit tie break is original batch order, as already defined by
Phase 9.5. There is no weighted scalar score, post-outcome normalization,
complexity bonus, novelty bonus, graph-hash bonus, or visual judgment.

Before the first search trial, evaluate all eligible primary references. Let
`R` be the largest `localizer_protection_proxy` among them. It is an error to
start search if no primary reference passes the clean topology and boundary
gates.

A Phase-9.7 `screening_strong_candidate_v1` is exactly a ranked candidate for
which:

- `clean_eligible is True`; and
- `localizer_protection_proxy >= 1.10 * R`.

This is an engineering screening definition, not scientific success. A search
trial succeeds when at least one of its 32 retained candidates satisfies it.
All 32 decisions are evaluated; invalid candidates and callback failures are
false and remain in the denominator.

Phase 9.7 reports the hit count, hit fraction, plug-in standard error, and a
two-sided 95% Wilson interval over all 32 trials. Candidate success fraction is
descriptive only. No p-value or independence claim is made for candidates.

## Frozen contract: selection and leakage boundary

After all search trials, their candidate ledgers, clean result manifests, and
rankings are sealed. Validation candidates are then selected once:

1. take the rank-1 screening-strong candidate from each successful trial;
2. order those candidates by the same frozen ranking values and then by exact
   geometry ID bytes;
3. retain at most the first eight distinct exact geometry IDs; and
4. write the selected IDs to an immutable selection manifest before using any
   validation seed.

The graph hash may flag possible isomorphism but does not deduplicate physical
snapshots. Validation never replaces a selected candidate. If fewer than eight
trials succeed, validate all available selections. If none succeeds, the
random-search benchmark is a valid negative result and validation does not
run.

## Frozen contract: separate disorder validation

Each selected candidate and each eligible primary reference is evaluated in
six separate ensembles. Every ensemble uses all 64 validation seeds and one
frozen stress level:

| Channel | Transform and stress level |
| --- | --- |
| onsite | uniform onsite width `1.0` |
| hopping | uniform hopping width `0.5` |
| pairing | chiral-p-wave pairing width `0.5` |
| coordinate | uniform coordinate width `0.20` |
| edge removal | independent removal probability `0.05` |
| node removal | independent removal probability `0.02` |

The channels are never composed. A realization succeeds only if its evaluation
passes its channel-appropriate geometry applicability plus the same topology,
protection-proxy, and boundary-signature gates. Matrix-disorder geometries must
still satisfy the full clean geometry gate. Coordinate disorder uses the fixed
topology cell and must retain connectedness, the explicit boundary, minimum
site separation `0.45`, maximum edge length `1.95`, and non-crossing straight
edges. Edge- and node-removal sources must satisfy the clean 64/112 contract,
but their transformed geometries are not required to retain the original site
or edge count; they must remain nonempty, connected, coordinate-bearing,
non-crossing, and equipped with a nonempty remapped explicit boundary. Empty
bulk masks or any other failed downstream gate remain unsuccessful.

Operational realization or evaluation failures and invalid transformed
candidates are false in the denominator and remain separately counted.

For one channel, call a finite-size candidate validated only when both:

- at least 52 of 64 realizations succeed; and
- the two-sided 95% Wilson lower bound is at least `0.70`.

The candidate passes the version-1 validation screen only if all six separate
channels pass. No average, minimum score, or combined disorder ensemble is
substituted for this conjunction.

For each channel, report the candidate fraction beside the strongest eligible
primary-reference fraction under the same seeds. A difference is descriptive;
version 1 declares no causal superiority test because it has only one system
size and incomplete matching to quasiperiodic and fractal controls.

## Frozen contract: final confirmation trigger

The 128 confirmation seeds remain untouched unless at least one selected
candidate passes all six validation channels. If triggered, confirm every such
candidate and every eligible primary reference without reranking or changing
the selected set.

At confirmation, the per-realization gates and six stress levels are unchanged.
A channel confirms only when both:

- at least 104 of 128 realizations succeed; and
- the two-sided 95% Wilson lower bound is at least `0.73`.

Version 1 has no cross-size candidate-family construction. Confirmation is
therefore still a finite-64-site result and cannot support a thermodynamic or
finite-size-scaling claim. Phase-8.12 and Phase-8.13 must not be invoked with
fabricated larger versions of an isolated random graph.

## Frozen contract: persistence and audit artifacts

Every search trial is persisted immediately after Phase-9.3 execution with the
existing Phase-9.4 archive, including invalid and failed candidates. Existing
files are never overwritten. A versioned Phase-9.8 summary manifest must retain
at least:

- protocol identifier, accepted protocol commit, code commit, Python, NumPy,
  SciPy, platform, and solver versions;
- all master seeds, derived generator seeds, ordered candidate indices,
  generator provenance, exact geometry IDs, graph fingerprints, and parameter
  IDs;
- all clean eligibility decisions and individual gate reasons;
- every topology convergence-grid scalar and warning;
- the eight boundary-state records used by the gate;
- all ranking values and ranks in batch order and rank order;
- reference values and the resulting numeric `R`;
- Phase-9.7 outcomes and uncertainty;
- the immutable validation selection manifest;
- every per-seed disorder decision, operational failure, Wilson interval, and
  Phase-8 provenance record; and
- the confirmation trigger decision and, if triggered, all confirmation
  records.

The manifest format is experiment-specific and versioned under Phase 9.8. It
does not claim to be the Phase-11 dataset format and does not add a generic
dataset writer. Numerical result arrays not supported by Phase-9 persistence
must either be stored in an explicitly versioned Phase-9.8 artifact or be
recomputable byte-for-byte from retained inputs; silent loss is not allowed.

Visualization may use Phase 9.6 only after ranking and must be generated from
the sealed result. It cannot alter eligibility, ranks, selections, or captions.

## Calibration record used to prepare this decision

The following known-reference checks were run with Python 3.14 and no bytecode
writes. They were not random-search trials:

- open square sizes 8, 10, and 12;
- `hopping=1`, `chemical_potential=2`, `pairing=1`, `chirality=+1`;
- component-major Nambu coordinates constructed with `np.tile`;
- Bott periods at 0.95, 1.00, and 1.05 times the nominal period;
- local-Chern bulk depths 2 and 3; and
- localizer kappa values 0.1, 0.2, and 0.3.

All three methods returned signed integer `+1` at all three sizes and all
tested convergence values. At size 8, the minimum localizer gap over the kappa
grid was approximately `0.231474`, the local-Chern estimates at bulk depths 2
and 3 were approximately `0.995287` and `0.997892`, and the four closest
particle-hole-related edge states all had boundary weight above `0.99`.
Individual-state Majorana self-conjugacy was zero in the solver eigenbasis,
which is why it is retained but not thresholded.

These numbers justify internal consistency only. They do not establish that a
random candidate, amorphous reference, quasiperiodic patch, or fractal will
pass.

The ten reserved dry-run seeds `9_799_900..9_799_909` were also passed through
a file-free prototype of the frozen geometric construction. All ten reached
64 sites, 112 edges, degree range 2--4, and the boundary-count constraint. The
largest observed construction cost was 24 complete attempts and 3,734 point
proposals. No model, Hamiltonian, topology method, observable, ranking,
reference comparison, or disorder transform was run. These dry-run seeds have
no scientific role and may be reused for implementation regression tests.

## Scientific basis and interpretation

Real-space invariants are required because the primary candidates lack
translation invariance. The Bott approach was developed as a real-space
topological obstruction without requiring a Brillouin-zone flux torus
([Hastings and Loring, 2010](https://doi.org/10.1063/1.3274817)). The local
Chern marker maps Chern order in coordinate space and is applicable to open
and inhomogeneous systems
([Bianco and Resta, 2011](https://doi.org/10.1103/PhysRevB.84.241106)). The
spectral-localizer half-signature supplies a finite-dimensional local index,
but its nonzero localizer gap and kappa dependence remain part of the method's
assumptions
([Loring and Schulz-Baldes](https://arxiv.org/abs/1802.04517)). Agreement among
these implementations is a stronger screen than any one method, but it is not
a substitute for scaling or method assumptions.

Random spatial systems can support topology without crystalline order, but
they can also be spectrally gapless while protected by a mobility gap. The
Shiba-glass study is therefore direct motivation not to relabel a finite
spectral separation as a bulk protection gap
([Pöyhönen et al., 2018](https://doi.org/10.1038/s41467-018-04532-x)). Random
point-set topological systems likewise motivate an explicit amorphous control
rather than treating every random geometry as a novel family
([Agarwala and Shenoy, 2018](https://doi.org/10.1038/s41567-017-0024-5)).

The accepted research charter's quasiperiodic and fractal literature boundary
remains incorporated by reference. Version 1's unmatched descriptive controls
must not be used to claim that those named families were fairly beaten.

## Acceptance record

The user explicitly accepted all of the following freeze decisions on
2026-09-01 before any search-role candidate, amorphous reference, validation
seed, or confirmation seed was evaluated:

1. the finite-size engineering claim boundary rather than a discovery claim;
2. the frozen `hard_core_planar_graph` candidate and amorphous-reference
   generators;
3. the exact 64-site/112-edge spatial and degree constraints;
4. the fixed chiral-p-wave parameter point;
5. the three-method convergence and boundary gates;
6. the localizer-gap proxy name and threshold;
7. the 32-by-32 search budget and all seed partitions;
8. the lexicographic ranking and 10% reference-relative screen;
9. the six separate validation channels, stress levels, and Wilson rules;
10. the explicitly unmatched status of the Ammann--Beenker and Sierpiński
    references; and
11. the Phase-9.8-specific summary artifact as a non-Phase-11 persistence
    format.

This acceptance does not waive any prerequisite, claim boundary, failure rule,
or amendment rule above. Until this accepted file has been committed as its
own protocol revision, Phase 9.8 must not be implemented or executed.
