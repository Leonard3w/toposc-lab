# Pre-Phase 9 research charter

## Status and purpose

Accepted on 2026-09-01 as the version-1 scientific direction for the first
Toposc-Lab discovery program.

This charter fixes the research question, evidence boundaries, and anti-bias rules
before automated search begins. It is not Phase 9.1 and adds no sampler, ranking,
dataset, or search behavior. Exact numerical model parameters, resource strata,
thresholds, sample sizes, and seed lists must be frozen in a separate versioned
experiment protocol before the reproducible Phase-9.8 random-search experiment.
They may not be chosen after inspecting search outcomes.

## Central research question

> Can constrained inverse connectivity design discover previously unnamed,
> nonperiodic, spatially embedded planar graph families that support more robust
> two-dimensional class-D topological superconductivity and chiral Majorana boundary
> signatures than resource-matched crystalline, amorphous, quasiperiodic, and fractal
> reference families under the same physical model and disorder protocols?

The explanatory follow-up question is:

> Which local graph motifs, mesoscopic structures, and global connectivity properties
> cause any independently validated robustness advantage?

The initial program studies a spinless chiral `p_x + i p_y` BdG model on a declared
physical two-dimensional plane. It is a controlled model study, not a material-specific
prediction. Arbitrary abstract graphs without compatible coordinates are outside this
initial physical scope.

Two-dimensional chiral class-D superconductors generally support chiral Majorana
boundary modes. They do not automatically provide the pair of spatially separated
Majorana zero modes associated with an open Kitaev chain. Claims about isolated
Majorana zero modes require a separately declared and validated vortex, defect,
domain-wall, corner, or other localization mechanism and are not part of the initial
Phase-9 question.

## Hypotheses

The null hypothesis is:

> After controlling physical model, site count, edge or coupling budget, spatial
> extent, boundary definition, and disorder exposure, searched artificial geometries
> have no reproducible topological-robustness advantage over the strongest eligible
> named reference family.

The working hypothesis is:

> One or more connectivity motifs or mesoscopic structures produce a robustness
> advantage that cannot be explained solely by additional couplings, higher degree,
> larger boundary fraction, altered spatial scale, or favorable parameter tuning.

A negative result is scientifically valid. Search efficiency, a visually unusual
graph, or a high unvalidated score does not reject the null hypothesis.

## Initial candidate-space contract

Every candidate in the initial discovery experiment must be a connected, spatially
embedded graph with a compatible physical two-dimensional plane and explicit oriented
edges. The experiment protocol must define and enforce at least:

- allowed site-count strata;
- allowed edge or total-coupling budgets;
- minimum site separation and spatial bounds;
- maximum physical coupling range;
- allowed degree range;
- connectivity and any required planarity or edge-crossing policy;
- the coordinate, hopping, and pairing conventions shared by all candidates;
- an explicit physical boundary policy rather than a boundary inferred from graph
  degree;
- fabrication constraints if a later experiment makes a fabrication claim.

Long edges, high-degree hubs, duplicate couplings, disconnected components, or changes
of physical scale may not provide unrecorded extra resources. The physically relevant
orientation of every `GeometryEdge` must be retained. Geometry, model parameters, and
disorder remain separate search dimensions even if a later experiment combines them.

The initial search must not silently include abstract `random_graph` geometries in the
chiral model. Coordinate-cutoff, nearest-neighbor, rule-based, or mutated graphs are
eligible only when their construction and physical constraints satisfy the frozen
candidate-space contract.

## Reference and fairness contract

The frozen experiment protocol must include named reference families drawn, where
compatible, from regular crystalline, amorphous, quasiperiodic, and fractal geometries.
Candidate and reference comparisons must be stratified or matched by the resources
that can trivially affect performance, including at least:

- number of physical sites;
- number or total strength of couplings;
- spatial extent and density;
- degree constraints;
- boundary-to-bulk exposure;
- physical model and parameter policy;
- disorder definition, strength, and requested seed count;
- solver and evaluation settings.

The primary comparison is against the strongest eligible reference under the frozen
protocol, not only against an average or deliberately weak baseline. Unmatched resource
differences must be reported and prevent a causal geometry claim.

## Evidence gates and outcomes

No single scalar establishes scientific success. Phase 9 may rank candidates for
engineering purposes, but a candidate enters scientific validation only if it passes
all predeclared eligibility gates.

### Topology gate

Topology must be resolved by methods that are explicitly applicable to the declared
class-D model and physical dimension. Applicable real-space methods such as the Bott
index, local Chern marker, and spectral localizer retain their individual assumptions,
convergence status, and warnings. The final experiment protocol must predeclare the
required methods and agreement policy. Graph structure, embedding dimension, a spectral
gap, or a boundary-localized state alone must never be treated as topology.

### Spectral-protection gate

The existing Phase-7 `gap` is the full finite-spectrum separation across the reference
energy. Chiral boundary states can make it small, so it is not automatically a bulk or
mobility gap and must not be renamed a `topological gap`. Before Phase 9.8, the
experiment protocol must define and validate the protection quantity used for ranking,
possibly including a declared bulk estimator or spectral-localizer gap. Proxies must
remain labeled as proxies.

### Boundary-state gate

Boundary evidence must use explicitly declared physical boundary sites and retain
energy, localization, particle-hole/Majorana diagnostics, and finite-size limitations.
It supports a chiral Majorana boundary-signature claim only together with the topology
gate. Accidental zero energy, high boundary weight, or high Majorana self-conjugacy in
isolation is insufficient.

### Robustness gate

Robustness uses Phase-8 disorder contracts and an explicit success predicate that
includes the required topology, protection, and boundary evidence. Onsite, hopping,
pairing, coordinate, edge-removal, and node-removal channels remain separate scientific
experiments unless a combination rule is frozen in advance. Operational failures stay
in the denominator and remain distinguishable from scientific failures. Per-size
fractions require uncertainty estimates; thermodynamic claims require justified
cross-size families and scaling.

### Multi-objective boundary

Raw topology, protection, boundary, Majorana, robustness, complexity, and resource
quantities remain separate. Prefer a predeclared Pareto or eligibility-plus-ranking
policy. Any normalization, weighting, lexicographic order, or scalar aggregation must be
fixed before outcomes are inspected and must be accompanied by sensitivity analysis.

## Search, validation, and leakage policy

The experiment protocol must assign disjoint, explicit seeds and candidate roles before
execution:

- generation and search seeds may guide Phase-9 candidate discovery and ranking;
- validation disorder seeds may be used only after candidate selection is frozen;
- final confirmation seeds and larger system sizes remain untouched until the reported
  validation policy calls for them.

Every evaluated search candidate, including invalid and unsuccessful candidates, must
be retained when Phase-9 persistence exists. Validation outcomes may not trigger
threshold tuning, candidate replacement, or repeated selection on the same validation
set. Any exploratory change after viewing outcomes creates a new versioned experiment
and requires fresh untouched validation seeds.

Model-parameter optimization must be separated from geometry advantage. A geometry may
not receive a broader or more favorable parameter search than its references. A later
joint search must use nested or otherwise leakage-safe validation capable of separating
geometry effects from parameter tuning.

## Identity, novelty, and priority boundary

The Phase-6 graph hash remains an isomorphism-candidate fingerprint, not a physical
deduplication identity. The Phase-7.11 geometry ID remains an exact,
representation-dependent snapshot identifier, not a canonical physical identity.
Novelty analysis must additionally consider exact graph isomorphism where applicable,
coordinate symmetries, relabeling, orientation conventions, known construction rules,
and physically equivalent deformations.

`Previously unnamed` is not itself a scientific result. Before claiming literature
priority, the project requires a documented systematic search of primary literature and
appropriate expert review. The targeted literature check that motivated this charter
found adjacent work on amorphous, quasiperiodic, fractal, and ML-assisted superconducting
systems, but it is not proof that no equivalent study exists.

The preliminary search boundary recorded on 2026-09-01 includes at least these adjacent
primary works:

- [Amorphous topological superconductivity in a Shiba glass](https://doi.org/10.1038/s41467-018-04532-x);
- [Topological superconductivity in Fibonacci quasicrystals](https://doi.org/10.1103/PhysRevB.110.134508);
- [Fractal Topology of Majorana Bound States in Superconducting Quasicrystals](https://arxiv.org/abs/2602.02796);
- [Fractal hierarchy enables exponential scaling of topological boundary states](https://doi.org/10.1038/s41467-026-75412-y);
- [Developing a complete AI-accelerated workflow for superconductor discovery](https://doi.org/10.1038/s41524-026-01964-8).

These establish that the neighboring ingredients are active research areas. They do not
by themselves establish or eliminate priority for the constrained graph-connectivity
discovery program defined here.

## From candidate discovery to a scientific rule

A top-ranked graph is a candidate discovery, not yet a physics discovery. A proposed
geometric mechanism requires, where possible:

1. independent seed and system-size confirmation;
2. comparison with the strongest resource-matched references;
3. control for simple confounders such as degree, density, edge count, and boundary
   fraction;
4. motif removal or edge/node ablation that weakens the effect;
5. motif transplantation into unseen geometries that reproduces the effect;
6. validation on newly generated, unseen families;
7. cross-model testing before claiming a general geometric principle.

The intended final result is therefore not `graph G has the highest score`, but a
falsifiable statement of the form:

> Under the frozen physical and resource controls, motif or structure X produces a
> reproducible improvement in class-D topological robustness relative to reference Y,
> survives independent disorder and finite-size validation, and loses that improvement
> under a predeclared ablation.

## Freeze points and amendments

This version freezes the research direction and scientific boundaries, not the numeric
Phase-9.8 experiment. Phase 9.1 through Phase 9.7 must remain general infrastructure
steps and may not weaken this charter silently. Before Phase 9.8, a separate experiment
decision must freeze all numeric choices, exact reference families, topology hooks,
success predicates, search budget, seed partitions, ranking policy, and statistical
decision rules.

Any later amendment must state what changed, why it changed, whether outcomes had
already been inspected, and which new untouched validation data will be used. The
original version remains in Git history.
