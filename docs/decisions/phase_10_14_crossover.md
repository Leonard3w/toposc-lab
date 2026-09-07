# Phase 10.14 — Crossover

## Status and scope

Accepted as a deliberately restricted Phase-10.14 crossover architecture.
Only fixed-scaffold edge-locus crossover is defined. A general crossover
between arbitrary geometry graphs remains scientifically undefined and is not
implemented.

This phase adds no parent pairing, crossover probability, retry or repair,
population construction, generation loop, diversity or novelty mechanism,
fitness evaluation, checkpoint, machine-learning behavior, or dataset
functionality.

## Research decision

Graph crossover is not representation-neutral. Exchanging arbitrary subgraphs
requires a correspondence between parent sites and a rule for reconnecting cut
boundaries. In Toposc-Lab those choices could silently change physical
positions, edge orientation, coupling distance, boundary meaning, plaquettes,
resource budgets, or the physical problem itself.

Prior network-design work makes crossover meaningful by fixing the node set
and encoding possible links as common loci, or by adding domain-specific
partition and connectivity rules. Examples include fixed-node link-topology
encoding in [Altiparmak et al.](https://eng.auburn.edu/~aesmith/files/ierc97org.pdf),
adjacency-based connected-network search in
[Lam and Szeto](https://arxiv.org/abs/1412.4218), and explicit domain linkage in
[Stonedahl, Rand, and Wilensky](https://doi.org/10.1145/1389095.1389290).
Graph-based structural optimization likewise requires a problem-specific
connected representation and partition operator
([Tai and Chee](https://doi.org/10.1016/j.compstruc.2004.05.005)). These methods
support constrained, domain-defined recombination; they do not justify a
universal crossover between unrelated physical graphs.

## Fixed-scaffold compatibility

`validate_fixed_scaffold_crossover_parents(...)` requires both parents to be
valid `GeometryGenome` snapshots with explicit coordinates. It does not align,
relabel, rotate, reflect, scale, interpolate, or otherwise infer a site map.

The following non-edge genes must match exactly:

- site count and coordinate rows, including float64 bit patterns and signed
  zero;
- embedding dimension;
- boundary sites and ordered boundary components;
- site types;
- sourced dimension records;
- rooted-tree structure;
- geometry metadata, including mapping order, typed values, array dtypes,
  shapes, and bytes; and
- ordered faces and their metadata.

The strict equality means site index `i` already denotes the same declared
physical site in both parents. Coordinate-free abstract graphs are rejected
because the present API has no separately justified site-correspondence
record. Differently sized or differently embedded graphs are also rejected.
Parents must differ at least at one edge locus; exact no-op recombination is
not reported as crossover.

## Edge-locus and inheritance contract

An edge locus is the sorted endpoint pair of the simple undirected graph. Its
allele in one parent is either absence or the parent's entire `GeometryEdge`
record. The record is never reduced to a Boolean: source-to-target orientation,
edge type, boundary-crossing flag, displacement, and typed metadata remain
inseparable.

`fixed_scaffold_edge_crossover(...)` orders differing loci lexicographically
and uses one local `numpy.random.PCG64` generator with an explicit nonnegative
seed. One binary decision is drawn per differing locus. The first candidate
receives the chosen parent allele and the second candidate receives the
complementary allele. Exact shared alleles pass to both without a random draw.
Candidate edge sequences use lexicographic locus order, removing dependence on
either parent's stored edge order without changing orientation.

Every `EdgeLocusDecision` records the locus and the first candidate's source;
the second source is its complement. `GeometryCrossoverResult` retains both
exact parents, seed, RNG identifier, version, policy, decisions, candidates,
validity reports, and limitations. Its constructor replays the versioned
operator and rejects altered decisions, candidates, or reports.

## Validity and resource boundary

Both candidates must materialize as valid production geometries. Each then
receives a Phase-10.9 `MutationValidityReport` under the caller's explicit
policy. Policy-invalid candidates remain in the result with their issues. The
operator does not retry, repair, reconnect, replace, discard, or evaluate them.

Complementary inheritance does not guarantee that either candidate preserves
parent edge count, connectedness, degree distribution, planarity, edge-length
budget, or physical performance. A frozen experiment must express its
model-independent admissibility bounds through `MutationValidityPolicy` and
retain all downstream model, topology, robustness, fairness, and evidence
checks. Exact resource preservation is not inferred from the word crossover.

## Explicitly undefined crossover cases

Phase 10.14 does not define:

- subgraph exchange between unrelated site sets;
- graph matching or canonical-hash-based site alignment;
- crossover between different site counts or coordinate embeddings;
- coordinate, boundary, face, dimension, rooted-tree, or metadata crossover;
- automatic alignment under translation, rotation, reflection, or relabeling;
- a variable-length genome operator; or
- topology-, fitness-, novelty-, or model-aware inheritance.

Those operations would require separately justified physical correspondence,
cut/reconnection semantics, resource controls, and benchmarks. The canonical
graph hash cannot provide that correspondence because it is only an
isomorphism-candidate fingerprint and intentionally ignores several physical
fields.

## Scientific boundary

A crossover candidate is an engineering proposal, not evidence. It does not
satisfy the research charter's topology, spectral-protection, boundary,
robustness, reference-matching, finite-size, or leakage gates and cannot by
itself support a scientific discovery claim.
