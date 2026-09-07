# Phase 10.9 — Mutation validity checks

## Status and scope

Accepted as the Phase-10.9 architecture for validating a proposed transition
from one `GeometryGenome` to another.

This phase adds no mutation sampling, retry loop, automatic repair, population,
fitness, selection, crossover, generation loop, topology calculation,
machine-learning behavior, or dataset functionality.

## Validation layers

Mutation validity is distinct from the existing layers:

1. genome validation determines whether stored genes can materialize a
   `Geometry`;
2. production geometry validation checks model-independent representation and
   graph integrity;
3. mutation validity compares a valid source with a valid result and applies an
   explicitly supplied search-space policy; and
4. model compatibility, topology, scientific evidence, and experiment success
   remain separate downstream decisions.

The validator reports invalidity without changing either genome. API type
errors still raise immediately. Callers that intentionally want exception
control flow may invoke `MutationValidityReport.raise_for_errors()`.

## Mandatory transition rule

Source and candidate must both pass the established genome-to-geometry
boundary. The candidate must retain the source embedding dimension. This rule
is mandatory rather than configurable because Phase 10.8 rejected
dimension-changing mutation for the current search program.

The validator does not require the candidate to differ from the source.
Mutation primitives already define their own change semantics; validity is
about admissibility, not mutation provenance or novelty.

## Explicit policy

`MutationValidityPolicy` contains only model-independent constraints. Every
optional search-space constraint is disabled by default, so the default adds
no hidden scientific assumptions beyond base validity and fixed dimension.

The policy can declare:

- connectedness;
- minimum and maximum site, edge, and boundary-site counts;
- minimum and maximum site degree;
- required coordinates and embedding dimension;
- per-axis inclusive coordinate bounds;
- minimum Euclidean site separation;
- maximum physical edge length; and
- absence of non-adjacent straight-edge crossings in a two-dimensional drawing.

Equal minimum and maximum count bounds express an exact resource budget.
Numerical spatial comparisons use one explicit positive tolerance.

Maximum edge length uses `Geometry.distance`, so an explicit oriented edge
displacement takes precedence over endpoint-coordinate subtraction. Site
separation and straight-edge crossings use the coordinate table. Crossing
checks are therefore available only for explicit two-dimensional coordinates;
requesting them elsewhere produces a validity issue rather than silently
projecting the geometry.

## Structured result

`validate_geometry_mutation(source, candidate, policy=...)` returns a
`MutationValidityReport` containing ordered, stable `MutationValidityIssue`
records and immutable measurements. Cheap structural measurements are always
recorded. Potentially quadratic spatial and crossing measurements are computed
only when their corresponding policy constraint is requested.

The report does not collapse distinct failures. For example, an unavailable
edge length is separate from an excessive edge length, and an undefined
crossing check is separate from a detected crossing. This preserves enough
information for later failure accounting without defining a population or
fitness policy early.

## Scientific boundary

These checks establish admissibility only under the supplied geometric policy.
They do not establish physical dimension, Hamiltonian compatibility,
topological character, robustness, fabrication feasibility, resource fairness,
or scientific success. A frozen experiment remains responsible for choosing
its bounds and for applying all downstream evidence gates.
