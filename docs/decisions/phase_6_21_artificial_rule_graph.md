# Phase 6.21: Declarative artificial rule geometries

## Status

Accepted.

## Context

Phase 6.21 needs a model-independent way to generate unnamed structures from
procedural rules. Accepting executable callbacks or expression strings would
make generator requests unsafe, difficult to serialize, and impossible to
compare reliably. A finite growth process also needs explicit limits and an
unambiguous meaning for its outer boundary.

## Decision

`artificial_rule_graph` accepts a finite, JSON-compatible list of integer
displacement vectors. Growth starts at the origin and applies the normalized
rules breadth-first to every newly reached site. `iterations` is therefore the
maximum graph distance from the origin under the supplied rules.

Rules are required to be nonzero, unique, and inverse closed. This matches the
undirected `GeometryEdge` contract and prevents a directed input language from
being silently reinterpreted. Rules are sorted lexicographically, sites are
ordered by `(rule distance, integer coordinate)`, and opposite rules share one
stable edge family. Integer coordinates are retained internally during growth;
`spacing` is applied only when the public floating-point coordinates and edge
displacements are built.

The selected graph is induced by the supplied rules: the final shell is also
processed for connections to already selected sites. Sites at exactly the
requested rule distance form one explicit procedural `outer` boundary
component. This does not claim a geometric or topological boundary inferred
from an embedding.

Independent budgets limit selected sites, stored undirected edges, and rule
applications. Each budget can be disabled explicitly with `None`.

## Consequences

- Requests are safe to persist and pass through the common generator registry.
- Output is deterministic and independent of input rule ordering.
- Arbitrary finite embedding dimensions are supported without inventing
  intrinsic-dimension, face, site-type, or physics metadata.
- More expressive stateful grammars require a future versioned rule language;
  they are not smuggled into this API as executable Python.
