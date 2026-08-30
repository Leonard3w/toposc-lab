# Phase 6.17 decision: silver-mean bond-tiling chain

## Geometry and convention

The second controlled quasiperiodic geometry is the open one-dimensional
silver-mean chain, also called the octonacci chain. It starts from `S` and
simultaneously applies `L -> LSL` and `S -> L`. This convention makes every
finite substitution word palindromic. Symbols are geometric bonds, typed
`silver_mean_long` and `silver_mean_short`, with explicit displacement.

The short length is `spacing`; the long length is
`long_short_ratio * spacing`. The default ratio is the silver mean
`1 + sqrt(2)`. Because its square equals twice itself plus one, both substituted
tiles have exactly the length obtained by inflating their predecessor by the
silver mean. A ratio greater than one may be supplied for controlled symbolic
experiments, but metadata then marks the geometry as non-silver.

The substitution, seed, interval lengths, Pell-type length recurrence, and
reflection symmetry follow:

- https://doi.org/10.1103/PhysRevB.62.15569
- https://doi.org/10.1103/PhysRevB.72.054203

## Shared substitution architecture

Fibonacci and silver-mean construction now use a private immutable binary
substitution specification and one shared engine. The engine validates order,
lengths, and budget; advances symbol counts before allocation; builds the word;
and creates coordinates and typed edges. Family-specific public functions own
their scientific defaults and metadata. This avoids duplicated numerical and
resource-control logic without prematurely exposing a generic public API.

## Separation from physical models

The geometry contains only positions, boundary sites, geometric bond types,
and lengths. It assigns no hopping, onsite potential, pairing, or other model
parameter. The embedding dimension is one and an exact topological-dimension
record of one describes the infinite family. No translation-lattice dimension
is claimed.

## Resources

The exact symbol-count recurrence is evaluated and checked against `max_sites`
before the substitution word is materialized. The default limit is 100,000
sites. The word itself is not duplicated in metadata because canonical edge
order and edge types already preserve it exactly.
