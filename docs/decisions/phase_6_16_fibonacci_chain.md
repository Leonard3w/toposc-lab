# Phase 6.16 decision: Fibonacci bond-tiling chain

## Geometry and substitution

The first controlled quasiperiodic geometry is an open one-dimensional
Fibonacci tiling. Starting from `L`, each generation simultaneously applies
`L -> LS` and `S -> L`. Every symbol is one geometric bond and therefore adds
one new site. Long and short bonds are explicitly typed `fibonacci_long` and
`fibonacci_short` and carry their physical displacement.

The short length is `spacing`; the long length is
`long_short_ratio * spacing`. The default ratio is the golden ratio
`(1 + sqrt(5)) / 2`, for which substitution is an exact geometric inflation.
Other ratios greater than one retain the symbolic Fibonacci order but are
marked as nongolden geometry in metadata.

This convention follows the standard Fibonacci tiling described by the
substitution and golden interval ratio in:

- https://doi.org/10.1107/S2053273318015528
- https://doi.org/10.1038/s41467-018-05950-7

## Separation from physical models

The geometry contains positions, boundary sites, bond types, and bond lengths.
It does not assign hopping strengths, onsite potentials, pairing amplitudes, or
any superconducting parameters. Models may map the two bond types to physical
couplings later.

The embedding dimension is 1. A separate exact topological-dimension record of
1 describes the infinite tiling family; no translation-lattice dimension is
claimed because the Fibonacci tiling is not periodic.

## Determinism and resources

Substitution order zero is one long bond and two sites. Bond counts are advanced
before materializing the word and checked against `max_sites` at every step.
The default limit is 100,000, so oversized orders fail during the small integer
recurrence rather than after allocating an exponentially long string. The word
is not duplicated in metadata: canonical edge order and edge types already
encode it exactly.
