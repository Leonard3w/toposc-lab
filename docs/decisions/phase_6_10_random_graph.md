# Phase 6.10 decision: seeded Erdős–Rényi random graph

## Ensemble

The generic random-graph generator implements the undirected simple
Erdős–Rényi `G(n, p)` ensemble. Every unordered pair of distinct sites is
sampled independently with probability `p`. The generator does not retry until
the graph is connected because conditioning on connectivity would define a
different distribution.

Abstract random graphs receive no invented spatial coordinates, boundary sites,
site types, or dimension records. Models and analyses must not infer physical
dimension from graph size or connectivity.

## Reproducibility

Generation requires an explicit nonnegative seed. Candidate pairs are visited
in lexicographic `(source, target)` order and sampled from an explicitly named
NumPy PCG64 stream. The common generator registry records the seed separately
from ordinary parameters, while direct generator metadata also retains it.

## Resource guard

`G(n, p)` has `n * (n - 1) / 2` candidate edges regardless of the realized edge
count. This number is checked before random sampling. The default
`max_candidate_edges` is 1,000,000; callers may deliberately raise it or pass
`None`. Recording the chosen budget makes resource policy reproducible.
