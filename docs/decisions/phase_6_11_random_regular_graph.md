# Phase 6.11 decision: uniform simple random-regular graph

## Ensemble and validity

The generator samples undirected simple `d`-regular graphs. Inputs must satisfy
`0 <= d < n` and `n * d` must be even. These conditions are checked before
allocation or sampling. Connectivity is not imposed because disconnected
regular graphs are valid members of the requested ensemble.

## Uniform sampler

The sampler creates `d` labeled stubs per site, uniformly shuffles all stubs,
pairs consecutive stubs, and rejects the complete attempt if it contains a
self-edge or repeated edge. Every simple `d`-regular graph corresponds to the
same number of labeled-stub pairings. Conditioning this configuration model on
simplicity therefore produces a uniform distribution over simple regular
graphs.

A finite `max_attempts` bound prevents unbounded rejection loops. Exhaustion is
reported as an error; the generator does not return a biased repair or fallback
graph.

## Dense degrees

If `n - 1 - d < d`, the generator uniformly samples the sparse
`(n - 1 - d)`-regular graph and returns its complement. Graph complementation is
a bijection between these two regular ensembles, so this optimization preserves
uniformity and greatly improves acceptance for degrees near `n - 1`.

## Reproducibility and resources

Sampling uses an explicit nonnegative seed and a named NumPy PCG64 stream.
Canonical edges are sorted before `Geometry` construction. Metadata records the
effective sampled degree, whether complementation occurred, and the attempts
used. The exact target edge count `n * d / 2` is checked against `max_edges`
before sampling.

As an abstract network, the result has no coordinates, boundary sites, site
types, or dimension records.
