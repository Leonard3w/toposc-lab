# Phase 6.13 decision: Barabási–Albert scale-free graph

## Model and initial condition

The scale-free generator implements Barabási–Albert preferential attachment.
For `m = attachments_per_site`, generation starts from the complete graph
`K_(m+1)`. Every subsequent site creates exactly `m` edges to distinct existing
sites. The construction is connected by design; connectivity is not imposed by
rejection or postprocessing.

Initial clique edges are typed `initial_clique`, while growth edges are typed
`preferential_attachment`.

## Weighted sampling without replacement

An attachment index stores each existing site once per current incident edge.
Drawing uniformly from this index is therefore exactly degree-proportional.
Already selected targets are skipped, which is equivalent to sequentially
renormalizing the remaining degree weights after each selection. After adding a
new site, selected targets receive one additional index entry and the new site
receives `m` entries.

This representation avoids rebuilding and scanning the complete degree vector
for every new site. Candidate draws and site growth use stable order with an
explicit NumPy PCG64 seed.

## Invariants and resources

Inputs require `1 <= m < n`. The exact edge count is
`m * n - m * (m + 1) / 2`; it is checked against `max_edges` before allocation.
Canonical final edges are sorted, and the model, initial graph, selection rule,
seed, RNG, and budget are retained in immutable metadata.

The result is an abstract network without coordinates, boundary sites, site
types, or dimension records.
