# Phase 7.6: Geometry descriptor extraction

## Status

Accepted.

## Context

Phase 7 requires scalar graph properties that can later support comparison, scoring, and
search. These quantities belong to the model-independent geometry layer and must not
silently include physical edge orientation, pairing conventions, Hamiltonian terms, or
topological conclusions.

Disconnected geometries require explicit shortest-path semantics because ordinary graph
diameter and all-pairs mean distance are undefined when unreachable pairs exist.

## Decision

`extract_geometry_descriptors` returns an immutable scalar mapping for the undirected
one-skeleton. It includes site and edge counts, degree mean and population variance,
component count and largest component size, connectedness, cyclomatic rank and cycle
presence, triangle count, mean local clustering, connected-pair count, reachable-pair
fraction, and mean/maximum finite shortest-path length.

Shortest-path statistics use distinct reachable unordered site pairs only.
`reachable_pair_fraction` reports their share of all unordered pairs. If no distinct pair
is reachable, mean and maximum are `None`. A one-site graph has reachable-pair fraction
one by the vacuous connectedness convention.

`evaluate_geometry_descriptors` copies the mapping into `GeometryEvaluation` and records
scope warnings. Stored edge orientation, coordinates, boundary labels, metadata, site and
edge types, faces, and physical model terms do not affect the descriptors. No external
graph dependency is introduced.

## Consequences

- Graph structure and physical attribution remain separate.
- Disconnected candidates retain meaningful finite-distance statistics without hiding
  unreachable pairs.
- Cycle rank and triangle-based clustering remain distinct graph features.
- Results are deterministic and dataset-safe scalar values.
- Scoring and multi-objective use remain Phase 7.7 and Phase 7.8.
