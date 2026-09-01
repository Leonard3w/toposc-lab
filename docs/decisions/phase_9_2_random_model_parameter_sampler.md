# Phase 9.2: Random model-parameter sampler

## Status

Accepted.

## Decision

Random model-parameter sampling lives in `toposc_lab.search` beside but independent of
the Phase-9.1 geometry sampler. A `RandomModelParameterSamplingConfig` contains a deeply
immutable base parameter set, a nonempty mapping of selected top-level parameter names
to explicit distributions, and a positive sample count. Distribution keys must already
exist in the base set; every sampled result therefore preserves the exact top-level
parameter schema and every unselected value.

The initial general distribution vocabulary is deliberately small:

- `UniformFloatDistribution(low, high)` uses finite continuous `[low, high)` bounds;
- `UniformIntegerDistribution(low, high)` uses an int64-compatible discrete `[low, high)`
  range; and
- `CategoricalDistribution(choices)` selects uniformly among explicit typed values and
  rejects exact duplicates.

Parameter values reuse the existing Phase-8 `ModelParameterSet` and
`DisorderParameterValue` domain: `None`, booleans, integers, finite floats, strings,
tuples, and nested string-keyed mappings. Configurations and samples are normalized in
lexicographic key order and made deeply immutable. Categorical duplicate detection uses
the existing exact, type-sensitive parameter-set snapshot encoding, so integer `1` and
floating-point `1.0` remain distinct values.

`sample_random_model_parameters(...)` accepts one explicit nonnegative root seed and
creates one local `numpy.random.PCG64` stream. Sampling proceeds first by ascending
sample index and then by lexicographically sorted parameter name. Every distribution
consumes exactly one high-level NumPy draw at its position. The result records the root
seed, RNG algorithm, sampler version, ordered samples, and each sample index. Identical
configuration, seed, NumPy PCG64 behavior, and code version therefore produce identical
ordered output without reading or mutating global NumPy random state.

## Validation and failure semantics

Nonportable parameter values, invalid or non-finite bounds, empty distributions,
unsupported distribution objects, unknown parameter names, invalid counts, and invalid
seeds are configuration errors and raise before any sample is returned. Unexpected
distribution execution failures stop at the first affected sample and parameter with a
`RandomModelParameterSamplingError` retaining the sample index, parameter name,
distribution type, original exception type, and message.

There is no clipping, retry, resampling, fallback value, or partial result. The sampler
does not apply Pydantic coercion or model-specific cross-field validation. Such
compatibility is not inferable from a plain parameter mapping and belongs to explicit
model construction and candidate validation in later orchestration.

## Scientific boundaries

Phase 9.2 does not receive or sample geometries, construct a Hamiltonian, execute a
solver, evaluate physics, rank, persist, visualize, or interpret a parameter set. It
does not use the UI-specific model registry and does not select a default physical
model. In particular, it contains no chiral-p-wave parameter ranges, scientific
thresholds, seed partitions, or search budgets.

Model parameters, geometry, and disorder remain separate search dimensions. A later
experiment using the research charter must keep disorder definitions outside this
model-parameter space and apply the same frozen parameter-search policy to candidate
and reference geometries. Joint execution, model validation, failure retention, and
fair comparison remain later Phase-9 responsibilities.

The existing exact model-parameter-set ID remains a representation-sensitive snapshot
identifier. Phase 9.2 uses its typed encoding only to reject exact duplicate categorical
choices; it creates no dataset identity, persistence record, or deduplication policy.
