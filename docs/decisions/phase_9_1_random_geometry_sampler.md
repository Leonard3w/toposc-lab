# Phase 9.1: Random geometry sampler

## Status

Accepted.

## Decision

Random geometry sampling lives in `toposc_lab.search`, separate from geometry
generators, model parameters, physical evaluation, ranking, visualization, and
persistence. A `RandomGeometrySamplingConfig` contains a positive sample count and an
ordered nonempty tuple of explicit `GeometrySamplingRecipe` values. Each recipe names a
registered generator and fixed JSON-safe geometry parameters. There is no implicit
default recipe or family, so the sampler cannot silently introduce abstract graphs into
a later spatially embedded physical experiment.

`sample_random_geometries(...)` accepts the configuration, one explicit nonnegative root
seed, and a `GeometryGeneratorRegistry`. It preflights every recipe key before creating
an RNG. The sampler uses `numpy.random.PCG64` locally and never obtains randomness from
global NumPy state. For every sample index in ascending order it:

1. chooses one recipe uniformly with `Generator.integers`;
2. draws one raw PCG64 word;
3. supplies that word through `GeometryGenerationRequest.seed` only when the registered
   generator declares `stochastic=True`, otherwise supplies `None`;
4. invokes the existing generator protocol exactly once;
5. checks the protocol generation provenance;
6. runs `validate_geometry(..., require_connected=False)` once; and
7. appends the successful sample in request order.

The seed word is drawn even for deterministic generators. Consequently, changing a
recipe between deterministic and stochastic does not by itself shift later recipe
selection positions. The configuration order is part of the sampling definition. The
sampler algorithm has explicit version 1, and the result records both that version and
the RNG algorithm.

Each `RandomGeometrySample` keeps its sample index, selected recipe index, exact
`GeometryGenerationRequest`, unmodified `Geometry`, and Phase-6.22 validation report.
Disconnectedness remains a warning under the neutral base policy. A later experiment
may impose connectivity or spatial/model compatibility through the existing candidate
validity policy; Phase 9.1 neither filters nor retries such candidates.

## Failure semantics

Malformed configurations, invalid root seeds, invalid registries, and unknown generator
keys raise before sampling starts. If a selected generator raises, returns the wrong
type, or violates generation provenance, sampling stops at that position with a
`RandomGeometrySamplingError` at the `generation` stage. If model-independent geometry
validation finds an error, it stops at the `geometry_validation` stage. The error records
the sample index, recipe index, generator key, original exception type, and message.

There is no retry, skip, fallback generator, or partial success result. These are sampler
infrastructure failures, not Phase-7 scientific candidate-validity outcomes. Ordinary
`Exception` values are contextualized; process-control exceptions are not captured.

## Scientific and identity boundaries

The sampler does not construct a model, sample model parameters, evaluate physics,
compute scores, rank, deduplicate, store, or visualize. It does not infer coordinates,
boundaries, embedding dimension, physical dimension, topology, or scientific validity.
It returns each generator's geometry unchanged, preserving the reference orientation of
every `GeometryEdge` and preserving the absence of geometric data on abstract graphs.

The Phase-6 Weisfeiler--Lehman graph hash is not computed: it remains only an
isomorphism-candidate fingerprint. The Phase-7.11 exact geometry ID is also not computed:
it remains a representation-dependent serialized snapshot identifier for later
evaluation and persistence. Phase 9.1 performs no duplicate removal or identity claim.

## Consequences

- Identical configuration, root seed, registry definitions, generator versions, NumPy
  PCG64 behavior, and code version produce the same ordered sampling output.
- Deterministic and stochastic generators retain the common Phase-6.1 seed contract.
- Geometry-family choice is explicit and auditable without embedding the first chiral
  p-wave experiment in general infrastructure.
- Later Phase-9 steps can consume ordered samples without Phase 9.1 taking ownership of
  parameter search, batch evaluation, persistence, ranking, or scientific interpretation.
