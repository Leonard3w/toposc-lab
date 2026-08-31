# Phase 8.8: Physical parameter perturbation

## Decision

Physical model parameters are neither geometries nor Hamiltonian matrices. Phase 8.8
therefore extends the neutral disorder target enum with `MODEL_PARAMETERS` and represents
one model configuration as a deeply immutable `ModelParameterSet`. The Phase-8.1
executor remains the sole owner of the explicit NumPy PCG64 stream and common disorder
provenance. Geometry, model construction, numerical evaluation, scientific results, and
future ensemble output remain separate.

A parameter snapshot uses the labeled
`toposc-model-parameter-set-v1-sha256` scheme. Its canonical payload sorts mapping keys
and type-tags nulls, booleans, integers, exact hexadecimal floats, strings, tuples, and
nested mappings. This distinguishes discrete integers from continuous floats and
preserves details such as signed zero. The executor requires a parameter transform to
retain the exact top-level key set, preventing a value perturbation from silently
changing the model schema.

## Uniform additive transform

`apply_uniform_parameter_perturbation(parameters, widths, seed)` perturbs only explicitly
selected top-level continuous floating-point values. Selected names are sorted
lexicographically, and one independent additive offset is drawn for each from
`[-width / 2, width / 2]`. Every width must be finite and nonnegative. Unselected values,
including nested configuration, remain unchanged and deeply frozen in the output.

Discrete integers, booleans, strings, tuples, and mappings cannot be selected by this
continuous additive transform. This prevents parameters such as site count, chirality,
boundary convention, basis ordering, or plane axes from being converted silently into
floating-point values. A caller that needs discrete alternatives must define a separate,
explicitly versioned physical perturbation in a later extension.

No model-specific range, sign, correlation, or conservation law is inferred. Values are
not clipped, resampled, or repaired. The caller explicitly reconstructs its model from
the returned parameter mapping; model construction and the Phase-7 validity pipeline own
the scientific admissibility decision. The exact source/result parameter snapshots,
selected widths, draw order, seed, RNG algorithm, and transform version make that
relationship reproducible and auditable.

The parameter-set ID is a representation-sensitive configuration snapshot. It is not a
geometry identity, Hamiltonian identity, physical equivalence proof, or replacement for
the Phase-7.11 evaluation `ReproducibilityRecord`.

This phase does not execute ensembles, compute robustness metrics or uncertainty,
perform finite-size analysis, rank candidates, generate datasets, or begin Phase 9.
