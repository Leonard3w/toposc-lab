# Phase 7.7: Basic scalar score

## Decision

The first scalar score is an explicit engineering ranking derived from an existing
`GeometryEvaluation`. It is returned as a separate immutable `BasicScalarScore`; the
scientific evaluation record is unchanged.

Callers must select components and provide a strictly positive weight for every selected
component. There are no default weights. The public components are:

- normalized gap: `gap / (gap + gap_scale)`;
- presence of at least one numerical zero mode;
- maximum IPR among evaluated states;
- maximum boundary weight among evaluated states;
- maximum Majorana self-conjugacy among evaluated states;
- fraction of resolved topology methods reporting a topological result.

The score is the weighted mean of these values. Its result retains each normalized
component, normalized weight, and weighted contribution. Requested data must be present,
and every topology result used by the fraction must be resolved. The implementation never
silently drops a component or changes the denominator for one candidate.

## Scientific boundary

The score is neither an observable nor a topological invariant. It is comparable only when
the component set, weights, `gap_scale`, and input completeness are identical. In
particular, a zero mode does not prove Majorana character, maximum state summaries can
hide distributions and degeneracy-basis dependence, and a topology-method fraction does
not replace the individual method results and their assumptions.

The score deliberately remains separate from `GeometryEvaluation` so later analysis can
continue to preserve the original scientific quantities. Multi-objective evaluation,
invalid-candidate filtering, and the unified evaluation pipeline remain Phase 7.8, 7.9,
and 7.10 work respectively.
