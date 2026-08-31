# Phase 7.5: Topology integration

## Status

Accepted.

## Context

The topology layer already provides specialized diagnostics, a unified
`TopologyResult`, and dimension-/symmetry-/capability-aware dispatch. Phase 7.5 must attach
that evidence to `GeometryEvaluation` without adding new invariant formulas or blindly
applying methods based on coordinate-column count.

Different methods need different physical inputs and may disagree or remain unresolved.
The integration must preserve those separate outcomes instead of collapsing them into a
single topology score.

## Decision

`integrate_topology_results` requires a `TopologyDispatchDecision`. Specialized diagnostic
objects are converted through `unify_topology_result`; already unified results retain
their recorded confidence and convergence status. Every method must be applicable under
the supplied dispatch decision. Duplicate methods are rejected and accepted results are
stored in stable `TopologyMethod` order.

The supplied tuple replaces previous topology results. Integration-generated warnings use
an explicit prefix and are replaced on reintegration, while unrelated evaluation warnings
remain. They record dispatch warnings, rejected methods and reasons, applicable methods
without supplied results, nested method warnings, absent independent cross-validation,
unresolved results, and disagreement between resolved topological classifications.

An empty result is valid when no implemented method applies. Topology integration does not
require Majorana diagnostics because non-superconducting topological models must use the
same evaluation layer.

## Consequences

- Physical dimension is explicit and never inferred from embedding dimension.
- Applicability failures stop integration before misleading results are stored.
- Multiple topology diagnostics remain separately inspectable and deterministically
  ordered.
- Disagreement and lack of cross-validation remain visible rather than becoming a score.
- Calculating and orchestrating method-specific inputs remains Phase 7.10 work.
