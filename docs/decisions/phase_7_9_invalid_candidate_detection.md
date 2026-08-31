# Phase 7.9: Invalid-candidate detection

## Decision

Candidate validity is represented by an immutable `CandidateValidityReport` containing
machine-readable warning and error issues. A candidate is valid exactly when the report
contains no error-level issue. Callers may inspect the report for filtering or explicitly
raise `InvalidCandidateError` through `raise_for_errors()`.

`validate_candidate` combines five sources of evidence without changing them:

- the existing model-independent `validate_geometry` report;
- explicit `ModelGeometryRequirements` supplied by a caller or future model adapter;
- consistency between geometry site count, model basis, and simulation-result basis;
- Hamiltonian Hermiticity, eigenvector orthonormality, and eigenpair residuals under an
  explicit numerical tolerance;
- optional topology resolution/convergence requirements and serializable failures from a
  named execution stage.

Model requirements can demand connectivity, at least one edge, explicit boundary sites,
or particular embedding axes with a usable direction on every edge. No such requirement
is a default: disconnected, abstract, boundary-free, or edgeless geometries are not
universally unphysical. The validity layer never infers physical requirements from a
model name, graph degree, embedding dimension alone, or warning text.

Stage failures retain only the stage, exception type name, and message. They do not retain
exception objects or tracebacks and are therefore suitable for later dataset records.
Ordinary API type errors still raise immediately; candidate-level failures are reported.

## Numerical semantics

Hamiltonian Hermiticity and eigenpair residuals are normalized by an explicit matrix or
energy scale. Eigenvector orthonormality uses an absolute residual. Unresolved topology
and missing convergence studies become exclusion reasons only when the selected policy
requires them. A finite-size result is not rejected merely because a scientific warning
exists.

This phase does not construct a model, catch pipeline exceptions, run a solver, execute an
observable, rank candidates, or define `evaluate_geometry(...)`. Phase 7.10 remains the
owner of orchestration and will be able to feed its caught failures into this contract.
