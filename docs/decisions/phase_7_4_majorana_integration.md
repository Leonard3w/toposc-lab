# Phase 7.4: Majorana diagnostic integration

## Status

Accepted.

## Context

Phase 7.3 provides state-resolved probabilities, IPR, and boundary localization for every
selected low-energy state. Phase 7.4 must integrate the existing Majorana-polarization
observable without treating numerical zero energy, localization, self-conjugacy, or
topology as interchangeable evidence.

Majorana polarization depends on an explicit particle-hole basis convention. A generic
`BasisLayout` identifies component placement but does not by itself declare which
components are particle-hole partners.

## Decision

`evaluate_majorana_diagnostics` requires an explicit `NambuBasis` and a complete Phase
7.3 result. It evaluates exactly the state indices in `low_energy_states`, including
selected states outside the numerical zero-mode tolerance, and stores the existing
`MajoranaDiagnostics` objects in `majorana_metrics`.

Before accepting each result, the integration verifies that localization component labels
and shapes agree with the Nambu convention. It then independently computes the
site-resolved Nambu probability and compares it with the Phase-7.3 localization
probability. This catches inconsistent site/component ordering instead of silently
combining results from different bases.

Warnings retain the unrotated matching-component convention, the dependence of
individual-state diagnostics on rotations within degenerate subspaces, and the fact that
high self-conjugacy alone establishes neither zero energy, spatial separation, boundary
localization, nor topology.

`FiniteSizeSplittingDiagnostics` remains a separate spectral observable. It is not
flattened into the state-indexed `majorana_metrics` mapping. No topology calculation is
performed.

## Consequences

- Component-major and site-major Nambu orderings share the same integration path.
- Matrix-level and standardized solver results remain usable with an explicit basis.
- Spectral, localization, and Majorana evidence remain separately inspectable.
- Basis inconsistencies fail before a scientifically misleading aggregate is returned.
- Topology integration remains exclusively in Phase 7.5.
