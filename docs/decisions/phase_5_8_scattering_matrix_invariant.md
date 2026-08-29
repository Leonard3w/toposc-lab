# Phase 5.8 decision: defer scattering-matrix invariants

Date: 2026-08-29  
Status: deferred until open-system transport infrastructure exists

## Decision

Do not add a scattering-matrix topological invariant in Phase 5.8.

This is a deliberate scope decision, not a rejection of the method. Reflection-matrix
invariants can provide an efficient boundary-level classification and a direct connection
to transport, especially for disordered systems. At present, however, TopoSC Lab only has
closed finite Hamiltonians. It has no physical route from a model and geometry to a
symmetry-compatible reflection matrix.

Adding only a determinant, Pfaffian, signature, or winding helper for an arbitrary matrix
would be scientifically unsafe: it could validate algebraic properties of the input but
could not establish that the matrix came from a unitary scattering problem, uses the
correct channel basis, respects the declared Altland-Zirnbauer symmetry, or represents an
insulating sample at the probe energy.

## Evidence from the current codebase

The project currently has no:

- lead or contact representation with deterministic channel ordering;
- retarded lead self-energy or surface Green-function solver;
- Fisher-Lee, Mahaux-Weidenmueller, or equivalent scattering solver;
- transmission/reflection block result with unitarity diagnostics;
- symmetry operators acting in the asymptotic channel basis;
- twisted-boundary or flux family for a two-dimensional reflection winding.

The existing closed-system methods already cover the current discovery workflow:

- the 1D class-D/BDI Pfaffian invariant;
- the 1D real-space chiral winding invariant;
- the 2D Bott index;
- the spatially resolved local Chern marker;
- the 2D spectral-localizer gap and half-signature.

A scattering invariant would therefore not add a trustworthy independent observable until
the open-system prerequisites are present.

## Conditions for reopening the decision

Implement scattering invariants only after all of the following exist:

1. A typed lead/contact model specifying couplings, propagating channels, channel order,
   and the action of particle-hole, time-reversal, and chiral symmetries.
2. A retarded open-system solver that returns the full scattering matrix and its reflection
   and transmission blocks at a requested energy.
3. Numerical diagnostics for scattering unitarity, symmetry residuals, lead convergence,
   and the insulating-reflection condition.
4. A canonical contact prescription for generated geometries, so the invariant is not an
   uncontrolled function of arbitrary lead placement.
5. Benchmarks showing agreement with existing invariants away from phase boundaries and
   controlled failure at gap closings.

The first useful restricted implementations would then be:

- 1D class D: the sign of the determinant of the zero-energy reflection matrix in a real
  Majorana channel basis;
- other nontrivial 1D classes: the class-appropriate determinant, Pfaffian, or signature;
- 2D class A: the winding of the reflection determinant over a validated twisted-boundary
  cycle.

## Required benchmark and acceptance tests

- Kitaev-chain agreement with the existing Pfaffian invariant over trivial, topological,
  disordered, and transition-point cases.
- QWZ or Haldane agreement between reflection winding, Bott index, local Chern marker, and
  spectral localizer.
- Stability under reasonable lead-coupling changes without changing the topological phase.
- Explicit rejection when the scattering matrix is nonunitary beyond tolerance, the channel
  symmetry representation is invalid, or transmission shows that the assumed insulating
  reflection problem is not realized.

## Primary references

- I. C. Fulga, F. Hassler, and A. R. Akhmerov,
  [*Scattering theory of topological insulators and superconductors*](https://arxiv.org/abs/1106.6351),
  Physical Review B 85, 165409 (2012).
- I. C. Fulga, F. Hassler, A. R. Akhmerov, and C. W. J. Beenakker,
  [*Scattering formula for the topological quantum number of a disordered multi-mode wire*](https://arxiv.org/abs/1101.1749),
  Physical Review B 83, 155429 (2011).
