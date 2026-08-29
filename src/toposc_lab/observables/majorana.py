"""Basis-aware Majorana polarization diagnostics for BdG eigenstates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from toposc_lab.core.results import SimulationResult
from toposc_lab.hamiltonians.nambu import NambuBasis


@dataclass(frozen=True)
class MajoranaDiagnostics:
    r"""Particle-hole content and polarization of one normalized BdG state.

    The convention is the unrotated Nambu basis implemented by
    :class:`NambuBasis`, with ``C = U_C K`` exchanging equal particle and hole
    components. The complex local polarization is
    ``P_i = psi_i^dagger C psi_i = 2 sum_a (u_(i,a) v_(i,a))*``.

    ``self_conjugacy = |sum_i P_i|`` lies between zero and one. The separate
    ``polarization_norm = sum_i |P_i|`` retains locally coherent contributions
    even when their complex phases cancel globally. A self-conjugacy near one
    does not establish zero energy, spatial separation, boundary localization,
    or non-trivial topology. Within a degenerate zero-energy subspace, arbitrary
    eigensolver rotations can also change the diagnostics of individual states.
    """

    site_probability: np.ndarray
    particle_probability: np.ndarray
    hole_probability: np.ndarray
    polarization: np.ndarray
    polarization_magnitude: np.ndarray
    total_polarization: complex
    self_conjugacy: float
    polarization_norm: float
    particle_weight: float
    hole_weight: float


def majorana_diagnostics(
    eigenvectors: np.ndarray,
    state_index: int,
    basis: NambuBasis,
) -> MajoranaDiagnostics:
    """Evaluate basis-aware Majorana polarization for one eigenvector column."""
    vectors = np.asarray(eigenvectors, dtype=complex)
    if vectors.ndim != 2:
        raise ValueError("eigenvectors must be a two-dimensional array")
    if vectors.shape[0] != basis.dimension:
        raise ValueError("eigenvectors must match the Nambu basis dimension")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("eigenvectors must contain only finite values")
    if not 0 <= state_index < vectors.shape[1]:
        raise ValueError("state_index is outside the available eigenvector range")

    state = basis.reorder_states(
        vectors[:, state_index],
        ordering="site_major",
    ).reshape(basis.n_sites, basis.nambu_components_per_site)
    normalization = float(np.sum(np.abs(state) ** 2))
    if normalization <= 0.0:
        raise ValueError("selected eigenvector must have positive norm")
    state = state / np.sqrt(normalization)

    n_components = basis.normal_components_per_site
    particle = state[:, :n_components]
    hole = state[:, n_components:]
    particle_probability = np.sum(np.abs(particle) ** 2, axis=1)
    hole_probability = np.sum(np.abs(hole) ** 2, axis=1)
    site_probability = particle_probability + hole_probability

    polarization = 2.0 * np.sum(np.conj(particle * hole), axis=1)
    total_polarization = complex(np.sum(polarization))
    self_conjugacy = float(np.clip(abs(total_polarization), 0.0, 1.0))

    return MajoranaDiagnostics(
        site_probability=site_probability,
        particle_probability=particle_probability,
        hole_probability=hole_probability,
        polarization=polarization,
        polarization_magnitude=np.abs(polarization),
        total_polarization=total_polarization,
        self_conjugacy=self_conjugacy,
        polarization_norm=float(
            np.clip(np.sum(np.abs(polarization)), 0.0, 1.0)
        ),
        particle_weight=float(np.sum(particle_probability)),
        hole_weight=float(np.sum(hole_probability)),
    )


def majorana_diagnostics_from_result(
    result: SimulationResult,
    state_index: int,
    basis: NambuBasis,
) -> MajoranaDiagnostics:
    """Evaluate diagnostics from a result after verifying its Nambu layout."""
    if result.basis_layout != basis.basis_layout:
        raise ValueError("result basis layout does not match the Nambu basis")
    return majorana_diagnostics(
        eigenvectors=result.eigenvectors,
        state_index=state_index,
        basis=basis,
    )
