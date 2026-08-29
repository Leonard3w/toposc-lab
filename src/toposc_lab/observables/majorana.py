"""Basis-aware Majorana polarization diagnostics for BdG eigenstates."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Literal, TypeAlias

import numpy as np

from toposc_lab.core.results import SimulationResult
from toposc_lab.hamiltonians.nambu import NambuBasis

SplittingClassification: TypeAlias = Literal[
    "numerical_zero_modes",
    "split_pair_candidate",
    "no_near_zero_structure",
]


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


@dataclass(frozen=True)
class FiniteSizeSplittingDiagnostics:
    """Spectral evidence for numerical zero modes or a split low-energy pair.

    ``quasiparticle_energy`` is the mean displacement of the selected negative
    and positive levels from zero. ``pair_level_separation`` is their full
    separation and equals twice that energy for an exactly symmetric pair.
    A split-pair classification is only a finite-size candidate: spatial
    localization, Majorana polarization, and topology require independent
    diagnostics.
    """

    classification: SplittingClassification
    zero_mode_indices: tuple[int, ...]
    negative_index: int | None
    positive_index: int | None
    negative_energy: float | None
    positive_energy: float | None
    quasiparticle_energy: float | None
    pair_level_separation: float | None
    pair_center_offset: float | None
    particle_hole_mismatch: float | None
    is_particle_hole_pair: bool
    is_split_pair_candidate: bool
    next_excitation_energy: float | None
    isolation_gap: float | None
    isolation_ratio: float | None


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


def finite_size_splitting_diagnostics(
    eigenvalues: np.ndarray,
    *,
    zero_tolerance: float = 1.0e-10,
    splitting_tolerance: float = 1.0e-3,
    particle_hole_tolerance: float = 1.0e-8,
) -> FiniteSizeSplittingDiagnostics:
    """Classify numerical zero levels versus a finite-size split pair.

    The input may be unsorted. Levels satisfying ``abs(E) <= zero_tolerance``
    are numerical zero modes. Otherwise, the closest negative and positive
    levels form a split-pair candidate only when both lie within
    ``splitting_tolerance`` and ``abs(E_plus + E_minus)`` does not exceed
    ``particle_hole_tolerance``.
    """
    energies = np.asarray(eigenvalues, dtype=float)
    if energies.ndim != 1 or energies.size == 0:
        raise ValueError("eigenvalues must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(energies)):
        raise ValueError("eigenvalues must contain only finite values")

    zero_tolerance = _nonnegative_finite_real(
        zero_tolerance,
        name="zero_tolerance",
    )
    splitting_tolerance = _nonnegative_finite_real(
        splitting_tolerance,
        name="splitting_tolerance",
    )
    particle_hole_tolerance = _nonnegative_finite_real(
        particle_hole_tolerance,
        name="particle_hole_tolerance",
    )
    if splitting_tolerance <= zero_tolerance:
        raise ValueError("splitting_tolerance must be greater than zero_tolerance")

    zero_indices_array = np.flatnonzero(np.abs(energies) <= zero_tolerance)
    zero_indices = tuple(int(index) for index in zero_indices_array)
    if zero_indices:
        nonzero_energies = np.abs(np.delete(energies, zero_indices_array))
        next_energy = (
            None if nonzero_energies.size == 0 else float(np.min(nonzero_energies))
        )
        return FiniteSizeSplittingDiagnostics(
            classification="numerical_zero_modes",
            zero_mode_indices=zero_indices,
            negative_index=None,
            positive_index=None,
            negative_energy=None,
            positive_energy=None,
            quasiparticle_energy=0.0,
            pair_level_separation=0.0,
            pair_center_offset=0.0,
            particle_hole_mismatch=0.0,
            is_particle_hole_pair=False,
            is_split_pair_candidate=False,
            next_excitation_energy=next_energy,
            isolation_gap=next_energy,
            isolation_ratio=None,
        )

    negative_indices = np.flatnonzero(energies < -zero_tolerance)
    positive_indices = np.flatnonzero(energies > zero_tolerance)
    if negative_indices.size == 0 or positive_indices.size == 0:
        return _no_split_pair_diagnostics()

    negative_index = int(negative_indices[np.argmax(energies[negative_indices])])
    positive_index = int(positive_indices[np.argmin(energies[positive_indices])])
    negative_energy = float(energies[negative_index])
    positive_energy = float(energies[positive_index])
    pair_scale = max(abs(negative_energy), abs(positive_energy))
    quasiparticle_energy = 0.5 * (
        abs(negative_energy) + abs(positive_energy)
    )
    pair_level_separation = positive_energy - negative_energy
    pair_center_offset = 0.5 * (positive_energy + negative_energy)
    particle_hole_mismatch = abs(positive_energy + negative_energy)
    is_particle_hole_pair = particle_hole_mismatch <= particle_hole_tolerance
    is_split_pair_candidate = (
        is_particle_hole_pair and pair_scale <= splitting_tolerance
    )

    remaining_mask = np.ones(energies.size, dtype=bool)
    remaining_mask[[negative_index, positive_index]] = False
    remaining = np.abs(energies[remaining_mask])
    next_energy = None if remaining.size == 0 else float(np.min(remaining))
    isolation_gap = None if next_energy is None else next_energy - pair_scale
    isolation_ratio = None if next_energy is None else next_energy / pair_scale

    classification: SplittingClassification = (
        "split_pair_candidate"
        if is_split_pair_candidate
        else "no_near_zero_structure"
    )
    return FiniteSizeSplittingDiagnostics(
        classification=classification,
        zero_mode_indices=(),
        negative_index=negative_index,
        positive_index=positive_index,
        negative_energy=negative_energy,
        positive_energy=positive_energy,
        quasiparticle_energy=quasiparticle_energy,
        pair_level_separation=pair_level_separation,
        pair_center_offset=pair_center_offset,
        particle_hole_mismatch=particle_hole_mismatch,
        is_particle_hole_pair=is_particle_hole_pair,
        is_split_pair_candidate=is_split_pair_candidate,
        next_excitation_energy=next_energy,
        isolation_gap=isolation_gap,
        isolation_ratio=isolation_ratio,
    )


def _no_split_pair_diagnostics() -> FiniteSizeSplittingDiagnostics:
    return FiniteSizeSplittingDiagnostics(
        classification="no_near_zero_structure",
        zero_mode_indices=(),
        negative_index=None,
        positive_index=None,
        negative_energy=None,
        positive_energy=None,
        quasiparticle_energy=None,
        pair_level_separation=None,
        pair_center_offset=None,
        particle_hole_mismatch=None,
        is_particle_hole_pair=False,
        is_split_pair_candidate=False,
        next_excitation_energy=None,
        isolation_gap=None,
        isolation_ratio=None,
    )


def _nonnegative_finite_real(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value
