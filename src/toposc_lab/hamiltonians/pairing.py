"""Geometry-based superconducting pairing-matrix construction."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from numbers import Integral
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from toposc_lab.geometry import Geometry, GeometryEdge


PairingValue: TypeAlias = ArrayLike
EdgePairingMap: TypeAlias = Mapping[GeometryEdge, PairingValue]
EdgePairingTerm: TypeAlias = (
    PairingValue | EdgePairingMap | Callable[[GeometryEdge], PairingValue]
)
OnsitePairingMap: TypeAlias = Mapping[int, PairingValue]
OnsitePairingTerm: TypeAlias = (
    PairingValue | OnsitePairingMap | Callable[[int], PairingValue]
)
ComplexMatrix: TypeAlias = NDArray[np.complex128]


def build_spinless_p_wave_pairing(
    geometry: Geometry,
    *,
    pairing: EdgePairingTerm,
) -> ComplexMatrix:
    r"""Build an antisymmetric spinless p-wave pairing matrix on graph edges.

    Each geometry edge supplies a reference orientation. For a resolved complex
    coefficient ``Delta_ij`` on the stored edge ``i -> j``, the builder sets
    ``Delta[i, j] = Delta_ij`` and ``Delta[j, i] = -Delta_ij``. The minus sign
    follows fermionic antisymmetry and does not complex-conjugate the value.

    ``pairing`` may be a global scalar, a complete mapping keyed by the
    geometry's ``GeometryEdge`` objects, or a callable evaluated per edge.
    """
    pairing_matrix = np.zeros(
        (geometry.n_sites, geometry.n_sites),
        dtype=np.complex128,
    )

    for edge in geometry.edges:
        value = _resolve_edge_pairing(pairing, edge)
        coefficient = _complex_scalar(
            value,
            name=f"pairing term on edge ({edge.source}, {edge.target})",
        )
        pairing_matrix[edge.source, edge.target] += coefficient
        pairing_matrix[edge.target, edge.source] -= coefficient

    return pairing_matrix


def build_onsite_s_wave_pairing(
    geometry: Geometry,
    *,
    pairing: OnsitePairingTerm,
) -> ComplexMatrix:
    r"""Build onsite spin-singlet s-wave pairing in a spin-up/down basis.

    The returned normal-state pairing block has dimension ``2 * n_sites`` and
    uses site-major order ``(site 0 up, site 0 down, site 1 up, ...)``. At each
    site ``i`` it contains

    .. math::

        \Delta_i i\sigma_y =
        \begin{pmatrix}0 & \Delta_i \\ -\Delta_i & 0\end{pmatrix},

    which is antisymmetric as required for a local even-parity spin singlet.
    ``pairing`` may be a global scalar, a complete site-index mapping, or a
    callable evaluated for every site.
    """
    normal_dimension = 2 * geometry.n_sites
    pairing_matrix = np.zeros(
        (normal_dimension, normal_dimension),
        dtype=np.complex128,
    )

    for site in geometry.site_indices:
        value = _resolve_onsite_pairing(pairing, site)
        coefficient = _complex_scalar(
            value,
            name=f"onsite pairing term at site {site}",
        )
        spin_up = 2 * site
        spin_down = spin_up + 1
        pairing_matrix[spin_up, spin_down] += coefficient
        pairing_matrix[spin_down, spin_up] -= coefficient

    return pairing_matrix


def build_chiral_p_wave_pairing(
    geometry: Geometry,
    *,
    pairing: EdgePairingTerm,
    chirality: int = 1,
    plane_axes: tuple[int, int] = (0, 1),
) -> ComplexMatrix:
    r"""Build spinless chiral ``p_x + i chirality p_y`` edge pairing.

    For the unit direction from an edge's stored source to target, the resolved
    edge amplitude is multiplied by ``d_x + 1j * chirality * d_y``. The two
    spatial components are selected by ``plane_axes`` from an arbitrary-
    dimensional embedding. Coordinates or explicit edge displacements are
    required; spatial directions are never invented for abstract graphs.
    """
    chirality = _validate_chirality(chirality)
    axes = _validate_plane_axes(plane_axes)

    def directional_pairing(edge: GeometryEdge) -> complex:
        try:
            direction = geometry.direction(edge.source, edge.target)
        except ValueError as error:
            raise ValueError(
                "chiral p-wave pairing requires spatial coordinates or "
                "explicit nonzero edge displacements"
            ) from error
        if max(axes) >= direction.size:
            raise ValueError(
                f"pairing plane axes {axes} are outside direction dimension "
                f"{direction.size}"
            )
        amplitude = _complex_scalar(
            _resolve_edge_pairing(pairing, edge),
            name=f"pairing term on edge ({edge.source}, {edge.target})",
        )
        return complex(
            amplitude
            * (direction[axes[0]] + 1.0j * chirality * direction[axes[1]])
        )

    return build_spinless_p_wave_pairing(
        geometry,
        pairing=directional_pairing,
    )


def build_d_wave_pairing(
    geometry: Geometry,
    *,
    pairing: EdgePairingTerm,
    plane_axes: tuple[int, int] = (0, 1),
) -> ComplexMatrix:
    r"""Build nearest-neighbor spin-singlet ``d_x2-y2`` edge pairing.

    The spatial form factor is ``d_x**2 - d_y**2`` for each unit edge
    direction in the selected embedding plane. It is even under reversal of
    the edge direction. Each spatial bond is multiplied by ``i sigma_y`` in
    the site-major spin basis ``(up, down)``, making the complete pairing block
    antisymmetric under transposition.
    """
    axes = _validate_plane_axes(plane_axes)
    normal_dimension = 2 * geometry.n_sites
    pairing_matrix = np.zeros(
        (normal_dimension, normal_dimension),
        dtype=np.complex128,
    )

    for edge in geometry.edges:
        try:
            direction = geometry.direction(edge.source, edge.target)
        except ValueError as error:
            raise ValueError(
                "d-wave pairing requires spatial coordinates or explicit "
                "nonzero edge displacements"
            ) from error
        if max(axes) >= direction.size:
            raise ValueError(
                f"pairing plane axes {axes} are outside direction dimension "
                f"{direction.size}"
            )
        amplitude = _complex_scalar(
            _resolve_edge_pairing(pairing, edge),
            name=f"pairing term on edge ({edge.source}, {edge.target})",
        )
        form_factor = direction[axes[0]] ** 2 - direction[axes[1]] ** 2
        coefficient = complex(amplitude * form_factor)

        source_up = 2 * edge.source
        source_down = source_up + 1
        target_up = 2 * edge.target
        target_down = target_up + 1

        pairing_matrix[source_up, target_down] += coefficient
        pairing_matrix[source_down, target_up] -= coefficient
        pairing_matrix[target_up, source_down] += coefficient
        pairing_matrix[target_down, source_up] -= coefficient

    return pairing_matrix


def _resolve_edge_pairing(
    term: EdgePairingTerm,
    edge: GeometryEdge,
) -> PairingValue:
    if callable(term):
        return term(edge)
    if isinstance(term, Mapping):
        try:
            return term[edge]
        except KeyError as error:
            raise ValueError(
                "pairing mapping has no value for edge "
                f"({edge.source}, {edge.target})"
            ) from error
    return term


def _resolve_onsite_pairing(
    term: OnsitePairingTerm,
    site: int,
) -> PairingValue:
    if callable(term):
        return term(site)
    if isinstance(term, Mapping):
        try:
            return term[site]
        except KeyError as error:
            raise ValueError(
                f"onsite pairing mapping has no value for site {site}"
            ) from error
    return term


def _complex_scalar(value: PairingValue, *, name: str) -> complex:
    try:
        array = np.asarray(value, dtype=np.complex128)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a scalar") from error
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar")

    coefficient = complex(array.item())
    if not np.isfinite(coefficient):
        raise ValueError(f"{name} must be finite")
    return coefficient


def _validate_chirality(chirality: int) -> int:
    if isinstance(chirality, bool) or not isinstance(chirality, Integral):
        raise TypeError("chirality must be either +1 or -1")
    chirality = int(chirality)
    if chirality not in (-1, 1):
        raise ValueError("chirality must be either +1 or -1")
    return chirality


def _validate_plane_axes(plane_axes: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(plane_axes, tuple) or len(plane_axes) != 2:
        raise TypeError("plane_axes must be a pair of integer axes")
    if any(isinstance(axis, bool) or not isinstance(axis, Integral) for axis in plane_axes):
        raise TypeError("plane_axes must be a pair of integer axes")
    axes = (int(plane_axes[0]), int(plane_axes[1]))
    if axes[0] < 0 or axes[1] < 0:
        raise ValueError("plane_axes must be non-negative")
    if axes[0] == axes[1]:
        raise ValueError("plane_axes must select two distinct axes")
    return axes
