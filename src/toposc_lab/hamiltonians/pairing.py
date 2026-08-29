"""Geometry-based superconducting pairing-matrix construction."""

from __future__ import annotations

from collections.abc import Callable, Mapping
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
