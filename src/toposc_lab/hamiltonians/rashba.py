"""Direction-dependent Rashba spin-orbit terms on graph edges."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from toposc_lab.geometry import Geometry, GeometryEdge
from toposc_lab.hamiltonians.tight_binding import build_tight_binding_hamiltonian


PlaneAxes: TypeAlias = tuple[int, int]
RashbaCouplingMap: TypeAlias = Mapping[GeometryEdge, float]
RashbaCoupling: TypeAlias = (
    float | RashbaCouplingMap | Callable[[GeometryEdge], float]
)
ComplexMatrix: TypeAlias = NDArray[np.complex128]


def rashba_hopping_matrix(
    direction: ArrayLike,
    *,
    coupling: float,
    plane_axes: PlaneAxes = (0, 1),
) -> ComplexMatrix:
    r"""Return ``i alpha (d_y sigma_x - d_x sigma_y)`` for one oriented edge.

    ``direction`` may belong to an arbitrary-dimensional embedding and is
    normalized before its components along ``plane_axes=(x_axis, y_axis)`` are
    selected. ``coupling`` is a real energy coefficient; lattice-spacing and
    discretization prefactors are not inserted implicitly.
    """
    axes = _validate_plane_axes(plane_axes)
    direction_vector = _unit_direction(direction)
    if max(axes) >= direction_vector.size:
        raise ValueError(
            f"Rashba plane axes {axes} are outside direction dimension "
            f"{direction_vector.size}"
        )
    coupling = _real_coupling(coupling)

    direction_x = direction_vector[axes[0]]
    direction_y = direction_vector[axes[1]]
    sigma_x = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sigma_y = np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    return np.asarray(
        1.0j * coupling * (direction_y * sigma_x - direction_x * sigma_y),
        dtype=np.complex128,
    )


def build_rashba_term(
    geometry: Geometry,
    *,
    coupling: RashbaCoupling,
    plane_axes: PlaneAxes = (0, 1),
) -> ComplexMatrix:
    r"""Build a site-major spinful Rashba term from geometry edge directions.

    The geometry must provide coordinates or explicit edge displacements.
    ``coupling`` may be a global real scalar, a complete mapping keyed by the
    geometry's oriented edges, or a callable evaluated for each edge.
    """
    axes = _validate_plane_axes(plane_axes)

    def hopping(edge: GeometryEdge) -> ComplexMatrix:
        try:
            displacement = geometry.displacement_between(edge.source, edge.target)
        except ValueError as error:
            raise ValueError(
                "Rashba coupling requires spatial coordinates or explicit "
                "edge displacements"
            ) from error
        return rashba_hopping_matrix(
            displacement,
            coupling=_resolve_coupling(coupling, edge),
            plane_axes=axes,
        )

    return build_tight_binding_hamiltonian(
        geometry,
        onsite=0.0,
        hopping=hopping,
        components_per_site=2,
    )


def _resolve_coupling(term: RashbaCoupling, edge: GeometryEdge) -> float:
    if callable(term):
        return term(edge)
    if isinstance(term, Mapping):
        try:
            return term[edge]
        except KeyError as error:
            raise ValueError(
                "Rashba coupling mapping has no value for edge "
                f"({edge.source}, {edge.target})"
            ) from error
    return term


def _validate_plane_axes(plane_axes: PlaneAxes) -> PlaneAxes:
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


def _unit_direction(direction: ArrayLike) -> NDArray[np.float64]:
    try:
        values = np.asarray(direction, dtype=np.complex128)
    except (TypeError, ValueError) as error:
        raise TypeError("direction must be a real one-dimensional vector") from error
    if values.ndim != 1 or values.size == 0:
        raise ValueError("direction must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(values)):
        raise ValueError("direction must contain only finite values")
    if np.any(values.imag != 0.0):
        raise ValueError("direction must be real")

    real_values = values.real.astype(np.float64, copy=False)
    norm = float(np.linalg.norm(real_values))
    if norm == 0.0:
        raise ValueError("Rashba edge direction must be nonzero")
    return real_values / norm


def _real_coupling(coupling: float) -> float:
    if isinstance(coupling, bool) or not isinstance(coupling, Real):
        raise TypeError("Rashba coupling must be a real number")
    coupling = float(coupling)
    if not np.isfinite(coupling):
        raise ValueError("Rashba coupling must be finite")
    return coupling
