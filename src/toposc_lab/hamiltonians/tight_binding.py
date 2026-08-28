"""Generic graph-based tight-binding Hamiltonian construction."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from numbers import Integral
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from toposc_lab.geometry import Geometry, GeometryEdge


TermValue: TypeAlias = ArrayLike
OnsiteMap: TypeAlias = Mapping[int, TermValue]
HoppingMap: TypeAlias = Mapping[GeometryEdge, TermValue]
OnsiteTerm: TypeAlias = TermValue | OnsiteMap | Callable[[int], TermValue]
HoppingTerm: TypeAlias = TermValue | HoppingMap | Callable[[GeometryEdge], TermValue]
ComplexMatrix: TypeAlias = NDArray[np.complex128]


def build_tight_binding_hamiltonian(
    geometry: Geometry,
    *,
    hopping: HoppingTerm,
    onsite: OnsiteTerm = 0.0,
    components_per_site: int = 1,
) -> ComplexMatrix:
    r"""Build a Hermitian tight-binding matrix on an arbitrary geometry.

    The matrix represents

    .. math::

        H = \sum_i c_i^\dagger h_i c_i
            + \sum_{(i,j) \in E}
              \left(c_i^\dagger t_{ij} c_j + \mathrm{h.c.}\right).

    Basis states use site-major ordering. For ``m = components_per_site``, the
    components belonging to site ``i`` occupy indices ``m*i`` through
    ``m*(i+1)-1``. A scalar term is expanded to that scalar times the ``m x m``
    identity matrix.

    A hopping callable is evaluated in each ``GeometryEdge``'s stored
    source-to-target orientation. The reverse block is inserted automatically
    as the Hermitian conjugate, so the hopping block itself need not be
    Hermitian. Site- and edge-dependent coefficients may be supplied either as
    callables or as complete mappings keyed by site index and ``GeometryEdge``,
    respectively. Scalars and matrices remain global coefficients.
    """
    components_per_site = _validate_components_per_site(components_per_site)
    dimension = geometry.n_sites * components_per_site
    hamiltonian = np.zeros((dimension, dimension), dtype=np.complex128)

    for site in geometry.site_indices:
        value = _resolve_onsite_term(onsite, site)
        onsite_matrix = _term_matrix(
            value,
            components_per_site=components_per_site,
            name=f"onsite term at site {site}",
        )
        if not np.allclose(onsite_matrix, onsite_matrix.conj().T):
            raise ValueError(f"onsite term at site {site} must be Hermitian")

        site_block = _site_block(site, components_per_site)
        hamiltonian[site_block, site_block] += onsite_matrix

    for edge in geometry.edges:
        value = _resolve_hopping_term(hopping, edge)
        hopping_matrix = _term_matrix(
            value,
            components_per_site=components_per_site,
            name=f"hopping term on edge ({edge.source}, {edge.target})",
        )
        source_block = _site_block(edge.source, components_per_site)
        target_block = _site_block(edge.target, components_per_site)
        hamiltonian[source_block, target_block] += hopping_matrix
        hamiltonian[target_block, source_block] += hopping_matrix.conj().T

    return hamiltonian


def _resolve_onsite_term(term: OnsiteTerm, site: int) -> TermValue:
    if callable(term):
        return term(site)
    if isinstance(term, Mapping):
        try:
            return term[site]
        except KeyError as error:
            raise ValueError(f"onsite mapping has no value for site {site}") from error
    return term


def _resolve_hopping_term(term: HoppingTerm, edge: GeometryEdge) -> TermValue:
    if callable(term):
        return term(edge)
    if isinstance(term, Mapping):
        try:
            return term[edge]
        except KeyError as error:
            raise ValueError(
                "hopping mapping has no value for edge "
                f"({edge.source}, {edge.target})"
            ) from error
    return term


def _validate_components_per_site(components_per_site: int) -> int:
    if isinstance(components_per_site, bool) or not isinstance(components_per_site, Integral):
        raise TypeError("components_per_site must be an integer")
    components_per_site = int(components_per_site)
    if components_per_site < 1:
        raise ValueError("components_per_site must be at least one")
    return components_per_site


def _term_matrix(
    value: TermValue,
    *,
    components_per_site: int,
    name: str,
) -> ComplexMatrix:
    try:
        array = np.asarray(value, dtype=np.complex128)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a scalar or square matrix") from error

    if array.ndim == 0:
        matrix = complex(array.item()) * np.eye(
            components_per_site,
            dtype=np.complex128,
        )
    elif array.shape == (components_per_site, components_per_site):
        matrix = array.copy()
    else:
        expected_shape = (components_per_site, components_per_site)
        raise ValueError(f"{name} must be a scalar or have shape {expected_shape}")

    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")

    return matrix


def _site_block(site: int, components_per_site: int) -> slice:
    start = site * components_per_site
    return slice(start, start + components_per_site)
