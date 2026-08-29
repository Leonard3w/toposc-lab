"""Spin-dependent onsite Zeeman terms for graph Hamiltonians."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from toposc_lab.geometry import Geometry
from toposc_lab.hamiltonians.tight_binding import build_tight_binding_hamiltonian


ZeemanField: TypeAlias = ArrayLike
ZeemanFieldMap: TypeAlias = Mapping[int, ZeemanField]
ZeemanTerm: TypeAlias = ZeemanField | ZeemanFieldMap | Callable[[int], ZeemanField]
ComplexMatrix: TypeAlias = NDArray[np.complex128]


def zeeman_matrix(field: ZeemanField) -> ComplexMatrix:
    r"""Return ``b_x sigma_x + b_y sigma_y + b_z sigma_z``.

    ``field`` is a real three-component vector of energy coefficients in the
    spin basis ``(up, down)``. No sign, g-factor, or Bohr magneton is inserted
    implicitly; callers must include such physical prefactors themselves.
    """
    field_vector = _real_field_vector(field)
    field_x, field_y, field_z = field_vector
    return np.asarray(
        [
            [field_z, field_x - 1.0j * field_y],
            [field_x + 1.0j * field_y, -field_z],
        ],
        dtype=np.complex128,
    )


def build_zeeman_term(
    geometry: Geometry,
    *,
    field: ZeemanTerm,
) -> ComplexMatrix:
    r"""Build a site-major spinful Zeeman term on an arbitrary geometry.

    ``field`` may be one global real three-vector, a complete mapping keyed by
    site index, or a callable evaluated at every site. The returned matrix has
    dimension ``2 * n_sites`` and contains no hopping contribution.
    """
    onsite = {
        site: zeeman_matrix(_resolve_field(field, site))
        for site in geometry.site_indices
    }
    return build_tight_binding_hamiltonian(
        geometry,
        onsite=onsite,
        hopping=0.0,
        components_per_site=2,
    )


def _resolve_field(term: ZeemanTerm, site: int) -> ZeemanField:
    if callable(term):
        return term(site)
    if isinstance(term, Mapping):
        try:
            return term[site]
        except KeyError as error:
            raise ValueError(f"Zeeman field mapping has no value for site {site}") from error
    return term


def _real_field_vector(field: ZeemanField) -> NDArray[np.float64]:
    try:
        values = np.asarray(field, dtype=np.complex128)
    except (TypeError, ValueError) as error:
        raise TypeError("Zeeman field must be a real three-component vector") from error
    if values.shape != (3,):
        raise ValueError("Zeeman field must have shape (3,)")
    if not np.all(np.isfinite(values)):
        raise ValueError("Zeeman field must contain only finite values")
    if np.any(values.imag != 0.0):
        raise ValueError("Zeeman field must be real")
    return values.real.astype(np.float64, copy=False)
