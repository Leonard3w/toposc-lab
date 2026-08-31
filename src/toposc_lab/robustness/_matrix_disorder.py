"""Internal basis and validation helpers shared by matrix disorder types."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from toposc_lab.geometry import Geometry
from toposc_lab.hamiltonians.nambu import NambuBasis


@dataclass(frozen=True, slots=True)
class MatrixDisorderBasisContract:
    """Explicit normal-state or Nambu matrix-layout contract."""

    normal_components_per_site: int
    representation: str
    ordering: str
    nambu_basis: NambuBasis | None

    @property
    def particle_hole_embedding(self) -> str:
        """Describe whether a normal perturbation is Nambu embedded."""
        return "not_applicable" if self.nambu_basis is None else "opposite_sign"


def resolve_matrix_disorder_basis(
    geometry: Geometry,
    *,
    components_per_site: int | None,
    nambu_basis: NambuBasis | None,
) -> MatrixDisorderBasisContract:
    """Validate an explicit normal-state or Nambu basis selection."""
    if nambu_basis is None:
        resolved_components = (
            1
            if components_per_site is None
            else positive_integer(
                components_per_site,
                name="components_per_site",
            )
        )
        return MatrixDisorderBasisContract(
            normal_components_per_site=resolved_components,
            representation="normal_state",
            ordering="site_major",
            nambu_basis=None,
        )
    if not isinstance(nambu_basis, NambuBasis):
        raise TypeError("nambu_basis must be NambuBasis or None")
    if components_per_site is not None:
        raise ValueError(
            "components_per_site must be omitted when nambu_basis is supplied"
        )
    if nambu_basis.n_sites != geometry.n_sites:
        raise ValueError("nambu_basis site count must match geometry")
    return MatrixDisorderBasisContract(
        normal_components_per_site=nambu_basis.normal_components_per_site,
        representation="nambu_bdg",
        ordering=nambu_basis.ordering,
        nambu_basis=nambu_basis,
    )


def nonnegative_finite_real(value: float, *, name: str) -> float:
    """Normalize a finite, nonnegative real parameter."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def positive_integer(value: int, *, name: str) -> int:
    """Normalize a strictly positive integer parameter."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be at least one")
    return result
