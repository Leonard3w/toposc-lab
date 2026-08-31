"""Concrete additive onsite disorder on finite Hamiltonian matrices."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np

from toposc_lab.evaluation.reproducibility import exact_geometry_id
from toposc_lab.geometry import Geometry
from toposc_lab.hamiltonians.disorder import sample_uniform_site_disorder
from toposc_lab.hamiltonians.nambu import NambuBasis
from toposc_lab.robustness.disorder import (
    DisorderParameterValue,
    DisorderRealization,
    DisorderRequest,
    DisorderState,
    DisorderTarget,
    FunctionDisorderTransform,
    realize_disorder,
)

UNIFORM_ONSITE_DISORDER_KEY = "uniform_onsite_disorder"
UNIFORM_ONSITE_DISORDER_VERSION = 1


def apply_uniform_onsite_disorder(
    geometry: Geometry,
    hamiltonian: np.ndarray,
    *,
    width: float,
    seed: int,
    components_per_site: int | None = None,
    nambu_basis: NambuBasis | None = None,
) -> DisorderRealization:
    r"""Add one uniform scalar onsite offset per physical site.

    Without ``nambu_basis``, the matrix must use undoubled site-major order.
    The sampled offset is added to every internal component at its site. With
    an explicit Nambu basis, it is added to particle states and subtracted
    from their hole partners, preserving the declared BdG convention.
    """
    if not isinstance(geometry, Geometry):
        raise TypeError("geometry must be Geometry")
    width = _nonnegative_finite_real(width, name="width")
    resolved_components, representation, ordering = _resolve_basis_contract(
        geometry,
        components_per_site=components_per_site,
        nambu_basis=nambu_basis,
    )
    request = DisorderRequest(
        seed=seed,
        parameters={
            "distribution": "uniform",
            "width": width,
            "geometry_id": exact_geometry_id(geometry),
            "n_sites": geometry.n_sites,
            "representation": representation,
            "normal_components_per_site": resolved_components,
            "basis_ordering": ordering,
            "particle_hole_embedding": (
                "not_applicable" if nambu_basis is None else "opposite_sign"
            ),
        },
    )

    def transform(
        source: DisorderState,
        parameters: Mapping[str, DisorderParameterValue],
        rng: np.random.Generator,
    ) -> DisorderState:
        assert isinstance(source, np.ndarray)
        parameter_width = parameters["width"]
        if isinstance(parameter_width, bool) or not isinstance(parameter_width, Real):
            raise TypeError("recorded onsite width must be a real number")
        offsets = sample_uniform_site_disorder(
            geometry,
            width=float(parameter_width),
            rng=rng,
        )
        if nambu_basis is None:
            return _apply_normal_onsite_offsets(
                source,
                offsets,
                n_sites=geometry.n_sites,
                components_per_site=resolved_components,
            )
        return _apply_nambu_onsite_offsets(source, offsets, basis=nambu_basis)

    disorder_transform = FunctionDisorderTransform(
        key=UNIFORM_ONSITE_DISORDER_KEY,
        version=UNIFORM_ONSITE_DISORDER_VERSION,
        target=DisorderTarget.HAMILTONIAN,
        function=transform,
    )
    return realize_disorder(
        hamiltonian,
        transform=disorder_transform,
        request=request,
    )


def _resolve_basis_contract(
    geometry: Geometry,
    *,
    components_per_site: int | None,
    nambu_basis: NambuBasis | None,
) -> tuple[int, str, str]:
    if nambu_basis is None:
        resolved_components = (
            1
            if components_per_site is None
            else _positive_integer(
                components_per_site,
                name="components_per_site",
            )
        )
        return resolved_components, "normal_state", "site_major"
    if not isinstance(nambu_basis, NambuBasis):
        raise TypeError("nambu_basis must be NambuBasis or None")
    if components_per_site is not None:
        raise ValueError(
            "components_per_site must be omitted when nambu_basis is supplied"
        )
    if nambu_basis.n_sites != geometry.n_sites:
        raise ValueError("nambu_basis site count must match geometry")
    return (
        nambu_basis.normal_components_per_site,
        "nambu_bdg",
        nambu_basis.ordering,
    )


def _apply_normal_onsite_offsets(
    source: np.ndarray,
    offsets: Mapping[int, float],
    *,
    n_sites: int,
    components_per_site: int,
) -> np.ndarray:
    expected_dimension = n_sites * components_per_site
    if source.shape != (expected_dimension, expected_dimension):
        raise ValueError(
            "normal-state Hamiltonian shape does not match geometry and "
            "components_per_site"
        )
    if all(offset == 0.0 for offset in offsets.values()):
        return source.copy()
    result = np.array(
        source,
        dtype=np.result_type(source.dtype, np.float64),
        copy=True,
    )
    for site, offset in offsets.items():
        start = site * components_per_site
        for component in range(components_per_site):
            index = start + component
            result[index, index] += offset
    return result


def _apply_nambu_onsite_offsets(
    source: np.ndarray,
    offsets: Mapping[int, float],
    *,
    basis: NambuBasis,
) -> np.ndarray:
    if source.shape != (basis.dimension, basis.dimension):
        raise ValueError("BdG Hamiltonian shape does not match nambu_basis")
    if all(offset == 0.0 for offset in offsets.values()):
        return source.copy()
    result = np.array(
        source,
        dtype=np.result_type(source.dtype, np.float64),
        copy=True,
    )
    for site, offset in offsets.items():
        for component in range(basis.normal_components_per_site):
            particle = basis.particle_index(site, component=component)
            hole = basis.hole_index(site, component=component)
            result[particle, particle] += offset
            result[hole, hole] -= offset
    return result


def _nonnegative_finite_real(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be at least one")
    return result
