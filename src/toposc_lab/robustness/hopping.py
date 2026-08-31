"""Concrete additive hopping disorder on finite Hamiltonian matrices."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real

import numpy as np

from toposc_lab.evaluation.reproducibility import exact_geometry_id
from toposc_lab.geometry import Geometry, GeometryEdge
from toposc_lab.hamiltonians.disorder import sample_uniform_edge_disorder
from toposc_lab.hamiltonians.nambu import NambuBasis
from toposc_lab.robustness._matrix_disorder import (
    nonnegative_finite_real,
    resolve_matrix_disorder_basis,
)
from toposc_lab.robustness.disorder import (
    DisorderParameterValue,
    DisorderRealization,
    DisorderRequest,
    DisorderState,
    DisorderTarget,
    FunctionDisorderTransform,
    realize_disorder,
)

UNIFORM_HOPPING_DISORDER_KEY = "uniform_hopping_disorder"
UNIFORM_HOPPING_DISORDER_VERSION = 1


def apply_uniform_hopping_disorder(
    geometry: Geometry,
    hamiltonian: np.ndarray,
    *,
    width: float,
    seed: int,
    components_per_site: int | None = None,
    nambu_basis: NambuBasis | None = None,
) -> DisorderRealization:
    r"""Add one real uniform hopping offset per stored oriented edge.

    The forward block follows each ``GeometryEdge`` source-to-target
    orientation and the reverse block receives its Hermitian conjugate. With
    an explicit Nambu basis, the normal hopping perturbation is embedded with
    opposite particle/hole signs.
    """
    if not isinstance(geometry, Geometry):
        raise TypeError("geometry must be Geometry")
    width = nonnegative_finite_real(width, name="width")
    basis_contract = resolve_matrix_disorder_basis(
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
            "n_edges": geometry.n_edges,
            "representation": basis_contract.representation,
            "normal_components_per_site": (
                basis_contract.normal_components_per_site
            ),
            "basis_ordering": basis_contract.ordering,
            "particle_hole_embedding": basis_contract.particle_hole_embedding,
            "edge_application": (
                "oriented_forward_plus_hermitian_conjugate"
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
            raise TypeError("recorded hopping width must be a real number")
        offsets = sample_uniform_edge_disorder(
            geometry,
            width=float(parameter_width),
            rng=rng,
        )
        if basis_contract.nambu_basis is None:
            return _apply_normal_hopping_offsets(
                source,
                offsets,
                n_sites=geometry.n_sites,
                components_per_site=(
                    basis_contract.normal_components_per_site
                ),
            )
        return _apply_nambu_hopping_offsets(
            source,
            offsets,
            basis=basis_contract.nambu_basis,
        )

    disorder_transform = FunctionDisorderTransform(
        key=UNIFORM_HOPPING_DISORDER_KEY,
        version=UNIFORM_HOPPING_DISORDER_VERSION,
        target=DisorderTarget.HAMILTONIAN,
        function=transform,
    )
    return realize_disorder(
        hamiltonian,
        transform=disorder_transform,
        request=request,
    )


def _apply_normal_hopping_offsets(
    source: np.ndarray,
    offsets: Mapping[GeometryEdge, float],
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
    for edge, offset in offsets.items():
        source_start = edge.source * components_per_site
        target_start = edge.target * components_per_site
        for component in range(components_per_site):
            source_index = source_start + component
            target_index = target_start + component
            result[source_index, target_index] += offset
            result[target_index, source_index] += offset
    return result


def _apply_nambu_hopping_offsets(
    source: np.ndarray,
    offsets: Mapping[GeometryEdge, float],
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
    for edge, offset in offsets.items():
        for component in range(basis.normal_components_per_site):
            particle_source = basis.particle_index(edge.source, component=component)
            particle_target = basis.particle_index(edge.target, component=component)
            hole_source = basis.hole_index(edge.source, component=component)
            hole_target = basis.hole_index(edge.target, component=component)
            result[particle_source, particle_target] += offset
            result[particle_target, particle_source] += offset
            result[hole_source, hole_target] -= offset
            result[hole_target, hole_source] -= offset
    return result
