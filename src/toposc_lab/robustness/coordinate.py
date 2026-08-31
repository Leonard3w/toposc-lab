"""Reproducible uniform perturbation of explicit geometry coordinates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from numbers import Real

import numpy as np

from toposc_lab.evaluation.reproducibility import exact_geometry_id
from toposc_lab.geometry import Geometry, GeometryEdge
from toposc_lab.robustness.disorder import (
    DisorderParameterValue,
    DisorderRealization,
    DisorderRequest,
    DisorderState,
    DisorderTarget,
    FunctionDisorderTransform,
    realize_disorder,
)

UNIFORM_COORDINATE_PERTURBATION_KEY = "uniform_coordinate_perturbation"
UNIFORM_COORDINATE_PERTURBATION_VERSION = 1


def apply_uniform_coordinate_perturbation(
    geometry: Geometry,
    *,
    width: float,
    seed: int,
) -> DisorderRealization:
    r"""Add independent offsets from ``[-width / 2, width / 2]``.

    One offset is drawn per explicit coordinate component. Existing oriented
    edge displacements receive the corresponding target-minus-source offset,
    preserving periodic image vectors while making directions respond to the
    perturbed positions.
    """
    if not isinstance(geometry, Geometry):
        raise TypeError("geometry must be Geometry")
    width = _nonnegative_finite_real(width, name="width")
    if geometry.coordinates is None:
        raise ValueError(
            "coordinate perturbation requires explicit geometry coordinates"
        )

    request = DisorderRequest(
        seed=seed,
        parameters={
            "distribution": "uniform",
            "width": width,
            "support": "[-width/2, width/2]",
            "geometry_id": exact_geometry_id(geometry),
            "n_sites": geometry.n_sites,
            "embedding_dimension": geometry.embedding_dimension,
            "input_edge_count": geometry.n_edges,
            "sampling_rule": "independent_offset_per_coordinate_component",
            "coordinate_iteration_order": "site_major_then_embedding_axis",
            "coordinate_update_rule": "source_coordinate_plus_sampled_offset",
            "explicit_edge_displacement_policy": (
                "source_displacement_plus_target_offset_minus_source_offset"
            ),
            "implicit_edge_displacement_policy": (
                "remain_implicit_and_follow_perturbed_coordinates"
            ),
            "topology_policy": "preserve_graph_and_geometry_annotations",
            "zero_width_policy": "return_exact_source_snapshot",
        },
    )

    def transform(
        source: DisorderState,
        parameters: Mapping[str, DisorderParameterValue],
        rng: np.random.Generator,
    ) -> DisorderState:
        assert isinstance(source, Geometry)
        parameter_width = parameters["width"]
        if isinstance(parameter_width, bool) or not isinstance(
            parameter_width,
            Real,
        ):
            raise TypeError("recorded coordinate width must be a real number")
        return _perturb_coordinates(
            source,
            width=float(parameter_width),
            rng=rng,
        )

    disorder_transform = FunctionDisorderTransform(
        key=UNIFORM_COORDINATE_PERTURBATION_KEY,
        version=UNIFORM_COORDINATE_PERTURBATION_VERSION,
        target=DisorderTarget.GEOMETRY,
        function=transform,
    )
    return realize_disorder(
        geometry,
        transform=disorder_transform,
        request=request,
    )


def _perturb_coordinates(
    geometry: Geometry,
    *,
    width: float,
    rng: np.random.Generator,
) -> Geometry:
    coordinates = geometry.coordinates
    assert coordinates is not None
    if width == 0.0:
        return geometry

    offsets = rng.uniform(
        low=-0.5 * width,
        high=0.5 * width,
        size=coordinates.shape,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        perturbed_coordinates = coordinates + offsets
    if not np.all(np.isfinite(perturbed_coordinates)):
        raise ValueError("coordinate perturbation produced non-finite coordinates")

    edges = tuple(
        _edge_with_perturbed_displacement(edge, offsets=offsets)
        for edge in geometry.edges
    )
    return replace(
        geometry,
        coordinates=perturbed_coordinates,
        edges=edges,
    )


def _edge_with_perturbed_displacement(
    edge: GeometryEdge,
    *,
    offsets: np.ndarray,
) -> GeometryEdge:
    if edge.displacement is None:
        return edge
    with np.errstate(over="ignore", invalid="ignore"):
        displacement = (
            np.asarray(edge.displacement, dtype=float)
            + offsets[edge.target]
            - offsets[edge.source]
        )
    if not np.all(np.isfinite(displacement)):
        raise ValueError(
            "coordinate perturbation produced a non-finite edge displacement"
        )
    return GeometryEdge(
        source=edge.source,
        target=edge.target,
        edge_type=edge.edge_type,
        boundary_crossing=edge.boundary_crossing,
        displacement=tuple(float(value) for value in displacement),
        metadata=edge.metadata,
    )


def _nonnegative_finite_real(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result
