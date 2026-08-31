"""Reproducible independent removal of stored geometry edges."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from numbers import Real

import numpy as np

from toposc_lab.evaluation.reproducibility import exact_geometry_id
from toposc_lab.geometry import Geometry, GeometryFace
from toposc_lab.robustness.disorder import (
    DisorderParameterValue,
    DisorderRealization,
    DisorderRequest,
    DisorderState,
    DisorderTarget,
    FunctionDisorderTransform,
    realize_disorder,
)

RANDOM_EDGE_REMOVAL_KEY = "random_edge_removal"
RANDOM_EDGE_REMOVAL_VERSION = 1


def apply_random_edge_removal(
    geometry: Geometry,
    *,
    removal_probability: float,
    seed: int,
) -> DisorderRealization:
    """Remove each stored edge independently in its original iteration order."""
    if not isinstance(geometry, Geometry):
        raise TypeError("geometry must be Geometry")
    probability = _unit_interval_real(
        removal_probability,
        name="removal_probability",
    )
    request = DisorderRequest(
        seed=seed,
        parameters={
            "removal_probability": probability,
            "geometry_id": exact_geometry_id(geometry),
            "n_sites": geometry.n_sites,
            "input_edge_count": geometry.n_edges,
            "sampling_rule": "independent_bernoulli_remove_if_draw_below_probability",
            "edge_iteration_order": "stored_geometry_edges",
            "surviving_edge_rule": "preserve_object_orientation_and_order",
            "face_policy": "drop_if_any_boundary_edge_is_removed",
            "rooted_tree_policy": "clear_if_any_edge_is_removed",
        },
    )

    def transform(
        source: DisorderState,
        parameters: Mapping[str, DisorderParameterValue],
        rng: np.random.Generator,
    ) -> DisorderState:
        assert isinstance(source, Geometry)
        parameter_probability = parameters["removal_probability"]
        if isinstance(parameter_probability, bool) or not isinstance(
            parameter_probability,
            Real,
        ):
            raise TypeError("recorded removal_probability must be a real number")
        return _remove_sampled_edges(
            source,
            removal_probability=float(parameter_probability),
            rng=rng,
        )

    disorder_transform = FunctionDisorderTransform(
        key=RANDOM_EDGE_REMOVAL_KEY,
        version=RANDOM_EDGE_REMOVAL_VERSION,
        target=DisorderTarget.GEOMETRY,
        function=transform,
    )
    return realize_disorder(
        geometry,
        transform=disorder_transform,
        request=request,
    )


def _remove_sampled_edges(
    geometry: Geometry,
    *,
    removal_probability: float,
    rng: np.random.Generator,
) -> Geometry:
    draws = rng.random(geometry.n_edges)
    retained_edges = tuple(
        edge
        for edge, draw in zip(geometry.edges, draws, strict=True)
        if draw >= removal_probability
    )
    if len(retained_edges) == geometry.n_edges:
        return geometry

    retained_keys = {
        _undirected_edge_key(edge.source, edge.target)
        for edge in retained_edges
    }
    retained_faces = tuple(
        face
        for face in geometry.faces
        if _face_boundary_is_retained(face, retained_keys=retained_keys)
    )
    return replace(
        geometry,
        edges=retained_edges,
        faces=retained_faces,
        rooted_tree=None,
    )


def _face_boundary_is_retained(
    face: GeometryFace,
    *,
    retained_keys: set[tuple[int, int]],
) -> bool:
    boundary_pairs = zip(
        face.sites,
        face.sites[1:] + face.sites[:1],
        strict=True,
    )
    return all(
        _undirected_edge_key(source, target) in retained_keys
        for source, target in boundary_pairs
    )


def _undirected_edge_key(source: int, target: int) -> tuple[int, int]:
    return (source, target) if source < target else (target, source)


def _unit_interval_real(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and between zero and one")
    return result
