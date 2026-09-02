"""Deterministic, model-independent mutations of geometry genomes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace
from numbers import Integral
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from toposc_lab.geometry import (
    GeometryBoundaryComponent,
    GeometryBoundaryKind,
    GeometryEdge,
    GeometryFace,
)
from toposc_lab.search.geometry_genome import (
    GeometryGenome,
    geometry_from_genome,
)

NodeCoordinate: TypeAlias = Sequence[float] | NDArray[np.float64]
BoundaryComponentKey: TypeAlias = tuple[GeometryBoundaryKind, int]


def add_edge_mutation(
    genome: GeometryGenome,
    edge: GeometryEdge,
) -> GeometryGenome:
    """Return a new genome with one fully specified oriented edge appended.

    The caller owns endpoint selection and every optional edge attribute. This
    primitive neither samples endpoints nor infers displacement, boundary, or
    physical information. Existing edge order and orientation are preserved.

    Representation-invalid results such as duplicate undirected edges or
    endpoints outside the genome are rejected through the established genome
    and production-geometry validation boundary. Search-space and scientific
    constraints remain outside this Phase-10.2 operation.
    """
    if not isinstance(genome, GeometryGenome):
        raise TypeError("genome must be a GeometryGenome instance")
    if not isinstance(edge, GeometryEdge):
        raise TypeError("edge must be a GeometryEdge instance")

    geometry_from_genome(genome)
    mutated = replace(
        genome,
        edges=genome.edges + (edge,),
        rooted_tree=None,
    )
    geometry_from_genome(mutated)
    return mutated


def remove_edge_mutation(
    genome: GeometryGenome,
    edge_index: int,
) -> GeometryGenome:
    """Return a new genome without the edge at one stored sequence index.

    Every surviving edge retains its exact object state, orientation, and
    relative order. Faces whose polygon boundary uses the removed undirected
    edge are dropped; unrelated faces and all non-tree geometry data remain
    unchanged. The operation performs no edge sampling or search-space checks.
    """
    if not isinstance(genome, GeometryGenome):
        raise TypeError("genome must be a GeometryGenome instance")
    prepared_index = _stored_edge_index(edge_index, edge_count=len(genome.edges))

    geometry_from_genome(genome)
    removed_edge = genome.edges[prepared_index]
    retained_edges = genome.edges[:prepared_index] + genome.edges[prepared_index + 1 :]
    retained_faces = tuple(
        face for face in genome.faces if not _face_uses_edge(face, edge=removed_edge)
    )
    mutated = replace(
        genome,
        edges=retained_edges,
        faces=retained_faces,
        rooted_tree=None,
    )
    geometry_from_genome(mutated)
    return mutated


def add_node_mutation(
    genome: GeometryGenome,
    *,
    coordinate: NodeCoordinate | None = None,
    site_type: str | None = None,
    boundary: bool = False,
    boundary_component_keys: Iterable[BoundaryComponentKey] = (),
) -> GeometryGenome:
    """Return a new genome with one explicitly described isolated site.

    The site receives index ``genome.n_sites``. Coordinate and site-type columns
    remain complete when present, while missing columns are never synthesized
    for older sites. Boundary membership is caller-declared and may reference
    only existing boundary components. No edge or physical interpretation is
    added implicitly.
    """
    if not isinstance(genome, GeometryGenome):
        raise TypeError("genome must be a GeometryGenome instance")
    geometry_from_genome(genome)

    coordinates = _coordinates_with_added_site(genome, coordinate=coordinate)
    site_types = _site_types_with_added_site(genome, site_type=site_type)
    component_keys = _boundary_component_key_set(boundary_component_keys)
    boundary_sites, boundary_components = _boundaries_with_added_site(
        genome,
        boundary=boundary,
        component_keys=component_keys,
    )
    mutated = replace(
        genome,
        n_sites=genome.n_sites + 1,
        coordinates=coordinates,
        boundary_sites=boundary_sites,
        boundary_components=boundary_components,
        site_types=site_types,
        rooted_tree=None,
    )
    geometry_from_genome(mutated)
    return mutated


def remove_node_mutation(
    genome: GeometryGenome,
    site_index: int,
) -> GeometryGenome:
    """Return a new genome without one stored site and its incident data.

    Surviving sites retain their original order and are compactly reindexed.
    Surviving edges keep their source-to-target orientation and every non-endpoint
    attribute. Faces containing the removed site and empty boundary components
    are dropped. No replacement boundary, edge, coordinate, or physical meaning
    is inferred.
    """
    if not isinstance(genome, GeometryGenome):
        raise TypeError("genome must be a GeometryGenome instance")
    prepared_index = _stored_site_index(site_index, site_count=genome.n_sites)

    geometry_from_genome(genome)
    if genome.n_sites == 1:
        raise ValueError("cannot remove the only site from a geometry genome")

    surviving_edges = tuple(
        replace(
            edge,
            source=_site_after_removal(edge.source, removed_site=prepared_index),
            target=_site_after_removal(edge.target, removed_site=prepared_index),
        )
        for edge in genome.edges
        if edge.source != prepared_index and edge.target != prepared_index
    )
    coordinates = (
        None
        if genome.coordinates is None
        else np.delete(genome.coordinates, prepared_index, axis=0)
    )
    boundary_sites = frozenset(
        _site_after_removal(site, removed_site=prepared_index)
        for site in genome.boundary_sites
        if site != prepared_index
    )
    boundary_components = tuple(
        replace(
            component,
            sites=frozenset(
                _site_after_removal(site, removed_site=prepared_index)
                for site in component.sites
                if site != prepared_index
            ),
        )
        for component in genome.boundary_components
        if component.sites != frozenset((prepared_index,))
    )
    site_types = (
        None
        if genome.site_types is None
        else genome.site_types[:prepared_index] + genome.site_types[prepared_index + 1 :]
    )
    surviving_faces = tuple(
        replace(
            face,
            sites=tuple(
                _site_after_removal(site, removed_site=prepared_index)
                for site in face.sites
            ),
        )
        for face in genome.faces
        if prepared_index not in face.sites
    )
    mutated = replace(
        genome,
        n_sites=genome.n_sites - 1,
        edges=surviving_edges,
        coordinates=coordinates,
        boundary_sites=boundary_sites,
        boundary_components=boundary_components,
        site_types=site_types,
        rooted_tree=None,
        faces=surviving_faces,
    )
    geometry_from_genome(mutated)
    return mutated


def _stored_edge_index(value: int, *, edge_count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("edge_index must be an integer")
    result = int(value)
    if not 0 <= result < edge_count:
        raise IndexError(
            f"edge_index {result} is outside the stored edge sequence of length {edge_count}"
        )
    return result


def _stored_site_index(value: int, *, site_count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("site_index must be an integer")
    result = int(value)
    if not 0 <= result < site_count:
        raise IndexError(
            f"site_index {result} is outside the stored site sequence of length {site_count}"
        )
    return result


def _site_after_removal(site: int, *, removed_site: int) -> int:
    return site if site < removed_site else site - 1


def _coordinates_with_added_site(
    genome: GeometryGenome,
    *,
    coordinate: NodeCoordinate | None,
) -> NDArray[np.float64] | None:
    if genome.coordinates is None:
        if coordinate is not None:
            raise ValueError("coordinate must be None when the genome has no coordinate table")
        return None
    if coordinate is None:
        raise ValueError("coordinate is required when the genome has a coordinate table")
    raw_coordinate = np.asarray(coordinate)
    if np.iscomplexobj(raw_coordinate):
        raise ValueError("coordinate must contain real values")
    try:
        values = np.asarray(coordinate, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("coordinate must be a finite real vector") from error
    expected_shape = (genome.coordinates.shape[1],)
    if values.shape != expected_shape:
        raise ValueError(f"coordinate must have shape {expected_shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("coordinate must contain only finite values")
    return np.vstack((genome.coordinates, values))


def _site_types_with_added_site(
    genome: GeometryGenome,
    *,
    site_type: str | None,
) -> tuple[str | None, ...] | None:
    if site_type is not None and not isinstance(site_type, str):
        raise TypeError("site_type must be a string or None")
    if genome.site_types is None:
        if site_type is not None:
            raise ValueError("site_type must be None when the genome has no site-type column")
        return None
    return genome.site_types + (site_type,)


def _boundary_component_key_set(
    values: Iterable[BoundaryComponentKey],
) -> frozenset[BoundaryComponentKey]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Iterable):
        raise TypeError("boundary_component_keys must be an iterable of component keys")
    keys: list[BoundaryComponentKey] = []
    for value in values:
        if (
            isinstance(value, (str, bytes, bytearray))
            or not isinstance(value, Sequence)
            or len(value) != 2
        ):
            raise TypeError("boundary component keys must be (kind, component_index) pairs")
        kind, component_index = value
        if kind not in ("outer", "hole"):
            raise ValueError(f"unsupported boundary component kind: {kind!r}")
        if isinstance(component_index, bool) or not isinstance(component_index, Integral):
            raise TypeError("boundary component index must be an integer")
        normalized_index = int(component_index)
        if normalized_index < 0:
            raise ValueError("boundary component index must be nonnegative")
        keys.append((cast(GeometryBoundaryKind, kind), normalized_index))
    if len(set(keys)) != len(keys):
        raise ValueError("boundary_component_keys must not contain duplicates")
    return frozenset(keys)


def _boundaries_with_added_site(
    genome: GeometryGenome,
    *,
    boundary: bool,
    component_keys: frozenset[BoundaryComponentKey],
) -> tuple[frozenset[int], tuple[GeometryBoundaryComponent, ...]]:
    if not isinstance(boundary, bool):
        raise TypeError("boundary must be a boolean")
    if not boundary and component_keys:
        raise ValueError("boundary_component_keys require boundary=True")

    existing_keys = frozenset(
        (component.kind, component.component_index) for component in genome.boundary_components
    )
    if component_keys - existing_keys:
        unknown = sorted(component_keys - existing_keys)
        raise ValueError(f"unknown boundary component keys: {unknown!r}")
    if not genome.boundary_components and component_keys:
        raise ValueError("a component key requires existing boundary components")
    if boundary and genome.boundary_components and not component_keys:
        raise ValueError("a boundary site requires component keys when components are present")

    new_site = genome.n_sites
    boundary_sites = (
        genome.boundary_sites | frozenset((new_site,)) if boundary else genome.boundary_sites
    )
    boundary_components = tuple(
        replace(
            component,
            sites=component.sites | frozenset((new_site,)),
        )
        if (component.kind, component.component_index) in component_keys
        else component
        for component in genome.boundary_components
    )
    return boundary_sites, boundary_components


def _face_uses_edge(face: GeometryFace, *, edge: GeometryEdge) -> bool:
    removed_key = _undirected_edge_key(edge.source, edge.target)
    boundary_pairs = zip(
        face.sites,
        face.sites[1:] + face.sites[:1],
        strict=True,
    )
    return any(
        _undirected_edge_key(source, target) == removed_key for source, target in boundary_pairs
    )


def _undirected_edge_key(source: int, target: int) -> tuple[int, int]:
    return (source, target) if source < target else (target, source)
