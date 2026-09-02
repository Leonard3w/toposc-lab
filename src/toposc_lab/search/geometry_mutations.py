"""Deterministic, model-independent mutations of geometry genomes."""

from __future__ import annotations

from dataclasses import replace
from numbers import Integral

from toposc_lab.geometry import GeometryEdge, GeometryFace
from toposc_lab.search.geometry_genome import (
    GeometryGenome,
    geometry_from_genome,
)


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


def _stored_edge_index(value: int, *, edge_count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("edge_index must be an integer")
    result = int(value)
    if not 0 <= result < edge_count:
        raise IndexError(
            f"edge_index {result} is outside the stored edge sequence of length {edge_count}"
        )
    return result


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
