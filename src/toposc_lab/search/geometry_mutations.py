"""Deterministic, model-independent mutations of geometry genomes."""

from __future__ import annotations

from dataclasses import replace

from toposc_lab.geometry import GeometryEdge
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
