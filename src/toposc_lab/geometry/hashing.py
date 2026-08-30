"""Canonical relabeling-invariant fingerprints for finite geometry graphs."""

from __future__ import annotations

import hashlib
from collections import Counter
from numbers import Integral
from typing import Protocol, TypeAlias

from toposc_lab.geometry.base import Geometry, GeometryEdge
from toposc_lab.geometry.validation import validate_geometry

CANONICAL_GRAPH_HASH_ALGORITHM = "wl1-v1-sha256"
DEFAULT_CANONICAL_GRAPH_HASH_MAX_REFINEMENT_ROUNDS = 64

_RefinementNeighbor: TypeAlias = tuple[bytes, int]
_RefinementSignature: TypeAlias = tuple[int, tuple[_RefinementNeighbor, ...]]
_ColoredEdgeSignature: TypeAlias = tuple[int, int, bytes]


class _Hasher(Protocol):
    def update(self, data: bytes) -> None: ...

    def hexdigest(self) -> str: ...


def canonical_graph_hash(
    geometry: Geometry,
    *,
    include_site_types: bool = False,
    include_edge_types: bool = False,
    include_boundary: bool = False,
    max_refinement_rounds: int = (
        DEFAULT_CANONICAL_GRAPH_HASH_MAX_REFINEMENT_ROUNDS
    ),
) -> str:
    """Return a versioned, site-relabeling-invariant graph fingerprint.

    The hash applies one-dimensional Weisfeiler--Leman color refinement and is
    suitable for grouping possible graph-isomorphic duplicates. Equality is
    not an isomorphism proof: non-isomorphic graphs can share a 1-WL hash and
    must be resolved by an exact isomorphism check before deduplication.

    Coordinates, displacements, faces, dimension records, rooted-tree roots,
    and metadata are intentionally excluded. Optional flags add site types,
    edge types, and simple boundary membership/crossing markers respectively.
    """
    if not isinstance(geometry, Geometry):
        raise TypeError("geometry must be a Geometry instance")
    include_site_types = _boolean_option(
        include_site_types,
        name="include_site_types",
    )
    include_edge_types = _boolean_option(
        include_edge_types,
        name="include_edge_types",
    )
    include_boundary = _boolean_option(
        include_boundary,
        name="include_boundary",
    )
    max_refinement_rounds = _positive_integer(
        max_refinement_rounds,
        name="max_refinement_rounds",
    )
    report = validate_geometry(geometry)
    report.raise_for_errors()

    initial_signatures = tuple(
        _site_signature(
            geometry,
            site,
            include_site_types=include_site_types,
            include_boundary=include_boundary,
        )
        for site in geometry.site_indices
    )
    edge_labels = tuple(
        _edge_signature(
            edge,
            include_edge_types=include_edge_types,
            include_boundary=include_boundary,
        )
        for edge in geometry.edges
    )
    incident_edges: list[list[tuple[int, bytes]]] = [
        [] for _ in geometry.site_indices
    ]
    for edge, edge_label in zip(geometry.edges, edge_labels, strict=True):
        incident_edges[edge.source].append((edge.target, edge_label))
        incident_edges[edge.target].append((edge.source, edge_label))

    hasher = hashlib.sha256()
    _update_token(hasher, b"toposc_lab.canonical_graph_hash")
    _update_token(hasher, CANONICAL_GRAPH_HASH_ALGORITHM.encode("ascii"))
    _update_boolean(hasher, include_site_types)
    _update_boolean(hasher, include_edge_types)
    _update_boolean(hasher, include_boundary)
    _update_integer(hasher, geometry.n_sites)
    _update_integer(hasher, geometry.n_edges)
    component_sizes = tuple(
        sorted(len(component) for component in report.connected_components)
    )
    _update_integer_sequence(hasher, component_sizes)
    _update_bytes_groups(hasher, initial_signatures)
    _update_bytes_groups(hasher, edge_labels)

    colors = _compress_bytes(initial_signatures)
    rounds_performed = 0
    stabilized = False
    for _ in range(max_refinement_rounds):
        refinement_signatures = tuple(
            (
                colors[site],
                tuple(
                    sorted(
                        (edge_label, colors[neighbor])
                        for neighbor, edge_label in incident_edges[site]
                    )
                ),
            )
            for site in geometry.site_indices
        )
        _update_refinement_groups(hasher, refinement_signatures)
        refined_colors = _compress_refinement(refinement_signatures)
        rounds_performed += 1
        if len(set(refined_colors)) == len(set(colors)):
            colors = refined_colors
            stabilized = True
            break
        colors = refined_colors

    _update_integer(hasher, rounds_performed)
    _update_boolean(hasher, stabilized)
    colored_edges = tuple(
        (
            min(colors[edge.source], colors[edge.target]),
            max(colors[edge.source], colors[edge.target]),
            edge_label,
        )
        for edge, edge_label in zip(geometry.edges, edge_labels, strict=True)
    )
    _update_colored_edge_groups(hasher, colored_edges)
    return f"{CANONICAL_GRAPH_HASH_ALGORITHM}:{hasher.hexdigest()}"


def _site_signature(
    geometry: Geometry,
    site: int,
    *,
    include_site_types: bool,
    include_boundary: bool,
) -> bytes:
    parts = [b"site"]
    if include_site_types:
        site_type = None if geometry.site_types is None else geometry.site_types[site]
        parts.append(_optional_string_signature(site_type))
    if include_boundary:
        parts.append(b"boundary:1" if site in geometry.boundary_sites else b"boundary:0")
    return _encode_parts(parts)


def _edge_signature(
    edge: GeometryEdge,
    *,
    include_edge_types: bool,
    include_boundary: bool,
) -> bytes:
    parts = [b"edge"]
    if include_edge_types:
        parts.append(_optional_string_signature(edge.edge_type))
    if include_boundary:
        parts.append(b"crossing:1" if edge.boundary_crossing else b"crossing:0")
    return _encode_parts(parts)


def _optional_string_signature(value: str | None) -> bytes:
    if value is None:
        return b"none"
    return _encode_parts((b"string", value.encode("utf-8")))


def _encode_parts(parts: tuple[bytes, ...] | list[bytes]) -> bytes:
    encoded = bytearray()
    for part in parts:
        encoded.extend(len(part).to_bytes(8, byteorder="big", signed=False))
        encoded.extend(part)
    return bytes(encoded)


def _compress_bytes(signatures: tuple[bytes, ...]) -> tuple[int, ...]:
    canonical_signatures = {
        signature: color
        for color, signature in enumerate(sorted(set(signatures)))
    }
    return tuple(canonical_signatures[signature] for signature in signatures)


def _compress_refinement(
    signatures: tuple[_RefinementSignature, ...],
) -> tuple[int, ...]:
    canonical_signatures = {
        signature: color
        for color, signature in enumerate(sorted(set(signatures)))
    }
    return tuple(canonical_signatures[signature] for signature in signatures)


def _update_bytes_groups(
    hasher: _Hasher,
    signatures: tuple[bytes, ...],
) -> None:
    counts = Counter(signatures)
    _update_integer(hasher, len(counts))
    for signature in sorted(counts):
        _update_token(hasher, signature)
        _update_integer(hasher, counts[signature])


def _update_refinement_groups(
    hasher: _Hasher,
    signatures: tuple[_RefinementSignature, ...],
) -> None:
    counts = Counter(signatures)
    _update_integer(hasher, len(counts))
    for signature in sorted(counts):
        color, neighbors = signature
        _update_integer(hasher, color)
        _update_integer(hasher, len(neighbors))
        for edge_label, neighbor_color in neighbors:
            _update_token(hasher, edge_label)
            _update_integer(hasher, neighbor_color)
        _update_integer(hasher, counts[signature])


def _update_colored_edge_groups(
    hasher: _Hasher,
    signatures: tuple[_ColoredEdgeSignature, ...],
) -> None:
    counts = Counter(signatures)
    _update_integer(hasher, len(counts))
    for source_color, target_color, edge_label in sorted(counts):
        _update_integer(hasher, source_color)
        _update_integer(hasher, target_color)
        _update_token(hasher, edge_label)
        _update_integer(
            hasher,
            counts[(source_color, target_color, edge_label)],
        )


def _update_token(hasher: _Hasher, value: bytes) -> None:
    _update_integer(hasher, len(value))
    hasher.update(value)


def _update_integer_sequence(hasher: _Hasher, values: tuple[int, ...]) -> None:
    _update_integer(hasher, len(values))
    for value in values:
        _update_integer(hasher, value)


def _update_integer(hasher: _Hasher, value: int) -> None:
    hasher.update(value.to_bytes(8, byteorder="big", signed=False))


def _update_boolean(hasher: _Hasher, value: bool) -> None:
    hasher.update(b"\x01" if value else b"\x00")


def _boolean_option(value: bool, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result
