from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


def _as_integer(value: object, *, name: str) -> int:
    """Return an integer-valued input as ``int`` while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


@dataclass(frozen=True, slots=True)
class GeometryEdge:
    """An undirected edge with a fixed source-to-target orientation.

    The orientation does not make the graph directed. It only supplies a
    reference orientation for quantities such as complex hopping phases or an
    explicitly provided displacement vector. Instances are immutable snapshots;
    standard mutable containers in ``metadata`` are recursively frozen.
    """

    source: int
    target: int
    edge_type: str | None = None
    boundary_crossing: bool = False
    displacement: tuple[float, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        source = _as_integer(self.source, name="source")
        target = _as_integer(self.target, name="target")
        if source == target:
            raise ValueError("self-edges are not supported")

        if self.edge_type is not None and not isinstance(self.edge_type, str):
            raise TypeError("edge_type must be a string or None")

        displacement: tuple[float, ...] | None = None
        if self.displacement is not None:
            displacement_array = np.asarray(self.displacement, dtype=float)
            if displacement_array.ndim != 1 or displacement_array.size == 0:
                raise ValueError("displacement must be a non-empty one-dimensional vector")
            if not np.all(np.isfinite(displacement_array)):
                raise ValueError("displacement must contain only finite values")
            displacement = tuple(float(value) for value in displacement_array)

        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "displacement", displacement)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True, eq=False)
class Geometry:
    """Model-independent representation of a finite discrete geometry.

    ``Geometry`` stores a simple undirected graph. Site indices are contiguous
    integers from zero to ``n_sites - 1``. Spatial coordinates, labels, and
    boundary information are optional so the same type can represent embedded
    lattices as well as abstract graphs.

    Instances are immutable snapshots. Constructor inputs are defensively
    copied and standard mutable metadata containers are recursively frozen.
    Geometry-changing algorithms must construct a new instance, for example
    with :func:`dataclasses.replace`, instead of modifying an existing object.
    """

    n_sites: int
    edges: tuple[GeometryEdge, ...] = ()
    coordinates: NDArray[np.float64] | None = None
    embedding_dimension: int | None = None
    boundary_sites: frozenset[int] = frozenset()
    site_types: tuple[str | None, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _neighbors: tuple[tuple[int, ...], ...] = field(init=False, repr=False)
    _edge_lookup: Mapping[tuple[int, int], GeometryEdge] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n_sites = _as_integer(self.n_sites, name="n_sites")
        if n_sites < 1:
            raise ValueError("n_sites must be at least one")

        edges = tuple(self.edges)
        if not all(isinstance(edge, GeometryEdge) for edge in edges):
            raise TypeError("edges must contain only GeometryEdge instances")

        dimension = self._validated_dimension()
        coordinates = self._validated_coordinates(n_sites, dimension)
        if coordinates is not None:
            coordinate_dimension = int(coordinates.shape[1])
            if dimension is not None and dimension != coordinate_dimension:
                raise ValueError("embedding_dimension does not match coordinates")
            dimension = coordinate_dimension

        displacement_dimensions = {
            len(edge.displacement)
            for edge in edges
            if edge.displacement is not None
        }
        if len(displacement_dimensions) > 1:
            raise ValueError("all edge displacements must have the same dimension")
        if displacement_dimensions:
            displacement_dimension = displacement_dimensions.pop()
            if dimension is not None and dimension != displacement_dimension:
                raise ValueError("edge displacement does not match embedding_dimension")
            dimension = displacement_dimension

        boundary_sites = frozenset(
            _as_integer(site, name="boundary site") for site in self.boundary_sites
        )
        for site in boundary_sites:
            self._validate_site_for_size(site, n_sites)

        site_types = None if self.site_types is None else tuple(self.site_types)
        if site_types is not None:
            if len(site_types) != n_sites:
                raise ValueError("site_types must contain one entry per site")
            if not all(site_type is None or isinstance(site_type, str) for site_type in site_types):
                raise TypeError("site_types entries must be strings or None")

        neighbor_sets: list[set[int]] = [set() for _ in range(n_sites)]
        edge_lookup: dict[tuple[int, int], GeometryEdge] = {}
        for edge in edges:
            self._validate_site_for_size(edge.source, n_sites)
            self._validate_site_for_size(edge.target, n_sites)
            key = self._edge_key(edge.source, edge.target)
            if key in edge_lookup:
                raise ValueError(f"duplicate edge between sites {key[0]} and {key[1]}")
            edge_lookup[key] = edge
            neighbor_sets[edge.source].add(edge.target)
            neighbor_sets[edge.target].add(edge.source)

        object.__setattr__(self, "n_sites", n_sites)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "embedding_dimension", dimension)
        object.__setattr__(self, "boundary_sites", boundary_sites)
        object.__setattr__(self, "site_types", site_types)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
        object.__setattr__(
            self,
            "_neighbors",
            tuple(tuple(sorted(neighbors)) for neighbors in neighbor_sets),
        )
        object.__setattr__(self, "_edge_lookup", MappingProxyType(edge_lookup))

    @property
    def site_indices(self) -> tuple[int, ...]:
        """All valid site indices in their canonical order."""
        return tuple(range(self.n_sites))

    @property
    def n_edges(self) -> int:
        """Number of undirected edges."""
        return len(self.edges)

    @property
    def dimension(self) -> int | None:
        """Embedding dimension, or ``None`` for a purely abstract graph."""
        return self.embedding_dimension

    def neighbors(self, site: int) -> tuple[int, ...]:
        """Return the sorted direct neighbors of a site."""
        site = self._validated_site(site)
        return self._neighbors[site]

    def degree(self, site: int) -> int:
        """Return the number of direct neighbors of a site."""
        return len(self.neighbors(site))

    def edge_between(self, source: int, target: int) -> GeometryEdge | None:
        """Return the edge joining two sites, if one exists."""
        source = self._validated_site(source)
        target = self._validated_site(target)
        if source == target:
            return None
        return self._edge_lookup.get(self._edge_key(source, target))

    def has_edge(self, source: int, target: int) -> bool:
        """Return whether two sites are directly connected."""
        return self.edge_between(source, target) is not None

    def is_boundary_site(self, site: int) -> bool:
        """Return whether a site is marked as part of the boundary."""
        return self._validated_site(site) in self.boundary_sites

    def position(self, site: int) -> NDArray[np.float64]:
        """Return a copy of a site's spatial coordinates."""
        site = self._validated_site(site)
        if self.coordinates is None:
            raise ValueError("geometry has no spatial coordinates")
        return cast(NDArray[np.float64], self.coordinates[site].copy())

    def displacement_between(self, source: int, target: int) -> NDArray[np.float64]:
        """Return the oriented spatial displacement from source to target.

        An edge-specific displacement takes precedence over coordinate
        subtraction. This permits periodic boundary-crossing edges to retain a
        physically local displacement even when their endpoint coordinates are
        far apart in the chosen unit cell.
        """
        source = self._validated_site(source)
        target = self._validated_site(target)
        edge = self.edge_between(source, target)

        if edge is not None and edge.displacement is not None:
            displacement = np.asarray(edge.displacement, dtype=float)
            if source == edge.target:
                displacement = -displacement
            return displacement

        if self.coordinates is None:
            raise ValueError("no displacement or spatial coordinates are available")
        return cast(
            NDArray[np.float64],
            self.coordinates[target] - self.coordinates[source],
        )

    def distance(self, source: int, target: int) -> float:
        """Return the Euclidean distance between two sites."""
        return float(np.linalg.norm(self.displacement_between(source, target)))

    def direction(self, source: int, target: int) -> NDArray[np.float64]:
        """Return the unit vector pointing from source to target."""
        displacement = self.displacement_between(source, target)
        distance = float(np.linalg.norm(displacement))
        if distance == 0.0:
            raise ValueError("direction is undefined for zero displacement")
        return displacement / distance

    def _validated_dimension(self) -> int | None:
        if self.embedding_dimension is None:
            return None
        dimension = _as_integer(self.embedding_dimension, name="embedding_dimension")
        if dimension < 1:
            raise ValueError("embedding_dimension must be at least one")
        return dimension

    def _validated_coordinates(
        self,
        n_sites: int,
        dimension: int | None,
    ) -> NDArray[np.float64] | None:
        if self.coordinates is None:
            return None

        coordinates = np.array(self.coordinates, dtype=float, copy=True)
        if coordinates.ndim != 2:
            raise ValueError("coordinates must be a two-dimensional array")
        if coordinates.shape[0] != n_sites:
            raise ValueError("coordinates must contain one row per site")
        if coordinates.shape[1] < 1:
            raise ValueError("coordinates must have at least one column")
        if dimension is not None and coordinates.shape[1] != dimension:
            raise ValueError("embedding_dimension does not match coordinates")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("coordinates must contain only finite values")
        return _immutable_array_copy(coordinates)

    def _validated_site(self, site: int) -> int:
        site = _as_integer(site, name="site")
        self._validate_site_for_size(site, self.n_sites)
        return site

    @staticmethod
    def _validate_site_for_size(site: int, n_sites: int) -> None:
        if not 0 <= site < n_sites:
            raise ValueError(f"site {site} is outside the geometry")

    @staticmethod
    def _edge_key(source: int, target: int) -> tuple[int, int]:
        return (source, target) if source < target else (target, source)


def _freeze_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {key: _freeze_metadata_value(value) for key, value in values.items()}
    )


def _freeze_metadata_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return _freeze_metadata_value(value.tolist())
        return _immutable_array_copy(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_metadata_value(item) for item in value)
    if isinstance(value, bytearray):
        return bytes(value)
    return deepcopy(value)


def _immutable_array_copy(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values)
    immutable_buffer = contiguous.tobytes(order="C")
    return np.frombuffer(immutable_buffer, dtype=contiguous.dtype).reshape(contiguous.shape)
