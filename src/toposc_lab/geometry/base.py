from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

GeometryDimensionKind: TypeAlias = Literal[
    "lattice",
    "topological",
    "hausdorff",
    "box_counting",
    "spectral",
    "walk",
]
GeometryDimensionScope: TypeAlias = Literal[
    "finite_geometry",
    "infinite_family",
]
GeometryBoundaryKind: TypeAlias = Literal["outer", "hole"]
_DIMENSION_KINDS = (
    "lattice",
    "topological",
    "hausdorff",
    "box_counting",
    "spectral",
    "walk",
)
_DIMENSION_SCOPES = ("finite_geometry", "infinite_family")
_BOUNDARY_KINDS = ("outer", "hole")


def _as_integer(value: object, *, name: str) -> int:
    """Return an integer-valued input as ``int`` while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


@dataclass(frozen=True, slots=True)
class GeometryDimension:
    """One explicitly named and sourced dimension of a geometry or family.

    This record never describes coordinate embedding; that remains
    ``Geometry.embedding_dimension``. It also must not be used implicitly as
    the physical dimension of a topology calculation.
    """

    kind: GeometryDimensionKind
    value: float
    scope: GeometryDimensionScope
    method: str
    exact: bool = False

    def __post_init__(self) -> None:
        if self.kind not in _DIMENSION_KINDS:
            raise ValueError(f"unsupported geometry dimension kind: {self.kind!r}")
        if isinstance(self.value, bool) or not isinstance(self.value, Real):
            raise TypeError("geometry dimension value must be a real number")
        value = float(self.value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("geometry dimension value must be finite and nonnegative")
        if self.scope not in _DIMENSION_SCOPES:
            raise ValueError(f"unsupported geometry dimension scope: {self.scope!r}")
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError("geometry dimension method must be a non-empty string")
        if not isinstance(self.exact, bool):
            raise TypeError("geometry dimension exact flag must be a boolean")

        object.__setattr__(self, "value", value)
        object.__setattr__(self, "method", self.method.strip())


@dataclass(frozen=True, slots=True)
class GeometryBoundaryComponent:
    """Sites adjacent to one geometrically distinct boundary component.

    Components may overlap at a site. This is useful for discrete fractals,
    where one retained cell can touch more than one distinct hole. The
    aggregate set remains available as ``Geometry.boundary_sites``. The sites
    need not induce a connected subgraph: cell centers bordering consecutive
    sides of a hole can meet only diagonally.
    """

    kind: GeometryBoundaryKind
    component_index: int
    sites: frozenset[int]

    def __post_init__(self) -> None:
        if self.kind not in _BOUNDARY_KINDS:
            raise ValueError(f"unsupported geometry boundary kind: {self.kind!r}")
        component_index = _as_integer(
            self.component_index,
            name="boundary component index",
        )
        if component_index < 0:
            raise ValueError("boundary component index must be nonnegative")
        sites = frozenset(
            _as_integer(site, name="boundary component site") for site in self.sites
        )
        if not sites:
            raise ValueError("boundary component sites must not be empty")

        object.__setattr__(self, "component_index", component_index)
        object.__setattr__(self, "sites", sites)


@dataclass(frozen=True, slots=True)
class RootedTreeStructure:
    """Validated rooted hierarchy associated with a tree geometry."""

    root_site: int
    parents: tuple[int | None, ...]
    _children: tuple[tuple[int, ...], ...] = field(init=False, repr=False)
    _depths: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        parents = tuple(self.parents)
        if not parents:
            raise ValueError("rooted tree parents must contain at least one site")
        root_site = _as_integer(self.root_site, name="root site")
        if not 0 <= root_site < len(parents):
            raise ValueError("root site is outside the rooted tree")

        normalized_parents: list[int | None] = []
        for site, parent in enumerate(parents):
            if parent is None:
                if site != root_site:
                    raise ValueError("only the root site may have no parent")
                normalized_parents.append(None)
                continue
            normalized_parent = _as_integer(parent, name="tree parent")
            if not 0 <= normalized_parent < len(parents):
                raise ValueError("tree parent is outside the rooted tree")
            if normalized_parent == site:
                raise ValueError("a tree site cannot be its own parent")
            normalized_parents.append(normalized_parent)
        if normalized_parents[root_site] is not None:
            raise ValueError("root site must have no parent")

        children: list[list[int]] = [[] for _ in parents]
        for site, parent in enumerate(normalized_parents):
            if parent is not None:
                children[parent].append(site)

        depths: list[int] = [-1] * len(parents)
        depths[root_site] = 0
        for site in range(len(parents)):
            if depths[site] >= 0:
                continue
            current = site
            path: list[int] = []
            path_positions: dict[int, int] = {}
            while depths[current] < 0:
                if current in path_positions:
                    raise ValueError("rooted tree parents contain a cycle")
                path_positions[current] = len(path)
                path.append(current)
                parent = normalized_parents[current]
                if parent is None:
                    raise ValueError("tree site does not descend from the root")
                current = parent

            depth = depths[current]
            for path_site in reversed(path):
                depth += 1
                depths[path_site] = depth

        object.__setattr__(self, "root_site", root_site)
        object.__setattr__(self, "parents", tuple(normalized_parents))
        object.__setattr__(
            self,
            "_children",
            tuple(tuple(site_children) for site_children in children),
        )
        object.__setattr__(self, "_depths", tuple(depths))

    @property
    def n_sites(self) -> int:
        """Number of sites in the hierarchy."""
        return len(self.parents)

    @property
    def leaf_sites(self) -> frozenset[int]:
        """Sites with no children, including a singleton root."""
        return frozenset(
            site for site, children in enumerate(self._children) if not children
        )

    def parent(self, site: int) -> int | None:
        """Return the parent of ``site`` or ``None`` for the root."""
        return self.parents[self._validated_site(site)]

    def children(self, site: int) -> tuple[int, ...]:
        """Return children of ``site`` in ascending site order."""
        return self._children[self._validated_site(site)]

    def depth(self, site: int) -> int:
        """Return the number of parent edges from the root to ``site``."""
        return self._depths[self._validated_site(site)]

    def _validated_site(self, site: int) -> int:
        result = _as_integer(site, name="site")
        if not 0 <= result < self.n_sites:
            raise ValueError(f"site {result} is outside the rooted tree")
        return result


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


@dataclass(frozen=True, slots=True)
class GeometryFace:
    """An ordered polygonal face bounded by geometry edges.

    ``sites`` follows the polygon boundary in either orientation. The enclosing
    :class:`Geometry` validates that every consecutive pair, including the
    closing pair, has a corresponding graph edge.
    """

    sites: tuple[int, ...]
    face_type: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        sites = tuple(_as_integer(site, name="face site") for site in self.sites)
        if len(sites) < 3:
            raise ValueError("a geometry face must contain at least three sites")
        if len(set(sites)) != len(sites):
            raise ValueError("a geometry face cannot repeat a site")
        if self.face_type is not None and not isinstance(self.face_type, str):
            raise TypeError("face_type must be a string or None")

        object.__setattr__(self, "sites", sites)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True, eq=False)
class Geometry:
    """Model-independent representation of a finite discrete geometry.

    ``Geometry`` stores a simple undirected graph. Site indices are contiguous
    integers from zero to ``n_sites - 1``. Spatial coordinates, labels, and
    boundary information are optional so the same type can represent embedded
    lattices as well as abstract graphs. Optional ordered polygonal faces extend
    the graph's one-skeleton into a validated cell complex when a generator has
    meaningful plaquettes or tiles.

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
    boundary_components: tuple[GeometryBoundaryComponent, ...] = ()
    site_types: tuple[str | None, ...] | None = None
    dimension_records: tuple[GeometryDimension, ...] = ()
    rooted_tree: RootedTreeStructure | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    faces: tuple[GeometryFace, ...] = ()
    _neighbors: tuple[tuple[int, ...], ...] = field(init=False, repr=False)
    _edge_lookup: Mapping[tuple[int, int], GeometryEdge] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n_sites = _as_integer(self.n_sites, name="n_sites")
        if n_sites < 1:
            raise ValueError("n_sites must be at least one")

        edges = tuple(self.edges)
        if not all(isinstance(edge, GeometryEdge) for edge in edges):
            raise TypeError("edges must contain only GeometryEdge instances")
        faces = tuple(self.faces)
        if not all(isinstance(face, GeometryFace) for face in faces):
            raise TypeError("faces must contain only GeometryFace instances")

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

        boundary_components = tuple(self.boundary_components)
        if not all(
            isinstance(component, GeometryBoundaryComponent)
            for component in boundary_components
        ):
            raise TypeError(
                "boundary_components must contain only "
                "GeometryBoundaryComponent instances"
            )
        component_keys = tuple(
            (component.kind, component.component_index)
            for component in boundary_components
        )
        if len(set(component_keys)) != len(component_keys):
            raise ValueError("boundary_components must have unique kind/index pairs")
        component_sites = frozenset(
            site
            for component in boundary_components
            for site in component.sites
        )
        for site in component_sites:
            self._validate_site_for_size(site, n_sites)
        if boundary_components and component_sites != boundary_sites:
            raise ValueError(
                "boundary_sites must equal the union of boundary component sites"
            )

        site_types = None if self.site_types is None else tuple(self.site_types)
        if site_types is not None:
            if len(site_types) != n_sites:
                raise ValueError("site_types must contain one entry per site")
            if not all(site_type is None or isinstance(site_type, str) for site_type in site_types):
                raise TypeError("site_types entries must be strings or None")

        dimension_records = tuple(self.dimension_records)
        if not all(
            isinstance(record, GeometryDimension) for record in dimension_records
        ):
            raise TypeError(
                "dimension_records must contain only GeometryDimension instances"
            )
        dimension_keys = tuple(
            (record.kind, record.scope, record.method)
            for record in dimension_records
        )
        if len(set(dimension_keys)) != len(dimension_keys):
            raise ValueError("dimension_records must not contain duplicate records")

        rooted_tree = self.rooted_tree
        if rooted_tree is not None:
            if not isinstance(rooted_tree, RootedTreeStructure):
                raise TypeError("rooted_tree must be a RootedTreeStructure or None")
            if rooted_tree.n_sites != n_sites:
                raise ValueError("rooted_tree must contain one parent per geometry site")

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

        face_keys: set[frozenset[int]] = set()
        for face in faces:
            for site in face.sites:
                self._validate_site_for_size(site, n_sites)
            face_key = frozenset(face.sites)
            if face_key in face_keys:
                raise ValueError("duplicate geometry face")
            face_keys.add(face_key)
            boundary_pairs = zip(
                face.sites,
                face.sites[1:] + face.sites[:1],
                strict=True,
            )
            for source, target in boundary_pairs:
                if self._edge_key(source, target) not in edge_lookup:
                    raise ValueError(
                        "geometry face boundary must consist of geometry edges"
                    )

        if rooted_tree is not None:
            rooted_edge_keys = {
                self._edge_key(site, parent)
                for site, parent in enumerate(rooted_tree.parents)
                if parent is not None
            }
            if set(edge_lookup) != rooted_edge_keys:
                raise ValueError(
                    "geometry edges must exactly match rooted_tree parent edges"
                )

        object.__setattr__(self, "n_sites", n_sites)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "embedding_dimension", dimension)
        object.__setattr__(self, "boundary_sites", boundary_sites)
        object.__setattr__(self, "boundary_components", boundary_components)
        object.__setattr__(self, "site_types", site_types)
        object.__setattr__(self, "dimension_records", dimension_records)
        object.__setattr__(self, "rooted_tree", rooted_tree)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
        object.__setattr__(self, "faces", faces)
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
    def n_faces(self) -> int:
        """Number of explicitly stored polygonal faces."""
        return len(self.faces)

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

        if np.iscomplexobj(np.asarray(self.coordinates)):
            raise ValueError("complex coordinates are not supported")
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
