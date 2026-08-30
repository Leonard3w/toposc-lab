"""Structured validation reports for finite model-independent geometries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal, TypeAlias

import numpy as np

from toposc_lab.geometry.base import (
    Geometry,
    GeometryBoundaryComponent,
    GeometryEdge,
    GeometryFace,
)

GeometryValidationSeverity: TypeAlias = Literal["warning", "error"]
_VALIDATION_SEVERITIES = ("warning", "error")
_SUPPORTED_ARRAY_KINDS = frozenset("biufcSU")
_MAX_METADATA_DEPTH = 100


@dataclass(frozen=True, slots=True)
class GeometryValidationIssue:
    """One stable, machine-readable issue found during geometry validation."""

    code: str
    severity: GeometryValidationSeverity
    message: str
    path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code.isidentifier():
            raise ValueError("validation issue code must be a Python-style identifier")
        if self.severity not in _VALIDATION_SEVERITIES:
            raise ValueError(f"unsupported validation severity: {self.severity!r}")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("validation issue message must be a non-empty string")
        if self.path is not None and (
            not isinstance(self.path, str) or not self.path.strip()
        ):
            raise ValueError("validation issue path must be a non-empty string or None")

        object.__setattr__(self, "message", self.message.strip())
        if self.path is not None:
            object.__setattr__(self, "path", self.path.strip())


@dataclass(frozen=True, slots=True)
class GeometryValidationReport:
    """Complete validation outcome, including the graph's component partition."""

    issues: tuple[GeometryValidationIssue, ...]
    connected_components: tuple[frozenset[int], ...]
    require_connected: bool

    def __post_init__(self) -> None:
        issues = tuple(self.issues)
        if not all(isinstance(issue, GeometryValidationIssue) for issue in issues):
            raise TypeError("issues must contain only GeometryValidationIssue instances")
        components = tuple(frozenset(component) for component in self.connected_components)
        if not isinstance(self.require_connected, bool):
            raise TypeError("require_connected must be a boolean")

        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "connected_components", components)

    @property
    def errors(self) -> tuple[GeometryValidationIssue, ...]:
        """All validation errors in deterministic check order."""
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[GeometryValidationIssue, ...]:
        """All non-fatal validation warnings in deterministic check order."""
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def is_valid(self) -> bool:
        """Whether no error-level issue was found."""
        return not self.errors

    @property
    def is_connected(self) -> bool:
        """Whether the validated site set has exactly one connected component."""
        return len(self.connected_components) == 1

    def raise_for_errors(self) -> None:
        """Raise ``GeometryValidationError`` when the report contains errors."""
        if self.errors:
            raise GeometryValidationError(self)


class GeometryValidationError(ValueError):
    """Error raised when a caller explicitly rejects an invalid geometry."""

    def __init__(self, report: GeometryValidationReport) -> None:
        self.report = report
        details = "; ".join(
            f"{issue.code} at {issue.path or 'geometry'}: {issue.message}"
            for issue in report.errors
        )
        super().__init__(f"geometry validation failed: {details}")


def validate_geometry(
    geometry: Geometry,
    *,
    require_connected: bool = False,
) -> GeometryValidationReport:
    """Validate one geometry without changing or automatically rejecting it.

    ``Geometry`` already enforces its normal construction invariants. This
    pipeline repeats persistence-sensitive checks defensively, reports graph
    components, and validates metadata for deterministic exact serialization.
    Disconnected graphs produce a warning unless ``require_connected`` is true.
    """
    if not isinstance(geometry, Geometry):
        raise TypeError("geometry must be a Geometry instance")
    if not isinstance(require_connected, bool):
        raise TypeError("require_connected must be a boolean")

    structure_issues, n_sites, edge_keys = _validate_structure(geometry)
    components = _connected_components(n_sites, edge_keys)
    connectivity_issues = _connectivity_issues(
        components,
        require_connected=require_connected,
    )
    dimensional_issues = _validate_dimensions_and_coordinates(geometry)
    metadata_issues = _validate_all_metadata(geometry)
    issues = tuple(
        structure_issues
        + connectivity_issues
        + dimensional_issues
        + metadata_issues
    )
    return GeometryValidationReport(
        issues=issues,
        connected_components=components,
        require_connected=require_connected,
    )


def _validate_structure(
    geometry: Geometry,
) -> tuple[list[GeometryValidationIssue], int | None, tuple[tuple[int, int], ...]]:
    issues: list[GeometryValidationIssue] = []
    n_sites = _runtime_integer(geometry.n_sites)
    if n_sites is None or n_sites < 1:
        issues.append(
            GeometryValidationIssue(
                "invalid_site_count",
                "error",
                "n_sites must be a positive integer",
                "n_sites",
            )
        )
        n_sites = None

    edge_keys: list[tuple[int, int]] = []
    seen_edges: set[tuple[int, int]] = set()
    for edge_index, edge in enumerate(geometry.edges):
        edge_path = f"edges[{edge_index}]"
        if not isinstance(edge, GeometryEdge):
            issues.append(
                GeometryValidationIssue(
                    "invalid_edge_record",
                    "error",
                    "edge entry must be a GeometryEdge",
                    edge_path,
                )
            )
            continue
        source = _validated_index(
            edge.source,
            n_sites=n_sites,
            path=f"{edge_path}.source",
            issues=issues,
        )
        target = _validated_index(
            edge.target,
            n_sites=n_sites,
            path=f"{edge_path}.target",
            issues=issues,
        )
        if source is None or target is None:
            continue
        if not isinstance(edge.boundary_crossing, bool):
            issues.append(
                GeometryValidationIssue(
                    "invalid_boundary_crossing",
                    "error",
                    "boundary_crossing must be a boolean",
                    f"{edge_path}.boundary_crossing",
                )
            )
        if source == target:
            issues.append(
                GeometryValidationIssue(
                    "self_loop",
                    "error",
                    f"edge {source}-{target} is a self-loop",
                    edge_path,
                )
            )
            continue
        key = (source, target) if source < target else (target, source)
        if key in seen_edges:
            issues.append(
                GeometryValidationIssue(
                    "duplicate_edge",
                    "error",
                    f"edge between sites {key[0]} and {key[1]} is duplicated",
                    edge_path,
                )
            )
            continue
        seen_edges.add(key)
        edge_keys.append(key)

    _validate_associated_indices(geometry, n_sites=n_sites, issues=issues)
    return issues, n_sites, tuple(edge_keys)


def _validate_associated_indices(
    geometry: Geometry,
    *,
    n_sites: int | None,
    issues: list[GeometryValidationIssue],
) -> None:
    for site in geometry.boundary_sites:
        _validated_index(
            site,
            n_sites=n_sites,
            path=f"boundary_sites[{site!r}]",
            issues=issues,
        )

    for component_index, component in enumerate(geometry.boundary_components):
        path = f"boundary_components[{component_index}]"
        if not isinstance(component, GeometryBoundaryComponent):
            issues.append(
                GeometryValidationIssue(
                    "invalid_boundary_component",
                    "error",
                    "boundary component entry has an invalid type",
                    path,
                )
            )
            continue
        for site in component.sites:
            _validated_index(
                site,
                n_sites=n_sites,
                path=f"{path}.sites[{site!r}]",
                issues=issues,
            )

    for face_index, face in enumerate(geometry.faces):
        path = f"faces[{face_index}]"
        if not isinstance(face, GeometryFace):
            issues.append(
                GeometryValidationIssue(
                    "invalid_face_record",
                    "error",
                    "face entry must be a GeometryFace",
                    path,
                )
            )
            continue
        for site_index, site in enumerate(face.sites):
            _validated_index(
                site,
                n_sites=n_sites,
                path=f"{path}.sites[{site_index}]",
                issues=issues,
            )


def _validated_index(
    value: object,
    *,
    n_sites: int | None,
    path: str,
    issues: list[GeometryValidationIssue],
) -> int | None:
    index = _runtime_integer(value)
    if index is None or n_sites is None or not 0 <= index < n_sites:
        issues.append(
            GeometryValidationIssue(
                "invalid_site_index",
                "error",
                f"site index {value!r} is outside the geometry",
                path,
            )
        )
        return None
    return index


def _runtime_integer(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        return None
    return int(value)


def _connected_components(
    n_sites: int | None,
    edge_keys: tuple[tuple[int, int], ...],
) -> tuple[frozenset[int], ...]:
    if n_sites is None:
        return ()
    neighbors: list[set[int]] = [set() for _ in range(n_sites)]
    for source, target in edge_keys:
        neighbors[source].add(target)
        neighbors[target].add(source)

    remaining = set(range(n_sites))
    components: list[frozenset[int]] = []
    while remaining:
        root = min(remaining)
        component: set[int] = set()
        frontier = [root]
        remaining.remove(root)
        while frontier:
            site = frontier.pop()
            component.add(site)
            new_sites = sorted(neighbors[site] & remaining, reverse=True)
            for neighbor in new_sites:
                remaining.remove(neighbor)
                frontier.append(neighbor)
        components.append(frozenset(component))
    return tuple(components)


def _connectivity_issues(
    components: tuple[frozenset[int], ...],
    *,
    require_connected: bool,
) -> list[GeometryValidationIssue]:
    if len(components) <= 1:
        return []
    return [
        GeometryValidationIssue(
            "disconnected_components",
            "error" if require_connected else "warning",
            f"geometry contains {len(components)} connected components",
            "edges",
        )
    ]


def _validate_dimensions_and_coordinates(
    geometry: Geometry,
) -> list[GeometryValidationIssue]:
    issues: list[GeometryValidationIssue] = []
    declared_dimension = _runtime_integer(geometry.embedding_dimension)
    if geometry.embedding_dimension is not None and (
        declared_dimension is None or declared_dimension < 1
    ):
        issues.append(
            GeometryValidationIssue(
                "invalid_embedding_dimension",
                "error",
                "embedding_dimension must be a positive integer or None",
                "embedding_dimension",
            )
        )
        declared_dimension = None

    coordinate_dimension = _validate_coordinates(
        geometry,
        declared_dimension=declared_dimension,
        issues=issues,
    )
    displacement_dimensions = _validate_edge_displacements(geometry, issues=issues)
    dimensions = set(displacement_dimensions)
    if len(dimensions) > 1:
        issues.append(
            GeometryValidationIssue(
                "inconsistent_displacement_dimensions",
                "error",
                "edge displacements do not all have the same dimension",
                "edges",
            )
        )
    displacement_dimension = next(iter(dimensions)) if len(dimensions) == 1 else None
    effective_dimension = declared_dimension or coordinate_dimension
    if (
        effective_dimension is not None
        and displacement_dimension is not None
        and displacement_dimension != effective_dimension
    ):
        issues.append(
            GeometryValidationIssue(
                "inconsistent_embedding_dimension",
                "error",
                "edge displacement dimension does not match the embedding dimension",
                "edges",
            )
        )
    return issues


def _validate_coordinates(
    geometry: Geometry,
    *,
    declared_dimension: int | None,
    issues: list[GeometryValidationIssue],
) -> int | None:
    if geometry.coordinates is None:
        return None
    try:
        source = np.asarray(geometry.coordinates)
    except (TypeError, ValueError, OverflowError):
        issues.append(
            GeometryValidationIssue(
                "bad_coordinates",
                "error",
                "coordinates must form a finite real matrix",
                "coordinates",
            )
        )
        return None
    if np.iscomplexobj(source):
        issues.append(
            GeometryValidationIssue(
                "bad_coordinates",
                "error",
                "coordinates must be real-valued",
                "coordinates",
            )
        )
        return None
    try:
        coordinates = np.asarray(source, dtype=float)
    except (TypeError, ValueError, OverflowError):
        issues.append(
            GeometryValidationIssue(
                "bad_coordinates",
                "error",
                "coordinates must form a finite real matrix",
                "coordinates",
            )
        )
        return None
    if (
        coordinates.ndim != 2
        or coordinates.shape[0] != geometry.n_sites
        or coordinates.shape[1] < 1
        or not np.all(np.isfinite(coordinates))
    ):
        issues.append(
            GeometryValidationIssue(
                "bad_coordinates",
                "error",
                "coordinates must contain one finite, non-empty row per site",
                "coordinates",
            )
        )
        return None
    coordinate_dimension = int(coordinates.shape[1])
    if (
        declared_dimension is not None
        and coordinate_dimension != declared_dimension
    ):
        issues.append(
            GeometryValidationIssue(
                "inconsistent_embedding_dimension",
                "error",
                "coordinate columns do not match embedding_dimension",
                "coordinates",
            )
        )
    return coordinate_dimension


def _validate_edge_displacements(
    geometry: Geometry,
    *,
    issues: list[GeometryValidationIssue],
) -> tuple[int, ...]:
    dimensions: list[int] = []
    for edge_index, edge in enumerate(geometry.edges):
        if not isinstance(edge, GeometryEdge) or edge.displacement is None:
            continue
        path = f"edges[{edge_index}].displacement"
        try:
            source = np.asarray(edge.displacement)
        except (TypeError, ValueError, OverflowError):
            issues.append(
                GeometryValidationIssue(
                    "invalid_edge_displacement",
                    "error",
                    "edge displacement must be a finite real vector",
                    path,
                )
            )
            continue
        if np.iscomplexobj(source):
            issues.append(
                GeometryValidationIssue(
                    "invalid_edge_displacement",
                    "error",
                    "edge displacement must be real-valued",
                    path,
                )
            )
            continue
        try:
            displacement = np.asarray(source, dtype=float)
        except (TypeError, ValueError, OverflowError):
            issues.append(
                GeometryValidationIssue(
                    "invalid_edge_displacement",
                    "error",
                    "edge displacement must be a finite real vector",
                    path,
                )
            )
            continue
        if (
            displacement.ndim != 1
            or displacement.size < 1
            or not np.all(np.isfinite(displacement))
        ):
            issues.append(
                GeometryValidationIssue(
                    "invalid_edge_displacement",
                    "error",
                    "edge displacement must be a finite non-empty vector",
                    path,
                )
            )
            continue
        dimensions.append(int(displacement.size))
    return tuple(dimensions)


def _validate_all_metadata(geometry: Geometry) -> list[GeometryValidationIssue]:
    issues: list[GeometryValidationIssue] = []
    _validate_metadata_mapping(
        geometry.metadata,
        path="metadata",
        active_containers=set(),
        depth=0,
        issues=issues,
    )
    for edge_index, edge in enumerate(geometry.edges):
        if isinstance(edge, GeometryEdge):
            _validate_metadata_mapping(
                edge.metadata,
                path=f"edges[{edge_index}].metadata",
                active_containers=set(),
                depth=0,
                issues=issues,
            )
    for face_index, face in enumerate(geometry.faces):
        if isinstance(face, GeometryFace):
            _validate_metadata_mapping(
                face.metadata,
                path=f"faces[{face_index}].metadata",
                active_containers=set(),
                depth=0,
                issues=issues,
            )
    return issues


def _validate_metadata_mapping(
    value: Any,
    *,
    path: str,
    active_containers: set[int],
    depth: int,
    issues: list[GeometryValidationIssue],
) -> None:
    if not isinstance(value, Mapping):
        issues.append(
            GeometryValidationIssue(
                "invalid_metadata_container",
                "error",
                "metadata fields must be mappings",
                path,
            )
        )
        return
    _validate_metadata_value(
        value,
        path=path,
        active_containers=active_containers,
        depth=depth,
        issues=issues,
    )


def _validate_metadata_value(
    value: Any,
    *,
    path: str,
    active_containers: set[int],
    depth: int,
    issues: list[GeometryValidationIssue],
) -> None:
    if depth > _MAX_METADATA_DEPTH:
        issues.append(
            GeometryValidationIssue(
                "invalid_metadata_depth",
                "error",
                f"metadata nesting exceeds {_MAX_METADATA_DEPTH} levels",
                path,
            )
        )
        return
    if isinstance(value, np.generic):
        if value.dtype.kind not in _SUPPORTED_ARRAY_KINDS:
            issues.append(
                GeometryValidationIssue(
                    "invalid_metadata_scalar",
                    "error",
                    "NumPy metadata scalars must have a supported dtype",
                    path,
                )
            )
        elif value.dtype.kind in "fc" and not np.isfinite(value):
            _append_nonfinite_metadata_issue(path=path, issues=issues)
        return
    if value is None or type(value) in (bool, str, bytes, int):
        return
    if type(value) is float:
        if not np.isfinite(value):
            _append_nonfinite_metadata_issue(path=path, issues=issues)
        return
    if type(value) is complex:
        if not np.isfinite(value.real) or not np.isfinite(value.imag):
            _append_nonfinite_metadata_issue(path=path, issues=issues)
        return
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or value.dtype.kind not in _SUPPORTED_ARRAY_KINDS:
            issues.append(
                GeometryValidationIssue(
                    "invalid_metadata_array",
                    "error",
                    "metadata arrays must have a supported non-object dtype",
                    path,
                )
            )
        elif value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            _append_nonfinite_metadata_issue(path=path, issues=issues)
        return
    if isinstance(value, Mapping):
        if not _enter_metadata_container(
            value,
            path=path,
            active_containers=active_containers,
            issues=issues,
        ):
            return
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    issues.append(
                        GeometryValidationIssue(
                            "invalid_metadata_key",
                            "error",
                            "metadata mapping keys must be strings",
                            f"{path}[{key!r}]",
                        )
                    )
                    continue
                _validate_metadata_value(
                    item,
                    path=f"{path}[{key!r}]",
                    active_containers=active_containers,
                    depth=depth + 1,
                    issues=issues,
                )
        finally:
            active_containers.remove(id(value))
        return
    if isinstance(value, (list, tuple)):
        if not _enter_metadata_container(
            value,
            path=path,
            active_containers=active_containers,
            issues=issues,
        ):
            return
        try:
            for index, item in enumerate(value):
                _validate_metadata_value(
                    item,
                    path=f"{path}[{index}]",
                    active_containers=active_containers,
                    depth=depth + 1,
                    issues=issues,
                )
        finally:
            active_containers.remove(id(value))
        return
    issues.append(
        GeometryValidationIssue(
            "invalid_metadata_type",
            "error",
            f"unsupported metadata value type {type(value).__name__}",
            path,
        )
    )


def _enter_metadata_container(
    value: object,
    *,
    path: str,
    active_containers: set[int],
    issues: list[GeometryValidationIssue],
) -> bool:
    identity = id(value)
    if identity in active_containers:
        issues.append(
            GeometryValidationIssue(
                "cyclic_metadata",
                "error",
                "metadata must not contain reference cycles",
                path,
            )
        )
        return False
    active_containers.add(identity)
    return True


def _append_nonfinite_metadata_issue(
    *,
    path: str,
    issues: list[GeometryValidationIssue],
) -> None:
    issues.append(
        GeometryValidationIssue(
            "nonfinite_metadata",
            "error",
            "metadata numeric values must be finite",
            path,
        )
    )
