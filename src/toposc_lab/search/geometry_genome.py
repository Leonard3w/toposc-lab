"""Lossless, model-independent genome snapshots for geometry search."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from toposc_lab.geometry import (
    Geometry,
    GeometryBoundaryComponent,
    GeometryDimension,
    GeometryEdge,
    GeometryFace,
    RootedTreeStructure,
    validate_geometry,
)

GEOMETRY_GENOME_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class GeometryGenomeValidationIssue:
    """One representation-level reason a genome cannot materialize a geometry."""

    code: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code.isidentifier():
            raise ValueError("genome validation issue code must be a Python identifier")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("genome validation issue message must be non-empty")
        object.__setattr__(self, "message", self.message.strip())


@dataclass(frozen=True, slots=True)
class GeometryGenomeValidationReport:
    """Representation-level validation result for one geometry genome.

    This report answers only whether the stored genes can materialize a
    :class:`~toposc_lab.geometry.Geometry`. The resulting production geometry is
    checked separately by :func:`toposc_lab.geometry.validate_geometry`.
    """

    issues: tuple[GeometryGenomeValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        issues = tuple(self.issues)
        if not all(isinstance(issue, GeometryGenomeValidationIssue) for issue in issues):
            raise TypeError("issues must contain only GeometryGenomeValidationIssue instances")
        object.__setattr__(self, "issues", issues)

    @property
    def is_valid(self) -> bool:
        """Whether the genome can be converted to a production geometry."""
        return not self.issues

    def raise_for_errors(self) -> None:
        """Raise :class:`GeometryGenomeValidationError` for an invalid genome."""
        if self.issues:
            raise GeometryGenomeValidationError(self)


class GeometryGenomeValidationError(ValueError):
    """Raised when a geometry genome cannot materialize a ``Geometry``."""

    def __init__(self, report: GeometryGenomeValidationReport) -> None:
        if not isinstance(report, GeometryGenomeValidationReport):
            raise TypeError("report must be a GeometryGenomeValidationReport")
        self.report = report
        details = "; ".join(f"{issue.code}: {issue.message}" for issue in report.issues)
        super().__init__(f"geometry genome validation failed: {details}")


@dataclass(frozen=True, slots=True, eq=False)
class GeometryGenome:
    """Lossless, immutable genes for one finite ``Geometry`` snapshot.

    The genome deliberately stores no neighbor caches, fitness, population
    state, physical model parameters, or mutation behavior. Existing immutable
    geometry value records are reused so edge orientation and all declared
    geometric semantics retain exactly one definition in the project.

    ``coordinates is None`` and ``embedding_dimension is None`` preserve a
    purely abstract graph. No missing coordinates, boundary data, dimension, or
    other physical interpretation is inferred during either conversion.
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
    schema_version: int = field(
        default=GEOMETRY_GENOME_SCHEMA_VERSION,
        init=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "coordinates", _freeze_coordinates(self.coordinates))
        object.__setattr__(self, "boundary_sites", frozenset(self.boundary_sites))
        object.__setattr__(self, "boundary_components", tuple(self.boundary_components))
        object.__setattr__(
            self,
            "site_types",
            None if self.site_types is None else tuple(self.site_types),
        )
        object.__setattr__(self, "dimension_records", tuple(self.dimension_records))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
        object.__setattr__(self, "faces", tuple(self.faces))

    @property
    def is_abstract(self) -> bool:
        """Whether the genome declares no spatial embedding at all."""
        return self.embedding_dimension is None

    @classmethod
    def from_geometry(cls, geometry: Geometry) -> GeometryGenome:
        """Create a lossless genome from a valid production geometry."""
        return geometry_to_genome(geometry)

    def to_geometry(self) -> Geometry:
        """Materialize and validate a separate production geometry snapshot."""
        return geometry_from_genome(self)


def geometry_to_genome(geometry: Geometry) -> GeometryGenome:
    """Encode every stored field of a valid production geometry as genes."""
    if not isinstance(geometry, Geometry):
        raise TypeError("geometry must be a Geometry instance")
    validate_geometry(geometry).raise_for_errors()
    return GeometryGenome(
        n_sites=geometry.n_sites,
        edges=geometry.edges,
        coordinates=geometry.coordinates,
        embedding_dimension=geometry.embedding_dimension,
        boundary_sites=geometry.boundary_sites,
        boundary_components=geometry.boundary_components,
        site_types=geometry.site_types,
        dimension_records=geometry.dimension_records,
        rooted_tree=geometry.rooted_tree,
        metadata=geometry.metadata,
        faces=geometry.faces,
    )


def validate_geometry_genome(genome: GeometryGenome) -> GeometryGenomeValidationReport:
    """Check only whether genome genes can materialize a ``Geometry`` instance."""
    if not isinstance(genome, GeometryGenome):
        raise TypeError("genome must be a GeometryGenome instance")
    try:
        _materialize_geometry(genome)
    except (TypeError, ValueError) as error:
        detail = str(error).strip() or "the genome could not materialize a geometry"
        return GeometryGenomeValidationReport(
            issues=(
                GeometryGenomeValidationIssue(
                    code="invalid_geometry_representation",
                    message=detail,
                ),
            )
        )
    return GeometryGenomeValidationReport()


def geometry_from_genome(genome: GeometryGenome) -> Geometry:
    """Decode a genome, then run the separate production geometry validator."""
    report = validate_geometry_genome(genome)
    report.raise_for_errors()
    geometry = _materialize_geometry(genome)
    validate_geometry(geometry).raise_for_errors()
    return geometry


def _materialize_geometry(genome: GeometryGenome) -> Geometry:
    return Geometry(
        n_sites=genome.n_sites,
        edges=genome.edges,
        coordinates=genome.coordinates,
        embedding_dimension=genome.embedding_dimension,
        boundary_sites=genome.boundary_sites,
        boundary_components=genome.boundary_components,
        site_types=genome.site_types,
        dimension_records=genome.dimension_records,
        rooted_tree=genome.rooted_tree,
        metadata=genome.metadata,
        faces=genome.faces,
    )


def _freeze_coordinates(
    coordinates: NDArray[np.float64] | None,
) -> NDArray[np.float64] | None:
    if coordinates is None:
        return None
    if np.iscomplexobj(np.asarray(coordinates)):
        raise ValueError("complex genome coordinates are not supported")
    values = np.array(coordinates, dtype=np.float64, copy=True)
    contiguous = np.ascontiguousarray(values)
    immutable_buffer = contiguous.tobytes(order="C")
    return np.frombuffer(immutable_buffer, dtype=contiguous.dtype).reshape(contiguous.shape)


def _freeze_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(values, Mapping):
        raise TypeError("genome metadata must be a mapping")
    return MappingProxyType({key: _freeze_metadata_value(value) for key, value in values.items()})


def _freeze_metadata_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return _freeze_metadata_value(value.tolist())
        contiguous = np.ascontiguousarray(value)
        immutable_buffer = contiguous.tobytes(order="C")
        return np.frombuffer(immutable_buffer, dtype=contiguous.dtype).reshape(contiguous.shape)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_metadata_value(item) for item in value)
    if isinstance(value, bytearray):
        return bytes(value)
    return deepcopy(value)
