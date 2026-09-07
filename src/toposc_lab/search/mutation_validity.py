"""Policy-driven validity checks for geometry-genome mutations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import MappingProxyType
from typing import TypeAlias

import numpy as np

from toposc_lab.geometry import Geometry, GeometryValidationError, validate_geometry
from toposc_lab.search.geometry_genome import (
    GeometryGenome,
    GeometryGenomeValidationError,
    geometry_from_genome,
)

MutationMeasurement: TypeAlias = bool | int | float | None


@dataclass(frozen=True, slots=True)
class MutationValidityIssue:
    """One stable reason a proposed mutated genome is not admissible."""

    code: str
    message: str
    path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code.isidentifier():
            raise ValueError("mutation validity issue code must be a Python identifier")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("mutation validity issue message must be non-empty")
        if self.path is not None and (
            not isinstance(self.path, str) or not self.path.strip()
        ):
            raise ValueError("mutation validity issue path must be non-empty or None")
        object.__setattr__(self, "message", self.message.strip())
        if self.path is not None:
            object.__setattr__(self, "path", self.path.strip())


@dataclass(frozen=True, slots=True)
class MutationValidityPolicy:
    """Explicit, model-independent constraints for one mutation result.

    Every optional bound is disabled by default. Preserving embedding dimension
    is mandatory under the accepted Phase-10.8 decision and is therefore not a
    configurable policy flag.
    """

    require_connected: bool = False
    require_coordinates: bool = False
    required_embedding_dimension: int | None = None
    minimum_site_count: int | None = None
    maximum_site_count: int | None = None
    minimum_edge_count: int | None = None
    maximum_edge_count: int | None = None
    minimum_boundary_site_count: int | None = None
    maximum_boundary_site_count: int | None = None
    minimum_degree: int | None = None
    maximum_degree: int | None = None
    coordinate_lower_bounds: tuple[float, ...] | None = None
    coordinate_upper_bounds: tuple[float, ...] | None = None
    minimum_site_separation: float | None = None
    maximum_edge_length: float | None = None
    forbid_straight_edge_crossings: bool = False
    numerical_tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        for name in (
            "require_connected",
            "require_coordinates",
            "forbid_straight_edge_crossings",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")

        integer_fields = (
            "required_embedding_dimension",
            "minimum_site_count",
            "maximum_site_count",
            "minimum_edge_count",
            "maximum_edge_count",
            "minimum_boundary_site_count",
            "maximum_boundary_site_count",
            "minimum_degree",
            "maximum_degree",
        )
        for name in integer_fields:
            value = getattr(self, name)
            normalized = _optional_nonnegative_integer(value, name=name)
            if name == "required_embedding_dimension" and normalized == 0:
                raise ValueError("required_embedding_dimension must be positive")
            object.__setattr__(self, name, normalized)

        for lower_name, upper_name in (
            ("minimum_site_count", "maximum_site_count"),
            ("minimum_edge_count", "maximum_edge_count"),
            ("minimum_boundary_site_count", "maximum_boundary_site_count"),
            ("minimum_degree", "maximum_degree"),
        ):
            _validate_ordered_bounds(
                getattr(self, lower_name),
                getattr(self, upper_name),
                lower_name=lower_name,
                upper_name=upper_name,
            )

        lower = _optional_finite_vector(
            self.coordinate_lower_bounds,
            name="coordinate_lower_bounds",
        )
        upper = _optional_finite_vector(
            self.coordinate_upper_bounds,
            name="coordinate_upper_bounds",
        )
        if lower is not None and upper is not None:
            if len(lower) != len(upper):
                raise ValueError("coordinate bound vectors must have the same length")
            if any(lower_value > upper_value for lower_value, upper_value in zip(lower, upper)):
                raise ValueError("coordinate lower bounds must not exceed upper bounds")
        object.__setattr__(self, "coordinate_lower_bounds", lower)
        object.__setattr__(self, "coordinate_upper_bounds", upper)

        for name in ("minimum_site_separation", "maximum_edge_length"):
            value = _optional_nonnegative_finite_real(getattr(self, name), name=name)
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "numerical_tolerance",
            _positive_finite_real(self.numerical_tolerance, name="numerical_tolerance"),
        )


@dataclass(frozen=True, slots=True)
class MutationValidityReport:
    """Structured acceptance result and deterministic candidate measurements."""

    issues: tuple[MutationValidityIssue, ...] = ()
    measurements: Mapping[str, MutationMeasurement] = field(default_factory=dict)

    def __post_init__(self) -> None:
        issues = tuple(self.issues)
        if not all(isinstance(issue, MutationValidityIssue) for issue in issues):
            raise TypeError("issues must contain only MutationValidityIssue instances")
        if not isinstance(self.measurements, Mapping):
            raise TypeError("measurements must be a mapping")
        measurements: dict[str, MutationMeasurement] = {}
        for key, value in self.measurements.items():
            if not isinstance(key, str) or not key.isidentifier():
                raise ValueError("measurement keys must be Python identifiers")
            if value is not None and (
                isinstance(value, np.bool_)
                or not isinstance(value, (bool, Integral, Real))
            ):
                raise TypeError("measurement values must be scalar numbers, booleans, or None")
            if isinstance(value, Real) and not isinstance(value, (bool, Integral)):
                if not np.isfinite(float(value)):
                    raise ValueError("measurement values must be finite")
                value = float(value)
            elif isinstance(value, Integral) and not isinstance(value, bool):
                value = int(value)
            measurements[key] = value
        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "measurements", MappingProxyType(measurements))

    @property
    def is_valid(self) -> bool:
        """Whether the proposed mutation has no validity issue."""
        return not self.issues

    def raise_for_errors(self) -> None:
        """Raise ``InvalidGeometryMutationError`` when the mutation is invalid."""
        if self.issues:
            raise InvalidGeometryMutationError(self)


class InvalidGeometryMutationError(ValueError):
    """Raised when a caller explicitly rejects an invalid mutation result."""

    def __init__(self, report: MutationValidityReport) -> None:
        if not isinstance(report, MutationValidityReport):
            raise TypeError("report must be a MutationValidityReport")
        self.report = report
        details = "; ".join(
            f"{issue.code} at {issue.path or 'mutation'}: {issue.message}"
            for issue in report.issues
        )
        super().__init__(f"geometry mutation validation failed: {details}")


def validate_geometry_mutation(
    source: GeometryGenome,
    candidate: GeometryGenome,
    *,
    policy: MutationValidityPolicy | None = None,
) -> MutationValidityReport:
    """Validate one proposed genome transition without changing either genome.

    Base geometry validity and fixed embedding dimension are mandatory. All
    other checks come only from the explicit policy. Invalid candidates are
    reported rather than repaired or raised; API misuse still raises.
    """
    if not isinstance(source, GeometryGenome):
        raise TypeError("source must be a GeometryGenome instance")
    if not isinstance(candidate, GeometryGenome):
        raise TypeError("candidate must be a GeometryGenome instance")
    if policy is None:
        policy = MutationValidityPolicy()
    elif not isinstance(policy, MutationValidityPolicy):
        raise TypeError("policy must be a MutationValidityPolicy or None")

    issues: list[MutationValidityIssue] = []
    source_geometry = _validated_geometry(source, role="source", issues=issues)
    candidate_geometry = _validated_geometry(candidate, role="candidate", issues=issues)

    if (
        source_geometry is not None
        and candidate_geometry is not None
        and source.embedding_dimension != candidate.embedding_dimension
    ):
        issues.append(
            _issue(
                "embedding_dimension_changed",
                "mutation results must preserve the source embedding dimension",
                "candidate.embedding_dimension",
            )
        )

    measurements: dict[str, MutationMeasurement] = {}
    if candidate_geometry is not None:
        measurements = _candidate_measurements(candidate_geometry)
        _check_policy(
            candidate_geometry,
            policy=policy,
            issues=issues,
            measurements=measurements,
        )
    return MutationValidityReport(issues=tuple(issues), measurements=measurements)


def _validated_geometry(
    genome: GeometryGenome,
    *,
    role: str,
    issues: list[MutationValidityIssue],
) -> Geometry | None:
    try:
        return geometry_from_genome(genome)
    except GeometryGenomeValidationError as error:
        issues.extend(
            _issue(
                f"{role}_{item.code}",
                item.message,
                role,
            )
            for item in error.report.issues
        )
    except GeometryValidationError as error:
        issues.extend(
            _issue(
                f"{role}_{item.code}",
                item.message,
                f"{role}.{item.path}" if item.path is not None else role,
            )
            for item in error.report.errors
        )
    return None


def _candidate_measurements(geometry: Geometry) -> dict[str, MutationMeasurement]:
    degrees = tuple(geometry.degree(site) for site in geometry.site_indices)
    return {
        "site_count": geometry.n_sites,
        "edge_count": geometry.n_edges,
        "boundary_site_count": len(geometry.boundary_sites),
        "minimum_degree": min(degrees),
        "maximum_degree": max(degrees),
        "is_connected": validate_geometry(geometry).is_connected,
        "minimum_site_separation": None,
        "maximum_edge_length": None,
        "straight_edge_crossing_count": None,
    }


def _check_policy(
    geometry: Geometry,
    *,
    policy: MutationValidityPolicy,
    issues: list[MutationValidityIssue],
    measurements: dict[str, MutationMeasurement],
) -> None:
    tolerance = policy.numerical_tolerance
    if policy.require_connected and measurements["is_connected"] is False:
        issues.append(
            _issue(
                "disconnected_candidate",
                "the mutation policy requires one connected component",
                "candidate.edges",
            )
        )

    _check_integer_range(
        geometry.n_sites,
        minimum=policy.minimum_site_count,
        maximum=policy.maximum_site_count,
        label="site count",
        code_prefix="site_count",
        path="candidate.n_sites",
        issues=issues,
    )
    _check_integer_range(
        geometry.n_edges,
        minimum=policy.minimum_edge_count,
        maximum=policy.maximum_edge_count,
        label="edge count",
        code_prefix="edge_count",
        path="candidate.edges",
        issues=issues,
    )
    _check_integer_range(
        len(geometry.boundary_sites),
        minimum=policy.minimum_boundary_site_count,
        maximum=policy.maximum_boundary_site_count,
        label="boundary-site count",
        code_prefix="boundary_site_count",
        path="candidate.boundary_sites",
        issues=issues,
    )
    degrees = tuple(geometry.degree(site) for site in geometry.site_indices)
    _check_integer_range(
        min(degrees),
        minimum=policy.minimum_degree,
        maximum=None,
        label="minimum site degree",
        code_prefix="minimum_degree",
        path="candidate.edges",
        issues=issues,
    )
    _check_integer_range(
        max(degrees),
        minimum=None,
        maximum=policy.maximum_degree,
        label="maximum site degree",
        code_prefix="maximum_degree",
        path="candidate.edges",
        issues=issues,
    )

    if (
        policy.required_embedding_dimension is not None
        and geometry.embedding_dimension != policy.required_embedding_dimension
    ):
        issues.append(
            _issue(
                "required_embedding_dimension",
                "candidate embedding dimension does not match the mutation policy",
                "candidate.embedding_dimension",
            )
        )

    spatial_policy = any(
        value is not None
        for value in (
            policy.coordinate_lower_bounds,
            policy.coordinate_upper_bounds,
            policy.minimum_site_separation,
        )
    ) or policy.forbid_straight_edge_crossings
    coordinates = geometry.coordinates
    if coordinates is None and (policy.require_coordinates or spatial_policy):
        issues.append(
            _issue(
                "missing_required_coordinates",
                "the mutation policy requires explicit site coordinates",
                "candidate.coordinates",
            )
        )
    elif coordinates is not None:
        _check_coordinate_bounds(
            coordinates,
            lower=policy.coordinate_lower_bounds,
            upper=policy.coordinate_upper_bounds,
            tolerance=tolerance,
            issues=issues,
        )
        if policy.minimum_site_separation is not None and geometry.n_sites >= 2:
            separation = _minimum_site_separation(coordinates)
            measurements["minimum_site_separation"] = separation
            if separation + tolerance < policy.minimum_site_separation:
                issues.append(
                    _issue(
                        "minimum_site_separation",
                        "candidate site separation is below the mutation-policy minimum",
                        "candidate.coordinates",
                    )
                )

    if policy.maximum_edge_length is not None:
        edge_length, unavailable_edge = _maximum_edge_length(geometry)
        measurements["maximum_edge_length"] = edge_length
        if unavailable_edge is not None:
            issues.append(
                _issue(
                    "edge_length_unavailable",
                    "edge length requires coordinates or an explicit displacement",
                    f"candidate.edges[{unavailable_edge}]",
                )
            )
        elif edge_length is not None and edge_length > policy.maximum_edge_length + tolerance:
            issues.append(
                _issue(
                    "maximum_edge_length",
                    "candidate edge length exceeds the mutation-policy maximum",
                    "candidate.edges",
                )
            )

    if policy.forbid_straight_edge_crossings:
        if coordinates is None or coordinates.shape[1] != 2:
            issues.append(
                _issue(
                    "straight_edge_crossing_check_unavailable",
                    "straight-edge crossing checks require explicit two-dimensional coordinates",
                    "candidate.coordinates",
                )
            )
        else:
            crossing_count = _straight_edge_crossing_count(
                geometry,
                coordinates=coordinates,
                tolerance=tolerance,
            )
            measurements["straight_edge_crossing_count"] = crossing_count
            if crossing_count:
                issues.append(
                    _issue(
                        "straight_edge_crossing",
                        f"candidate contains {crossing_count} non-adjacent straight-edge crossings",
                        "candidate.edges",
                    )
                )


def _check_integer_range(
    value: int,
    *,
    minimum: int | None,
    maximum: int | None,
    label: str,
    code_prefix: str,
    path: str,
    issues: list[MutationValidityIssue],
) -> None:
    if minimum is not None and value < minimum:
        issues.append(
            _issue(
                f"{code_prefix}_below_minimum",
                f"{label} {value} is below the required minimum {minimum}",
                path,
            )
        )
    if maximum is not None and value > maximum:
        issues.append(
            _issue(
                f"{code_prefix}_above_maximum",
                f"{label} {value} exceeds the allowed maximum {maximum}",
                path,
            )
        )


def _check_coordinate_bounds(
    coordinates: np.ndarray,
    *,
    lower: tuple[float, ...] | None,
    upper: tuple[float, ...] | None,
    tolerance: float,
    issues: list[MutationValidityIssue],
) -> None:
    dimension = coordinates.shape[1]
    if lower is not None:
        if len(lower) != dimension:
            issues.append(
                _issue(
                    "coordinate_lower_bounds_dimension",
                    "coordinate lower-bound dimension does not match the candidate embedding",
                    "policy.coordinate_lower_bounds",
                )
            )
        elif np.any(coordinates < np.asarray(lower) - tolerance):
            issues.append(
                _issue(
                    "coordinate_below_lower_bound",
                    "candidate coordinates fall below the mutation-policy lower bounds",
                    "candidate.coordinates",
                )
            )
    if upper is not None:
        if len(upper) != dimension:
            issues.append(
                _issue(
                    "coordinate_upper_bounds_dimension",
                    "coordinate upper-bound dimension does not match the candidate embedding",
                    "policy.coordinate_upper_bounds",
                )
            )
        elif np.any(coordinates > np.asarray(upper) + tolerance):
            issues.append(
                _issue(
                    "coordinate_above_upper_bound",
                    "candidate coordinates exceed the mutation-policy upper bounds",
                    "candidate.coordinates",
                )
            )


def _minimum_site_separation(coordinates: np.ndarray) -> float:
    minimum = np.inf
    for first_index in range(coordinates.shape[0] - 1):
        differences = coordinates[first_index + 1 :] - coordinates[first_index]
        minimum = min(minimum, float(np.min(np.linalg.norm(differences, axis=1))))
    return float(minimum)


def _maximum_edge_length(geometry: Geometry) -> tuple[float | None, int | None]:
    maximum = 0.0
    for edge_index, edge in enumerate(geometry.edges):
        try:
            length = geometry.distance(edge.source, edge.target)
        except ValueError:
            return None, edge_index
        maximum = max(maximum, length)
    return maximum, None


def _straight_edge_crossing_count(
    geometry: Geometry,
    *,
    coordinates: np.ndarray,
    tolerance: float,
) -> int:
    edges = tuple((edge.source, edge.target) for edge in geometry.edges)
    return sum(
        _segments_intersect(
            coordinates[first[0]],
            coordinates[first[1]],
            coordinates[second[0]],
            coordinates[second[1]],
            tolerance=tolerance,
        )
        for first_index, first in enumerate(edges)
        for second in edges[first_index + 1 :]
        if not set(first).intersection(second)
    )


def _segments_intersect(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    *,
    tolerance: float,
) -> bool:
    def orientation(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> float:
        first = end - start
        second = point - start
        return float(first[0] * second[1] - first[1] * second[0])

    first_a = orientation(first_start, first_end, second_start)
    first_b = orientation(first_start, first_end, second_end)
    second_a = orientation(second_start, second_end, first_start)
    second_b = orientation(second_start, second_end, first_end)
    if first_a * first_b < -(tolerance**2) and second_a * second_b < -(tolerance**2):
        return True

    def on_segment(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> bool:
        return bool(
            np.all(point >= np.minimum(start, end) - tolerance)
            and np.all(point <= np.maximum(start, end) + tolerance)
        )

    return (
        (abs(first_a) <= tolerance and on_segment(first_start, first_end, second_start))
        or (abs(first_b) <= tolerance and on_segment(first_start, first_end, second_end))
        or (abs(second_a) <= tolerance and on_segment(second_start, second_end, first_start))
        or (abs(second_b) <= tolerance and on_segment(second_start, second_end, first_end))
    )


def _issue(code: str, message: str, path: str | None) -> MutationValidityIssue:
    return MutationValidityIssue(code=code, message=message, path=path)


def _optional_nonnegative_integer(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer or None")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _validate_ordered_bounds(
    lower: int | None,
    upper: int | None,
    *,
    lower_name: str,
    upper_name: str,
) -> None:
    if lower is not None and upper is not None and lower > upper:
        raise ValueError(f"{lower_name} must not exceed {upper_name}")


def _optional_finite_vector(
    value: Sequence[float] | None,
    *,
    name: str,
) -> tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a finite real sequence or None")
    result = tuple(_finite_real(item, name=name) for item in value)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _optional_nonnegative_finite_real(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    result = _finite_real(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _positive_finite_real(value: object, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must contain real numbers")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must contain only finite numbers")
    return result
