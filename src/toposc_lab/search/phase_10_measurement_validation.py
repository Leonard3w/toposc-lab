"""Frozen inputs for TOPOSC-P10-MEAS-VAL-001."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from toposc_lab.evaluation.reproducibility import exact_geometry_id
from toposc_lab.geometry import Geometry, GeometryBoundaryComponent, GeometryEdge
from toposc_lab.geometry.generators.square import square
from toposc_lab.geometry.validation import validate_geometry
from toposc_lab.models.chiral_p_wave import ChiralPWaveParameters
from toposc_lab.search.geometry_genome import GeometryGenome
from toposc_lab.search.phase_9_8_evaluation import (
    PHASE_9_8_GEOMETRY_TOLERANCE,
    Phase98TopologyInputs,
    _boundary_graph_distances,
    _clipped_voronoi_areas,
    _segments_intersect,
)
from toposc_lab.search.phase_10_size_methods import MASK_NAMES

ModelRole = Literal["topological", "trivial", "undefined"]
Arm = Literal["boundary", "interior"]

MEASUREMENT_VALIDATION_PROTOCOL_ID = "TOPOSC-P10-MEAS-VAL-001"
MEASUREMENT_VALIDATION_PROTOCOL_COMMIT = "94e6032dda44cc87e450486a1d694d3321212029"
MEASUREMENT_VALIDATION_PROTOCOL_PATH = (
    "docs/decisions/pre_phase_10_measurement_validation_protocol_v1.md"
)
PREFLIGHT_SIZE = 12
FULL_SIZES = (16, 20)
SUPPORTED_SIZES = (PREFLIGHT_SIZE, *FULL_SIZES)
MODEL_ROLES: tuple[ModelRole, ...] = ("topological", "trivial", "undefined")
VARIANT_MODEL_ROLES: tuple[ModelRole, ...] = ("topological", "trivial")
OFFSETS = (-2, 0, 2)
ARMS: tuple[Arm, ...] = ("boundary", "interior")


def model_parameters(role: ModelRole) -> ChiralPWaveParameters:
    """Return the exact model parameters assigned to one frozen role."""
    if role not in MODEL_ROLES:
        raise ValueError(f"model role must be one of {MODEL_ROLES}")
    hopping, chemical_potential, pairing = {
        "topological": (1.0, 2.0, 1.0),
        "trivial": (1.0, 12.0, 1.0),
        "undefined": (0.0, 0.0, 0.0),
    }[role]
    return ChiralPWaveParameters(
        hopping=hopping,
        chemical_potential=chemical_potential,
        pairing=pairing,
        chirality=1,
        plane_axes=(0, 1),
    )


def measurement_reference(n: int) -> Geometry:
    """Build an unchanged open square with neutral SIZE-001 semantics."""
    _require_size(n)
    base = square(n, n, spacing=1.0, boundary_x="open", boundary_y="open")
    coordinates = base.coordinates
    assert coordinates is not None
    pairs = tuple(sorted((edge.source, edge.target) for edge in base.edges))
    return _build_geometry(n, coordinates, pairs)


def intervention_definition(n: int, offset: int, arm: Arm) -> dict[str, Any]:
    """Return the exact degree-preserving local intervention and patch."""
    _require_size(n)
    if offset not in OFFSETS:
        raise ValueError(f"offset must be one of {OFFSETS}")
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    x_value = 0 if arm == "boundary" else n // 4
    y_value = n // 2 - 1 + offset
    points = (
        (x_value, y_value),
        (x_value + 1, y_value),
        (x_value + 1, y_value + 1),
        (x_value + 2, y_value + 1),
    )
    removed = (_edge(n, points[0], points[1]), _edge(n, points[2], points[3]))
    added = (_edge(n, points[0], points[2]), _edge(n, points[1], points[3]))
    patch = tuple(
        sorted(_site(n, (x_value + u, y_value + v)) for u in range(4) for v in range(-1, 3))
    )
    if any(not 0 <= coordinate < n for point in points for coordinate in point):
        raise ValueError("intervention lies outside the square")
    if len(patch) != 16 or len(set(patch)) != 16:
        raise ValueError("measurement patch must contain 16 unique sites")
    return {
        "offset": offset,
        "arm": arm,
        "affected_sites": tuple(sorted(_site(n, point) for point in points)),
        "removed_edges": tuple(sorted(removed)),
        "added_edges": tuple(sorted(added)),
        "measurement_sites": patch,
        "intervention_depth": min(
            min(x_coord, y_coord, n - 1 - x_coord, n - 1 - y_coord)
            for x_coord, y_coord in points
        ),
    }


def build_measurement_cell(
    n: int,
    role: ModelRole,
    *,
    block: Literal["control_start", "intervention", "control_end"],
    offset: int | None = None,
    arm: Arm | None = None,
) -> dict[str, Any]:
    """Build and audit one immutable campaign input cell."""
    _require_size(n)
    parameters = model_parameters(role)
    reference = measurement_reference(n)
    if block == "intervention":
        if role not in VARIANT_MODEL_ROLES or offset is None or arm is None:
            raise ValueError("interventions require a gapped role, offset and arm")
        definition = intervention_definition(n, offset, arm)
        reference_edges = {(edge.source, edge.target) for edge in reference.edges}
        removed = set(definition["removed_edges"])
        added = set(definition["added_edges"])
        if not removed.issubset(reference_edges) or added.intersection(reference_edges):
            raise ValueError("intervention does not replace the frozen square edges")
        geometry = _build_geometry(
            n,
            _coordinates(reference),
            tuple(sorted((reference_edges - removed) | added)),
        )
    elif block in ("control_start", "control_end"):
        if offset is not None or arm is not None:
            raise ValueError("controls must not have an offset or arm")
        definition = {
            "offset": None,
            "arm": None,
            "affected_sites": (),
            "removed_edges": (),
            "added_edges": (),
            "measurement_sites": (),
            "intervention_depth": None,
        }
        geometry = reference
    else:
        raise ValueError("unknown measurement-validation block")
    validation = validate_measurement_geometry(geometry, n)
    if not validation["is_valid"]:
        raise ValueError(f"invalid measurement-validation geometry: {validation['issues']}")
    if _degrees(geometry) != _degrees(reference):
        raise ValueError("intervention changed the degree sequence")
    expected_depth = None if block != "intervention" else (0 if arm == "boundary" else n // 4)
    if definition["intervention_depth"] != expected_depth:
        raise ValueError("intervention has the wrong depth")
    return {
        "n": n,
        "block": block,
        "model_role": role,
        "model_parameters": parameters.model_dump(mode="json"),
        **definition,
        "genome": GeometryGenome.from_geometry(geometry),
        "geometry_id": exact_geometry_id(geometry),
        "validation": validation,
        "degree_sequence": _degrees(geometry),
        "edge_distance": len(definition["removed_edges"]),
        "evaluation_seed": None,
    }


def build_measurement_plan(*, preflight: bool) -> tuple[dict[str, Any], ...]:
    """Return the exact 6-slot preflight or 36-slot full plan."""
    sizes = (PREFLIGHT_SIZE,) if preflight else FULL_SIZES
    cells: list[dict[str, Any]] = []
    for n in sizes:
        cells.extend(
            build_measurement_cell(n, role, block="control_start")
            for role in MODEL_ROLES
        )
        if not preflight:
            cells.extend(
                build_measurement_cell(
                    n, role, block="intervention", offset=offset, arm=arm
                )
                for role in VARIANT_MODEL_ROLES
                for offset in OFFSETS
                for arm in ARMS
            )
        cells.extend(
            build_measurement_cell(n, role, block="control_end")
            for role in MODEL_ROLES
        )
    expected = 6 if preflight else 36
    if len(cells) != expected:
        raise AssertionError("measurement-validation plan differs from its frozen budget")
    return tuple(cells)


def build_measurement_topology_inputs(
    geometry: Geometry, n: int
) -> tuple[Phase98TopologyInputs, dict[str, Any]]:
    """Build the five fixed masks and the variable-size topology inputs."""
    report = validate_measurement_geometry(geometry, n)
    if not report["is_valid"]:
        raise ValueError(f"invalid measurement geometry: {report['issues']}")
    coordinates = _coordinates(geometry)
    areas = _clipped_voronoi_areas(
        coordinates, cell=(-0.5, n - 0.5, -0.5, n - 0.5)
    )
    if np.any(areas <= 0.0) or not np.isclose(
        float(np.sum(areas)), float(n * n), rtol=0.0, atol=1.0e-10
    ):
        raise ValueError("measurement-validation Voronoi areas are invalid")
    order = np.lexsort((coordinates[:, 1], coordinates[:, 0]))
    graph_distances = _boundary_graph_distances(geometry)
    base = _base_coordinates(n)
    fixed_distances = np.min(
        np.column_stack(
            (base[:, 0], base[:, 1], n - 1 - base[:, 0], n - 1 - base[:, 1])
        ),
        axis=1,
    )
    masks = (
        graph_distances[order] >= 2,
        graph_distances[order] >= 3,
        fixed_distances[order] >= 2,
        fixed_distances[order] >= 3,
        fixed_distances[order] >= n / 4,
    )
    inputs = Phase98TopologyInputs(
        basis_coordinates=np.tile(coordinates, (2, 1)),
        position_areas=areas[order],
        bulk_masks=masks,
        bott_periods=((0.95 * n, 0.95 * n), (float(n), float(n)), (1.05 * n, 1.05 * n)),
        localizer_probe=((n - 1) / 2, (n - 1) / 2),
        localizer_kappas=(0.1, 0.2, 0.3),
    )
    return inputs, {
        "mask_names": MASK_NAMES,
        "unique_coordinate_order": order,
        "position_areas_in_coordinate_order": inputs.position_areas,
        "bulk_masks_in_coordinate_order": inputs.bulk_masks,
        "bott_periods": inputs.bott_periods,
        "localizer_probe": inputs.localizer_probe,
        "localizer_kappas": inputs.localizer_kappas,
    }


def validate_measurement_geometry(geometry: Geometry, n: int) -> dict[str, Any]:
    """Validate the complete geometry and neutral representation contract."""
    _require_size(n)
    issues: list[str] = []
    base = validate_geometry(geometry, require_connected=True)
    if not base.is_valid:
        issues.append("base_geometry_invalid")
    coordinates = geometry.coordinates
    expected_boundary = frozenset(
        x_value * n + y_value
        for x_value in range(n)
        for y_value in range(n)
        if x_value in (0, n - 1) or y_value in (0, n - 1)
    )
    if geometry.n_sites != n * n:
        issues.append("site_count")
    if geometry.n_edges != 2 * n * (n - 1):
        issues.append("edge_count")
    minimum_separation: float | None = None
    maximum_edge_length: float | None = None
    crossing_count: int | None = None
    if coordinates is None or coordinates.shape != (n * n, 2):
        issues.append("coordinates")
    else:
        differences = coordinates[:, None, :] - coordinates[None, :, :]
        distances = np.linalg.norm(differences, axis=2)
        np.fill_diagonal(distances, np.inf)
        minimum_separation = float(np.min(distances))
        maximum_edge_length = max(
            float(np.linalg.norm(coordinates[edge.target] - coordinates[edge.source]))
            for edge in geometry.edges
        )
        pairs = tuple((edge.source, edge.target) for edge in geometry.edges)
        crossing_count = sum(
            _segments_intersect(
                coordinates[first[0]], coordinates[first[1]],
                coordinates[second[0]], coordinates[second[1]],
            )
            for index, first in enumerate(pairs)
            for second in pairs[index + 1 :]
            if not set(first).intersection(second)
        )
        if minimum_separation < 0.55:
            issues.append("minimum_site_separation")
        if maximum_edge_length > 1.75:
            issues.append("maximum_edge_length")
        if crossing_count:
            issues.append("straight_edge_crossing")
        if not (
            np.allclose(np.min(coordinates, axis=0), (0.0, 0.0), rtol=0.0,
                        atol=PHASE_9_8_GEOMETRY_TOLERANCE)
            and np.allclose(np.max(coordinates, axis=0), (n - 1, n - 1), rtol=0.0,
                            atol=PHASE_9_8_GEOMETRY_TOLERANCE)
        ):
            issues.append("bounding_box")
        shell = frozenset(
            site for site, (x_value, y_value) in enumerate(coordinates)
            if min(x_value, y_value, n - 1 - x_value, n - 1 - y_value) <= 0.875
        )
        if shell != expected_boundary:
            issues.append("boundary_shell")
        if any(
            edge.displacement is None
            or not np.allclose(
                edge.displacement,
                coordinates[edge.target] - coordinates[edge.source],
                rtol=0.0,
                atol=PHASE_9_8_GEOMETRY_TOLERANCE,
            )
            for edge in geometry.edges
        ):
            issues.append("edge_displacement")
    degrees = _degrees(geometry)
    if not degrees or min(degrees) < 2 or max(degrees) > 4:
        issues.append("degree_range")
    pairs = tuple((edge.source, edge.target) for edge in geometry.edges)
    if pairs != tuple(sorted(pairs)) or any(source >= target for source, target in pairs):
        issues.append("edge_order")
    if any(
        edge.edge_type != "size_calibration_coupling"
        or edge.boundary_crossing
        or bool(edge.metadata)
        for edge in geometry.edges
    ):
        issues.append("edge_semantics")
    if geometry.boundary_sites != expected_boundary:
        issues.append("boundary_membership")
    if (
        len(geometry.boundary_components) != 1
        or geometry.boundary_components[0]
        != GeometryBoundaryComponent("outer", 0, expected_boundary)
    ):
        issues.append("boundary_component")
    if (
        geometry.dimension_records
        or geometry.faces
        or geometry.rooted_tree is not None
        or geometry.site_types is not None
        or dict(geometry.metadata) != {"generator": "phase10_size_square_deformation_v1"}
    ):
        issues.append("neutral_metadata")
    return {
        "is_valid": not issues,
        "issues": tuple(dict.fromkeys(issues)),
        "measurements": {
            "site_count": geometry.n_sites,
            "edge_count": geometry.n_edges,
            "minimum_degree": None if not degrees else min(degrees),
            "maximum_degree": None if not degrees else max(degrees),
            "boundary_site_count": len(geometry.boundary_sites),
            "minimum_site_separation": minimum_separation,
            "maximum_edge_length": maximum_edge_length,
            "straight_edge_crossing_count": crossing_count,
            "base_valid": base.is_valid,
        },
    }


def _build_geometry(
    n: int, coordinates: np.ndarray, edge_pairs: tuple[tuple[int, int], ...]
) -> Geometry:
    boundary = frozenset(
        x_value * n + y_value
        for x_value in range(n)
        for y_value in range(n)
        if x_value in (0, n - 1) or y_value in (0, n - 1)
    )
    points = np.asarray(coordinates, dtype=float)
    return Geometry(
        n_sites=n * n,
        edges=tuple(
            GeometryEdge(
                source,
                target,
                edge_type="size_calibration_coupling",
                displacement=tuple(float(value) for value in points[target] - points[source]),
            )
            for source, target in sorted(edge_pairs)
        ),
        coordinates=points,
        boundary_sites=boundary,
        boundary_components=(GeometryBoundaryComponent("outer", 0, boundary),),
        dimension_records=(),
        metadata={"generator": "phase10_size_square_deformation_v1"},
    )


def _base_coordinates(n: int) -> np.ndarray:
    coordinates = square(n, n, spacing=1.0).coordinates
    assert coordinates is not None
    return coordinates.copy()


def _coordinates(geometry: Geometry) -> np.ndarray:
    coordinates = geometry.coordinates
    if coordinates is None:
        raise ValueError("measurement geometry requires coordinates")
    return coordinates


def _degrees(geometry: Geometry) -> tuple[int, ...]:
    return tuple(len(geometry.neighbors(site)) for site in range(geometry.n_sites))


def _site(n: int, point: tuple[int, int]) -> int:
    return point[0] * n + point[1]


def _edge(n: int, first: tuple[int, int], second: tuple[int, int]) -> tuple[int, int]:
    source, target = _site(n, first), _site(n, second)
    return (source, target) if source < target else (target, source)


def _require_size(n: int) -> None:
    if type(n) is not int or n not in SUPPORTED_SIZES:
        raise ValueError(f"n must be one of {SUPPORTED_SIZES}")
