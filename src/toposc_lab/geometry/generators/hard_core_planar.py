"""Frozen hard-core planar generators for the Phase-9.8 benchmark."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from importlib.metadata import version
from numbers import Integral
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay, QhullError  # type: ignore[import-untyped]

from toposc_lab.geometry.base import (
    Geometry,
    GeometryBoundaryComponent,
    GeometryEdge,
)

HARD_CORE_PLANAR_GENERATOR_VERSION = 1
HARD_CORE_PLANAR_N_SITES = 64
HARD_CORE_PLANAR_N_EDGES = 112
HARD_CORE_PLANAR_BOX_MAXIMUM = 7.0
HARD_CORE_PLANAR_MINIMUM_SEPARATION = 0.55
HARD_CORE_PLANAR_MAXIMUM_EDGE_LENGTH = 1.75
HARD_CORE_PLANAR_MINIMUM_DEGREE = 2
HARD_CORE_PLANAR_MAXIMUM_DEGREE = 4
HARD_CORE_PLANAR_BOUNDARY_SHELL_THICKNESS = 0.875
HARD_CORE_PLANAR_MINIMUM_BOUNDARY_SITES = 24
HARD_CORE_PLANAR_MAXIMUM_BOUNDARY_SITES = 32
HARD_CORE_PLANAR_MAX_POINT_PROPOSALS = 1_000_000
HARD_CORE_PLANAR_MAX_COMPLETE_ATTEMPTS = 10_000
HARD_CORE_PLANAR_QHULL_OPTIONS = "Qbb Qc Qz Q12"
HARD_CORE_PLANAR_TRIANGLE_AREA_TOLERANCE = 1.0e-10

_GeneratorMode: TypeAlias = Literal["candidate", "reference"]
_Edge: TypeAlias = tuple[int, int]


@dataclass(frozen=True)
class HardCorePlanarConfig:
    """Explicit construction limits; defaults reproduce the frozen v1 generator."""

    n_sites: int = 64
    n_edges: int = 112
    box_maximum: float = 7.0
    minimum_separation: float = 0.55
    maximum_edge_length: float = 1.75
    minimum_degree: int = 2
    maximum_degree: int = 4
    boundary_shell_thickness: float = 0.875
    minimum_boundary_sites: int = 24
    maximum_boundary_sites: int = 32
    max_point_proposals: int = 1_000_000
    max_complete_attempts: int = 10_000
    forbid_crossings: bool = True
    require_connected: bool = True

    def __post_init__(self) -> None:
        for name in ("n_sites", "n_edges", "minimum_degree", "maximum_degree",
                     "minimum_boundary_sites", "maximum_boundary_sites",
                     "max_point_proposals", "max_complete_attempts"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 4 <= self.n_sites <= 4096:
            raise ValueError("n_sites must be between 4 and 4096")
        if not self.minimum_degree <= self.maximum_degree < self.n_sites:
            raise ValueError("degree bounds must be ordered and below n_sites")
        if not self.minimum_boundary_sites <= self.maximum_boundary_sites <= self.n_sites:
            raise ValueError("boundary-site bounds must be ordered and at most n_sites")
        if not self.n_sites - 1 <= self.n_edges <= self.n_sites * (self.n_sites - 1) // 2:
            raise ValueError("edge budget is impossible for a connected simple graph")
        if not self.n_sites * self.minimum_degree <= 2 * self.n_edges <= self.n_sites * self.maximum_degree:
            raise ValueError("edge budget is incompatible with degree bounds")
        if type(self.forbid_crossings) is not bool or type(self.require_connected) is not bool:
            raise ValueError("crossing and connectivity settings must be boolean")
        if not self.require_connected:
            raise ValueError("this generator constructs connected graphs only")
        if self.forbid_crossings and self.n_edges > 3 * self.n_sites - 6:
            raise ValueError("edge budget exceeds the planar simple-graph bound")
        for name in ("box_maximum", "minimum_separation", "maximum_edge_length", "boundary_shell_thickness"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.boundary_shell_thickness >= self.box_maximum / 2:
            raise ValueError("boundary shell must be smaller than half the box")
        if self.minimum_separation > min(self.maximum_edge_length, np.sqrt(2) * self.box_maximum):
            raise ValueError("minimum separation precludes connected bonds")
        # Disjoint separation disks fit inside this expanded enclosing square.
        radius = self.minimum_separation / 2
        if self.n_sites * np.pi * radius**2 > (self.box_maximum + 2 * radius)**2:
            raise ValueError("site count and minimum separation cannot fit in the box")
        if not self.n_sites <= self.max_point_proposals <= 10_000_000:
            raise ValueError("max_point_proposals must be n_sites..10000000")
        if self.max_complete_attempts > 100_000:
            raise ValueError("max_complete_attempts must be at most 100000")


_DEFAULT_CONFIG = HardCorePlanarConfig()


def _config(value: HardCorePlanarConfig | dict | None) -> HardCorePlanarConfig:
    if value is None:
        return _DEFAULT_CONFIG
    if isinstance(value, dict):
        return HardCorePlanarConfig(**value)
    if not isinstance(value, HardCorePlanarConfig):
        raise TypeError("config must be a HardCorePlanarConfig or mapping")
    return value


class HardCorePlanarGenerationError(RuntimeError):
    """The frozen generator exhausted its declared construction budget."""


@dataclass(slots=True)
class _DisjointSet:
    parents: list[int]
    ranks: list[int]

    @classmethod
    def create(cls, size: int) -> _DisjointSet:
        return cls(parents=list(range(size)), ranks=[0] * size)

    def find(self, item: int) -> int:
        parent = self.parents[item]
        while parent != self.parents[parent]:
            parent = self.parents[parent]
        while item != parent:
            next_item = self.parents[item]
            self.parents[item] = parent
            item = next_item
        return parent

    def union(self, first: int, second: int) -> bool:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return False
        if self.ranks[first_root] < self.ranks[second_root]:
            first_root, second_root = second_root, first_root
        self.parents[second_root] = first_root
        if self.ranks[first_root] == self.ranks[second_root]:
            self.ranks[first_root] += 1
        return True


def hard_core_planar_graph(*, seed: int, config: HardCorePlanarConfig | dict | None = None) -> Geometry:
    """Generate the frozen randomly prioritized Phase-9.8 candidate graph."""
    return _hard_core_planar_geometry(seed=seed, mode="candidate", config=_config(config))


def hard_core_planar_reference(*, seed: int, config: HardCorePlanarConfig | dict | None = None) -> Geometry:
    """Generate the frozen length-completed amorphous reference graph."""
    return _hard_core_planar_geometry(seed=seed, mode="reference", config=_config(config))


def constrained_random_embedded_graph(*, seed: int, config: HardCorePlanarConfig | dict | None = None) -> Geometry:
    """64-site hard-core graph with random edges from the full radius pool.

    Reuses the frozen point process, spanning/completion algorithm and rejection
    rules, replacing only the Delaunay edge pool by the existing cutoff builder.
    This is a constrained rejection ensemble, not uniform over all valid graphs.
    """
    return _hard_core_planar_geometry(seed=seed, mode="candidate", edge_pool_kind="radius", config=_config(config))


def hard_core_planar_edge_pool(coordinates: NDArray[np.float64]) -> tuple[_Edge, ...]:
    """Reconstruct the frozen v1 Delaunay pool without sampling or changing coordinates.

    Uses the same Qhull options, degeneracy rejection and length cutoff as both
    generators. Callers still enforce complete graph and physical-boundary rules.
    """
    values = np.asarray(coordinates, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.all(np.isfinite(values)):
        raise ValueError("edge pool requires finite two-dimensional coordinates")
    return _delaunay_edges(values)


def _hard_core_planar_geometry(*, seed: int, mode: _GeneratorMode,
                              edge_pool_kind: Literal["delaunay", "radius"] = "delaunay",
                              config: HardCorePlanarConfig = _DEFAULT_CONFIG) -> Geometry:
    prepared_seed = _nonnegative_integer(seed, name="seed")
    rng = np.random.Generator(np.random.PCG64(prepared_seed))
    proposal_count = 0
    rejected_attempts: Counter[str] = Counter()

    for complete_attempt_count in range(1, config.max_complete_attempts + 1):
        accepted: list[tuple[float, float, int]] = []
        while len(accepted) < config.n_sites:
            if proposal_count >= config.max_point_proposals:
                raise HardCorePlanarGenerationError(
                    "hard-core point construction exceeded "
                    f"{config.max_point_proposals} proposals"
                )
            proposal_count += 1
            proposal = rng.uniform(
                0.0,
                config.box_maximum,
                size=2,
            )
            if accepted:
                existing = np.asarray(
                    [(x_coordinate, y_coordinate) for x_coordinate, y_coordinate, _ in accepted],
                    dtype=float,
                )
                distances = np.linalg.norm(existing - proposal[np.newaxis, :], axis=1)
                if float(np.min(distances)) < config.minimum_separation:
                    continue
            accepted.append(
                (float(proposal[0]), float(proposal[1]), len(accepted))
            )

        coordinates = _normalized_coordinates(accepted, config=config)
        if _minimum_separation(coordinates) < config.minimum_separation:
            rejected_attempts["normalized_minimum_separation"] += 1
            continue

        try:
            if edge_pool_kind == "delaunay":
                candidate_edges = _delaunay_edges(coordinates, config=config)
            else:
                from toposc_lab.geometry.generators.coordinate_cutoff import coordinate_cutoff_graph
                pool = coordinate_cutoff_graph(coordinates, config.maximum_edge_length)
                candidate_edges = tuple((e.source, e.target) for e in pool.edges)
        except QhullError:
            rejected_attempts["qhull_failure"] += 1
            continue
        except _DegenerateTriangleError:
            rejected_attempts["degenerate_delaunay_triangle"] += 1
            continue

        if len(candidate_edges) < config.n_edges:
            rejected_attempts["insufficient_candidate_edges"] += 1
            continue

        priorities = {
            edge: int(rng.bit_generator.random_raw()) for edge in candidate_edges
        }
        selected_edges, degrees, failure_reason = _select_edges(
            coordinates,
            candidate_edges=candidate_edges,
            priorities=priorities,
            mode=mode,
            prevent_crossings=edge_pool_kind == "radius" and config.forbid_crossings,
            config=config,
        )
        if failure_reason is not None:
            rejected_attempts[failure_reason] += 1
            continue
        if min(degrees) < config.minimum_degree:
            rejected_attempts["minimum_degree"] += 1
            continue
        if config.forbid_crossings and _has_straight_edge_crossing(coordinates, selected_edges):
            rejected_attempts["straight_edge_crossing"] += 1
            continue

        boundary_sites = _outer_boundary_sites(coordinates, config=config)
        if not (
            config.minimum_boundary_sites
            <= len(boundary_sites)
            <= config.maximum_boundary_sites
        ):
            rejected_attempts["boundary_site_count"] += 1
            continue

        geometry_edges = tuple(
            GeometryEdge(
                source,
                target,
                edge_type="hard_core_planar_coupling",
                displacement=tuple(
                    float(value) for value in coordinates[target] - coordinates[source]
                ),
            )
            for source, target in selected_edges
        )
        return Geometry(
            n_sites=config.n_sites,
            edges=geometry_edges,
            coordinates=coordinates,
            boundary_sites=boundary_sites,
            boundary_components=(
                GeometryBoundaryComponent("outer", 0, boundary_sites),
            ),
            metadata={
                "generator": (
                    "constrained_random_embedded_graph" if edge_pool_kind == "radius" else "hard_core_planar_graph"
                    if mode == "candidate"
                    else "hard_core_planar_reference"
                ),
                "algorithm_version": (HARD_CORE_PLANAR_GENERATOR_VERSION if config == _DEFAULT_CONFIG else 2),
                **({} if config == _DEFAULT_CONFIG else {"resolved_config": asdict(config)}),
                "construction_mode": mode,
                "rng_algorithm": "numpy.random.PCG64",
                "proposal_count": proposal_count,
                "complete_attempt_count": complete_attempt_count,
                "rejected_attempt_reason_counts": dict(
                    sorted(rejected_attempts.items())
                ),
                "point_proposal_distribution": f"uniform_[0,{config.box_maximum:g}]^2",
                "point_iteration_order": "sequential_acceptance_order",
                "site_index_order": "lexicographic_x_y_acceptance_index",
                "affine_normalization": f"independent_axes_to_exact_[0,{config.box_maximum:g}]",
                "minimum_separation": config.minimum_separation,
                "maximum_edge_length": config.maximum_edge_length,
                "delaunay_implementation": "scipy.spatial.Delaunay" if edge_pool_kind == "delaunay" else "not_used_radius_pool",
                "delaunay_version": version("scipy"),
                "qhull_options": HARD_CORE_PLANAR_QHULL_OPTIONS,
                "qhull_jitter": False,
                "triangle_doubled_area_tolerance": (
                    HARD_CORE_PLANAR_TRIANGLE_AREA_TOLERANCE
                ),
                "spanning_tree_priority": (
                    "one_pcg64_raw_word_per_lexicographic_candidate_edge"
                ),
                "completion_priority": (
                    "pcg64_raw_word_then_source_target"
                    if mode == "candidate"
                    else "edge_length_then_source_target"
                ),
                "edge_storage_order": "lexicographic_source_target",
                "edge_orientation": "lower_site_index_to_higher_site_index",
                "boundary_definition": f"distance_to_[0,{config.box_maximum:g}]^2_side_at_most_{config.boundary_shell_thickness:g}",
                "boundary_shell_thickness": (
                    config.boundary_shell_thickness
                ),
                "hole_boundary_policy": "none_in_declared_outer_boundary_model",
                "target_site_count": config.n_sites,
                "target_edge_count": config.n_edges,
                "allowed_degree_range": (
                    config.minimum_degree,
                    config.maximum_degree,
                ),
                "max_point_proposals": config.max_point_proposals,
                "max_complete_attempts": config.max_complete_attempts,
            },
        )

    raise HardCorePlanarGenerationError(
        "hard-core planar construction exhausted "
        f"{config.max_complete_attempts} complete attempts; "
        f"rejected_attempt_reason_counts={dict(sorted(rejected_attempts.items()))!r}"
    )


class _DegenerateTriangleError(ValueError):
    pass


def _normalized_coordinates(
    accepted: list[tuple[float, float, int]],
    *, config: HardCorePlanarConfig = _DEFAULT_CONFIG,
) -> np.ndarray:
    ordered = sorted(accepted, key=lambda item: (item[0], item[1], item[2]))
    coordinates = np.asarray(
        [(x_coordinate, y_coordinate) for x_coordinate, y_coordinate, _ in ordered],
        dtype=float,
    )
    minima = np.min(coordinates, axis=0)
    spans = np.ptp(coordinates, axis=0)
    if np.any(spans <= 0.0):
        raise HardCorePlanarGenerationError(
            "hard-core point set cannot be affinely normalized"
        )
    return np.asarray(
        (coordinates - minima) * (config.box_maximum / spans),
        dtype=float,
    )


def _delaunay_edges(coordinates: np.ndarray, *, config: HardCorePlanarConfig = _DEFAULT_CONFIG) -> tuple[_Edge, ...]:
    triangulation = Delaunay(
        coordinates,
        qhull_options=HARD_CORE_PLANAR_QHULL_OPTIONS,
    )
    edges: set[_Edge] = set()
    for raw_simplex in np.asarray(triangulation.simplices, dtype=np.int64):
        simplex = tuple(int(value) for value in raw_simplex)
        if len(simplex) != 3 or any(
            site < 0 or site >= coordinates.shape[0] for site in simplex
        ):
            raise _DegenerateTriangleError("Delaunay returned an invalid triangle")
        first, second, third = coordinates[np.asarray(simplex, dtype=np.intp)]
        doubled_area = abs(_cross_product_2d(second - first, third - first))
        if doubled_area <= HARD_CORE_PLANAR_TRIANGLE_AREA_TOLERANCE:
            raise _DegenerateTriangleError("Delaunay returned a degenerate triangle")
        for source, target in (
            (simplex[0], simplex[1]),
            (simplex[1], simplex[2]),
            (simplex[2], simplex[0]),
        ):
            edge = (source, target) if source < target else (target, source)
            length = float(np.linalg.norm(coordinates[edge[1]] - coordinates[edge[0]]))
            if length <= config.maximum_edge_length:
                edges.add(edge)
    return tuple(sorted(edges))


def _select_edges(
    coordinates: np.ndarray,
    *,
    candidate_edges: tuple[_Edge, ...],
    priorities: dict[_Edge, int],
    mode: _GeneratorMode,
    prevent_crossings: bool = False,
    config: HardCorePlanarConfig = _DEFAULT_CONFIG,
) -> tuple[tuple[_Edge, ...], tuple[int, ...], str | None]:
    spanning_order = sorted(
        candidate_edges,
        key=lambda edge: (priorities[edge], edge[0], edge[1]),
    )
    disjoint_set = _DisjointSet.create(config.n_sites)
    selected: set[_Edge] = set()
    degrees = [0] * config.n_sites

    def crosses_selected(edge: _Edge) -> bool:
        return prevent_crossings and any(
            not set(edge).intersection(other) and _segments_intersect(
                coordinates[edge[0]], coordinates[edge[1]],
                coordinates[other[0]], coordinates[other[1]]) for other in selected
        )

    for edge in spanning_order:
        if crosses_selected(edge):
            continue
        if not disjoint_set.union(*edge):
            continue
        selected.add(edge)
        degrees[edge[0]] += 1
        degrees[edge[1]] += 1
        if len(selected) == config.n_sites - 1:
            break
    if len(selected) != config.n_sites - 1:
        return (), tuple(degrees), "candidate_graph_disconnected"
    if max(degrees) > config.maximum_degree:
        return (), tuple(degrees), "spanning_tree_degree_limit"

    remaining = (edge for edge in candidate_edges if edge not in selected)
    if mode == "candidate":
        completion_order = sorted(
            remaining,
            key=lambda edge: (priorities[edge], edge[0], edge[1]),
        )
    else:
        completion_order = sorted(
            remaining,
            key=lambda edge: (
                float(np.linalg.norm(coordinates[edge[1]] - coordinates[edge[0]])),
                edge[0],
                edge[1],
            ),
        )
    for edge in completion_order:
        if len(selected) == config.n_edges:
            break
        if (
            degrees[edge[0]] >= config.maximum_degree
            or degrees[edge[1]] >= config.maximum_degree
        ):
            continue
        if crosses_selected(edge):
            continue
        selected.add(edge)
        degrees[edge[0]] += 1
        degrees[edge[1]] += 1
    if len(selected) != config.n_edges:
        return (), tuple(degrees), "edge_budget_unreachable"
    return tuple(sorted(selected)), tuple(degrees), None


def _outer_boundary_sites(coordinates: np.ndarray, *, config: HardCorePlanarConfig = _DEFAULT_CONFIG) -> frozenset[int]:
    maximum = config.box_maximum
    shell = config.boundary_shell_thickness
    return frozenset(
        site
        for site, (x_coordinate, y_coordinate) in enumerate(coordinates)
        if min(
            x_coordinate,
            y_coordinate,
            maximum - x_coordinate,
            maximum - y_coordinate,
        )
        <= shell
    )


def _minimum_separation(coordinates: np.ndarray) -> float:
    differences = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    distances = np.linalg.norm(differences, axis=2)
    np.fill_diagonal(distances, np.inf)
    return float(np.min(distances))


def _has_straight_edge_crossing(
    coordinates: np.ndarray,
    edges: tuple[_Edge, ...],
) -> bool:
    for first_index, first_edge in enumerate(edges):
        for second_edge in edges[first_index + 1 :]:
            if set(first_edge).intersection(second_edge):
                continue
            if _segments_intersect(
                coordinates[first_edge[0]],
                coordinates[first_edge[1]],
                coordinates[second_edge[0]],
                coordinates[second_edge[1]],
            ):
                return True
    return False


def _segments_intersect(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> bool:
    tolerance = HARD_CORE_PLANAR_TRIANGLE_AREA_TOLERANCE

    def orientation(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> float:
        return _cross_product_2d(end - start, point - start)

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


def _nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _cross_product_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])
