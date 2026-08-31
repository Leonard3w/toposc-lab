from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.evaluation import GeometryEvaluation, evaluate_geometry_descriptors
from toposc_lab.geometry import (
    Geometry,
    GeometryEdge,
    chain,
    extract_geometry_descriptors,
    ring,
)


def test_chain_descriptors_have_exact_graph_statistics() -> None:
    descriptors = extract_geometry_descriptors(chain(4))

    assert descriptors == {
        "site_count": 4,
        "edge_count": 3,
        "mean_degree": 1.5,
        "degree_variance": 0.25,
        "component_count": 1,
        "largest_component_site_count": 4,
        "is_connected": True,
        "cycle_rank": 0,
        "has_cycles": False,
        "triangle_count": 0,
        "mean_local_clustering": 0.0,
        "connected_pair_count": 6,
        "reachable_pair_fraction": 1.0,
        "mean_finite_shortest_path_length": pytest.approx(10.0 / 6.0),
        "maximum_finite_shortest_path_length": 3,
    }


def test_ring_cycle_and_shortest_path_descriptors() -> None:
    descriptors = extract_geometry_descriptors(ring(4))

    assert descriptors["cycle_rank"] == 1
    assert descriptors["has_cycles"] is True
    assert descriptors["mean_degree"] == pytest.approx(2.0)
    assert descriptors["mean_local_clustering"] == 0.0
    assert descriptors["mean_finite_shortest_path_length"] == pytest.approx(8.0 / 6.0)
    assert descriptors["maximum_finite_shortest_path_length"] == 2


def test_triangle_count_and_clustering_are_not_cycle_rank_aliases() -> None:
    triangle = Geometry(
        n_sites=3,
        edges=(GeometryEdge(0, 1), GeometryEdge(1, 2), GeometryEdge(2, 0)),
    )

    descriptors = extract_geometry_descriptors(triangle)

    assert descriptors["cycle_rank"] == 1
    assert descriptors["triangle_count"] == 1
    assert descriptors["mean_local_clustering"] == pytest.approx(1.0)
    assert descriptors["mean_finite_shortest_path_length"] == pytest.approx(1.0)


def test_disconnected_shortest_paths_are_explicitly_finite_pair_statistics() -> None:
    geometry = Geometry(
        n_sites=5,
        edges=(GeometryEdge(0, 1), GeometryEdge(2, 3)),
    )

    descriptors = extract_geometry_descriptors(geometry)

    assert descriptors["component_count"] == 3
    assert descriptors["largest_component_site_count"] == 2
    assert descriptors["is_connected"] is False
    assert descriptors["connected_pair_count"] == 2
    assert descriptors["reachable_pair_fraction"] == pytest.approx(0.2)
    assert descriptors["mean_finite_shortest_path_length"] == pytest.approx(1.0)
    assert descriptors["maximum_finite_shortest_path_length"] == 1


def test_singleton_has_no_distinct_shortest_path_pair() -> None:
    descriptors = extract_geometry_descriptors(Geometry(n_sites=1))

    assert descriptors["component_count"] == 1
    assert descriptors["mean_degree"] == 0.0
    assert descriptors["degree_variance"] == 0.0
    assert descriptors["connected_pair_count"] == 0
    assert descriptors["reachable_pair_fraction"] == 1.0
    assert descriptors["mean_finite_shortest_path_length"] is None
    assert descriptors["maximum_finite_shortest_path_length"] is None


def test_descriptors_ignore_reference_orientation_and_coordinates() -> None:
    first = Geometry(
        n_sites=3,
        edges=(GeometryEdge(0, 1), GeometryEdge(1, 2)),
        coordinates=np.asarray([[0.0], [1.0], [2.0]]),
    )
    second = Geometry(
        n_sites=3,
        edges=(GeometryEdge(1, 0), GeometryEdge(2, 1)),
        coordinates=np.asarray([[10.0, 0.0], [-3.0, 2.0], [7.0, 9.0]]),
    )

    assert extract_geometry_descriptors(first) == extract_geometry_descriptors(second)


def test_descriptor_mapping_is_read_only() -> None:
    descriptors = extract_geometry_descriptors(chain(2))

    with pytest.raises(TypeError):
        descriptors["site_count"] = 3  # type: ignore[index]


def test_evaluation_populates_only_geometry_descriptor_field() -> None:
    initial = GeometryEvaluation(
        gap=0.4,
        low_energy_states={0: -0.2, 1: 0.2},
        zero_mode_count=0,
        warnings=("Existing warning.",),
    )

    result = evaluate_geometry_descriptors(initial, chain(4))

    assert result.gap == initial.gap
    assert result.low_energy_states == initial.low_energy_states
    assert result.ipr == {}
    assert result.majorana_metrics == {}
    assert result.topology == ()
    assert result.geometry_descriptors["site_count"] == 4
    assert result.warnings[0] == "Existing warning."
    assert any("undirected one-skeleton" in item for item in result.warnings)


def test_evaluation_warns_about_disconnected_path_scope() -> None:
    result = evaluate_geometry_descriptors(
        GeometryEvaluation(),
        Geometry(n_sites=3, edges=(GeometryEdge(0, 1),)),
    )

    assert any("geometry is disconnected" in item for item in result.warnings)
    assert result.geometry_descriptors["reachable_pair_fraction"] == pytest.approx(1.0 / 3.0)


def test_descriptor_reintegration_is_warning_idempotent() -> None:
    first = evaluate_geometry_descriptors(GeometryEvaluation(), chain(3))
    second = evaluate_geometry_descriptors(first, chain(3))

    assert second.geometry_descriptors == first.geometry_descriptors
    assert second.warnings == first.warnings


def test_descriptor_extraction_rejects_non_geometry() -> None:
    with pytest.raises(TypeError, match="geometry must be a Geometry"):
        extract_geometry_descriptors(object())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="geometry must be a Geometry"):
        evaluate_geometry_descriptors(
            GeometryEvaluation(),
            object(),  # type: ignore[arg-type]
        )
