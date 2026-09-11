from __future__ import annotations

from dataclasses import replace

import numpy as np
from dataset_fixtures import representative_dataset_record

from toposc_lab.geometry import Geometry, GeometryEdge
from toposc_lab.ml import GraphInputSchema, graph_input_from_record


def _graph_record(*, relabeled: bool = False):
    if relabeled:
        geometry = Geometry(
            n_sites=3,
            edges=(GeometryEdge(2, 0, edge_type="hop"), GeometryEdge(0, 1, edge_type="hop")),
            coordinates=np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 0.0]]),
            boundary_sites=frozenset({1, 2}),
            site_types=("bulk", "end", "end"),
        )
    else:
        geometry = Geometry(
            n_sites=3,
            edges=(GeometryEdge(0, 1, edge_type="hop"), GeometryEdge(1, 2, edge_type="hop")),
            coordinates=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            boundary_sites=frozenset({0, 2}),
            site_types=("end", "bulk", "end"),
        )
    return representative_dataset_record(geometry=geometry, seed=1)


def test_graph_input_has_bidirectional_edges_and_exact_provenance() -> None:
    record = _graph_record()
    schema = GraphInputSchema.fit((record,))

    graph = graph_input_from_record(record, schema=schema)

    assert graph.node_features.shape[0] == 3
    assert graph.edge_index.shape == (2, 4)
    assert graph.edge_features.shape[0] == 4
    assert graph.global_features.ndim == 1
    assert graph.record_id == record.record_id
    assert graph.geometry_exact_id == record.geometry.exact_id
    assert graph.dataset_schema_version == record.schema_version


def test_graph_representation_does_not_read_exact_labels() -> None:
    record = _graph_record()
    changed = replace(record, observables=(), topology=(), robustness=())
    schema = GraphInputSchema.fit((record,))

    original = graph_input_from_record(record, schema=schema)
    relabeled = graph_input_from_record(changed, schema=schema)

    np.testing.assert_array_equal(original.node_features, relabeled.node_features)
    np.testing.assert_array_equal(original.edge_features, relabeled.edge_features)
    np.testing.assert_array_equal(original.global_features, relabeled.global_features)


def test_graph_level_aggregates_are_relabeling_invariant() -> None:
    first = _graph_record()
    second = _graph_record(relabeled=True)
    schema = GraphInputSchema.fit((first,))

    graph_a = graph_input_from_record(first, schema=schema)
    graph_b = graph_input_from_record(second, schema=schema)

    np.testing.assert_allclose(np.mean(graph_a.node_features, axis=0), np.mean(graph_b.node_features, axis=0))
    np.testing.assert_allclose(np.mean(graph_a.edge_features, axis=0), np.mean(graph_b.edge_features, axis=0))
