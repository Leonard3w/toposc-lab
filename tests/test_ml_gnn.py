from __future__ import annotations

import numpy as np
from dataset_fixtures import representative_dataset_record

from toposc_lab.geometry import chain, ring
from toposc_lab.ml import (
    GraphInputSchema,
    MinimalGNNRegressor,
    graph_inputs_from_records,
    regression_metrics,
)


def test_minimal_gnn_is_reproducible_and_trainable() -> None:
    records = tuple(
        representative_dataset_record(
            geometry=chain(size) if size % 2 else ring(size),
            seed=size,
            family_label=f"graph_{size}",
        )
        for size in range(3, 17)
    )
    targets = np.asarray([record.geometry.to_geometry().n_sites / 10.0 for record in records])
    schema = GraphInputSchema.fit(records)
    graphs = graph_inputs_from_records(records, schema=schema)
    config = {"hidden_dimension": 8, "epochs": 700, "learning_rate": 0.008, "seed": 7}

    first = MinimalGNNRegressor(**config).fit(graphs, targets)
    second = MinimalGNNRegressor(**config).fit(graphs, targets)
    first_predictions = first.predict(graphs)

    np.testing.assert_array_equal(first_predictions, second.predict(graphs))
    assert regression_metrics(targets, first_predictions).rmse < 0.12


def test_minimal_gnn_predictions_are_permutation_invariant() -> None:
    from test_ml_graph import _graph_record

    first = _graph_record()
    second = _graph_record(relabeled=True)
    records = (first, second)
    schema = GraphInputSchema.fit(records)
    graphs = graph_inputs_from_records(records, schema=schema)
    model = MinimalGNNRegressor(epochs=100, seed=3).fit(graphs, [1.0, 1.0])

    predictions = model.predict(graphs)
    np.testing.assert_allclose(predictions[0], predictions[1], atol=1e-12)
