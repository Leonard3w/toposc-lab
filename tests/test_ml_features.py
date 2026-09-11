from __future__ import annotations

from dataclasses import replace

import numpy as np
from dataset_fixtures import representative_dataset_record

from toposc_lab.data import ModelParametersRecord
from toposc_lab.geometry import Geometry, GeometryEdge, chain
from toposc_lab.ml import HandcraftedFeatureSchema, extract_handcrafted_features


def test_handcrafted_features_are_stable_and_use_only_inputs() -> None:
    first = representative_dataset_record(geometry=chain(4), seed=1)
    changed_labels = replace(
        first,
        spectrum=replace(first.spectrum, eigenvalues=(-4.0, -2.0, 2.0, 4.0)),
        observables=(),
        topology=(),
        robustness=(),
    )
    schema = HandcraftedFeatureSchema.fit((first,))

    original = extract_handcrafted_features((first,), schema=schema)
    relabeled = extract_handcrafted_features((changed_labels,), schema=schema)

    np.testing.assert_array_equal(original.values, relabeled.values)
    assert original.values.shape[1] == len(schema.feature_names)
    assert "parameter::pairing::real" in schema.feature_names
    assert "parameter::pairing::imag" in schema.feature_names


def test_features_are_invariant_to_site_relabeling_for_abstract_graph() -> None:
    original_geometry = Geometry(
        n_sites=4,
        edges=(GeometryEdge(0, 1), GeometryEdge(1, 2), GeometryEdge(2, 3)),
    )
    relabeled_geometry = Geometry(
        n_sites=4,
        edges=(GeometryEdge(2, 0), GeometryEdge(0, 3), GeometryEdge(3, 1)),
    )
    first = representative_dataset_record(geometry=original_geometry, seed=1)
    second = representative_dataset_record(geometry=relabeled_geometry, seed=2)
    schema = HandcraftedFeatureSchema.fit((first,))

    matrix = extract_handcrafted_features((first, second), schema=schema)

    np.testing.assert_allclose(matrix.values[0], matrix.values[1])


def test_schema_is_fit_on_training_parameters_and_marks_missing_values() -> None:
    train = representative_dataset_record(seed=1)
    missing = replace(
        representative_dataset_record(seed=2),
        model=ModelParametersRecord(
            model_name=train.model.model_name,
            model_version=train.model.model_version,
            parameters={"mu": 0.5, "unseen_test_parameter": 99.0},
        ),
    )
    schema = HandcraftedFeatureSchema.fit((train,))

    matrix = extract_handcrafted_features((missing,), schema=schema)

    assert not any("unseen_test_parameter" in name for name in schema.feature_names)
    pairing_missing = schema.feature_names.index("parameter::pairing::missing")
    assert matrix.values[0, pairing_missing] == 1.0
