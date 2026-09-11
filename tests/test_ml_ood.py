from __future__ import annotations

import numpy as np
from dataset_fixtures import representative_dataset_record

from toposc_lab.geometry import Geometry, GeometryEdge, chain
from toposc_lab.ml import (
    OOD_CAUTION,
    FeatureOODDetector,
    HandcraftedFeatureSchema,
    extract_handcrafted_features,
)


def test_unusual_generated_graph_is_flagged_without_physics_claim() -> None:
    training = tuple(
        representative_dataset_record(
            geometry=chain(size), seed=size, family_label=f"chain_{size}"
        )
        for size in range(4, 13)
    )
    unusual_geometry = Geometry(
        n_sites=40,
        edges=tuple(GeometryEdge(0, site) for site in range(1, 40)),
    )
    unusual = representative_dataset_record(
        geometry=unusual_geometry, seed=99, family_label="large_star"
    )
    schema = HandcraftedFeatureSchema.fit(training)
    train_features = extract_handcrafted_features(training, schema=schema)
    unusual_features = extract_handcrafted_features((unusual,), schema=schema)
    detector = FeatureOODDetector(reference_quantile=0.9).fit(train_features.values)

    result = detector.assess(
        unusual_features.values, record_ids=unusual_features.record_ids
    )

    assert result.is_ood[0]
    assert result.record_ids == (unusual.record_id,)
    assert result.caution == OOD_CAUTION
    assert "Exact simulation remains required" in result.caution


def test_ood_detector_is_reproducible_for_in_distribution_points() -> None:
    training = np.arange(20, dtype=float).reshape(-1, 1)
    first = FeatureOODDetector(reference_quantile=0.9).fit(training)
    second = FeatureOODDetector(reference_quantile=0.9).fit(training)

    first_result = first.assess([[10.5]], record_ids=("candidate",))
    second_result = second.assess([[10.5]], record_ids=("candidate",))

    np.testing.assert_array_equal(first_result.scores, second_result.scores)
    assert not first_result.is_ood[0]
