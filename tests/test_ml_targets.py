from __future__ import annotations

from dataclasses import replace

import numpy as np
from dataset_fixtures import representative_dataset_record

from toposc_lab.data import ExactPhysicsDataset, TopologyResultRecord, TopologyValidity
from toposc_lab.ml import (
    RegressionTargetDefinition,
    RegressionTargetSource,
    extract_regression_targets,
)


def test_observable_and_robustness_targets_keep_exact_provenance() -> None:
    record = representative_dataset_record()
    dataset = ExactPhysicsDataset((record,))
    gap_definition = RegressionTargetDefinition(
        name="gap",
        source=RegressionTargetSource.OBSERVABLE,
        result_kind="spectral_gap",
        result_version="1",
        value_key="gap",
    )
    robustness_definition = RegressionTargetDefinition(
        name="robustness_success",
        source=RegressionTargetSource.ROBUSTNESS,
        result_kind="onsite_uniform",
        result_version="1",
        value_key="success_fraction",
    )

    gaps = extract_regression_targets(dataset, gap_definition)
    robustness = extract_regression_targets(dataset, robustness_definition)

    np.testing.assert_array_equal(gaps.values, [0.2])
    np.testing.assert_array_equal(robustness.values, [2 / 3])
    assert gaps.valid_values_by_id() == {record.record_id: 0.2}
    assert gaps.dataset_fingerprint.startswith("exact-dataset-v1-sha256:")
    assert gaps.dataset_schema_version == record.schema_version


def test_invalid_topology_is_masked_and_never_coerced_to_a_label() -> None:
    record = representative_dataset_record()
    invalid = replace(
        record,
        topology=(
            TopologyResultRecord(
                method="pfaffian_1d",
                version="1",
                validity=TopologyValidity.NOT_APPLICABLE,
                invariant_value=None,
                is_topological=None,
                parameters={},
                tolerances={},
                reason="geometry is not a valid 1D periodic reference",
            ),
        ),
    )
    definition = RegressionTargetDefinition(
        name="topology_score",
        source=RegressionTargetSource.TOPOLOGY,
        result_kind="pfaffian_1d",
        result_version="1",
        value_key="invariant_value",
    )

    targets = extract_regression_targets(ExactPhysicsDataset((invalid,)), definition)

    assert not targets.valid_mask[0]
    assert np.isnan(targets.values[0])
    assert "not_applicable" in targets.invalid_reasons[0]
    assert targets.valid_values_by_id() == {}
