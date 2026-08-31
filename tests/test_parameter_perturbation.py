from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from toposc_lab.models.chiral_p_wave import ChiralPWaveParameters
from toposc_lab.robustness import (
    DISORDER_RNG_ALGORITHM,
    MODEL_PARAMETER_SET_ID_SCHEME,
    UNIFORM_PARAMETER_PERTURBATION_KEY,
    UNIFORM_PARAMETER_PERTURBATION_VERSION,
    DisorderParameterValue,
    DisorderRequest,
    DisorderState,
    DisorderTarget,
    FunctionDisorderTransform,
    apply_uniform_parameter_perturbation,
    exact_model_parameter_set_id,
    realize_disorder,
)


def _model_parameters() -> dict[str, DisorderParameterValue]:
    return {
        "hopping": 1.0,
        "chemical_potential": 0.5,
        "pairing": 0.8,
        "chirality": 1,
        "plane_axes": (0, 1),
        "pairing_convention": "chiral_p_wave",
        "options": {"normalize_directions": True},
    }


def test_fixed_pcg64_sample_uses_lexicographic_parameter_order() -> None:
    parameters = _model_parameters()

    realization = apply_uniform_parameter_perturbation(
        parameters,
        widths={"pairing": 0.4, "hopping": 1.0, "chemical_potential": 0.2},
        seed=42,
    )

    assert isinstance(realization.state, Mapping)
    assert realization.state["chemical_potential"] == pytest.approx(
        0.5547912097111927
    )
    assert realization.state["hopping"] == pytest.approx(0.9388784397520523)
    assert realization.state["pairing"] == pytest.approx(0.943439167964553)
    assert realization.state["chirality"] == 1
    assert realization.state["plane_axes"] == (0, 1)
    assert realization.state["pairing_convention"] == "chiral_p_wave"
    assert realization.state["options"] == {"normalize_directions": True}


def test_same_seed_and_reordered_inputs_reproduce_snapshot_and_provenance() -> None:
    first = apply_uniform_parameter_perturbation(
        {"pairing": 0.8, "mass": -0.25, "label": "first"},
        widths={"pairing": 0.2, "mass": 0.5},
        seed=1729,
    )
    second = apply_uniform_parameter_perturbation(
        {"label": "first", "mass": -0.25, "pairing": 0.8},
        widths={"mass": 0.5, "pairing": 0.2},
        seed=1729,
    )

    assert isinstance(first.state, Mapping)
    assert isinstance(second.state, Mapping)
    assert first.state == second.state
    assert first.provenance == second.provenance
    assert exact_model_parameter_set_id(first.state) == (
        exact_model_parameter_set_id(second.state)
    )


def test_parameter_snapshots_are_typed_order_independent_and_exact() -> None:
    ordered_differently: dict[str, DisorderParameterValue] = {
        "nested": {"flag": True},
        "value": 1.0,
    }

    assert exact_model_parameter_set_id(ordered_differently) == (
        exact_model_parameter_set_id(
            {"value": 1.0, "nested": {"flag": True}}
        )
    )
    assert exact_model_parameter_set_id({"value": 1}) != (
        exact_model_parameter_set_id({"value": 1.0})
    )
    assert exact_model_parameter_set_id({"value": -0.0}) != (
        exact_model_parameter_set_id({"value": 0.0})
    )


def test_source_and_result_parameter_mappings_are_deeply_immutable() -> None:
    nested = {"values": [1, 2]}
    parameters: dict[str, DisorderParameterValue] = {
        "hopping": 1.0,
        "nested": nested,  # type: ignore[dict-item]
    }

    realization = apply_uniform_parameter_perturbation(
        parameters,
        widths={"hopping": 0.2},
        seed=5,
    )
    nested["values"].append(3)

    assert isinstance(realization.state, Mapping)
    assert realization.state["nested"] == {"values": (1, 2)}
    with pytest.raises(TypeError):
        realization.state["hopping"] = 2.0  # type: ignore[index]
    nested_result = realization.state["nested"]
    assert isinstance(nested_result, Mapping)
    with pytest.raises(TypeError):
        nested_result["values"] = ()  # type: ignore[index]


def test_result_can_be_explicitly_used_to_construct_a_model_parameter_object() -> None:
    realization = apply_uniform_parameter_perturbation(
        {
            "hopping": 1.0,
            "chemical_potential": 0.5,
            "pairing": 0.8,
            "chirality": 1,
            "plane_axes": (0, 1),
        },
        widths={"chemical_potential": 0.2, "pairing": 0.1},
        seed=7,
    )

    assert isinstance(realization.state, Mapping)
    reconstructed = ChiralPWaveParameters.model_validate(dict(realization.state))
    assert reconstructed.chirality == 1
    assert reconstructed.plane_axes == (0, 1)
    assert reconstructed.chemical_potential == realization.state[
        "chemical_potential"
    ]


def test_zero_and_empty_widths_produce_exact_no_op_snapshots() -> None:
    parameters = _model_parameters()

    zero = apply_uniform_parameter_perturbation(
        parameters,
        widths={"hopping": 0.0},
        seed=11,
    )
    empty = apply_uniform_parameter_perturbation(
        parameters,
        widths={},
        seed=12,
    )

    assert zero.provenance.source == zero.provenance.result
    assert empty.provenance.source == empty.provenance.result
    assert zero.state == parameters
    assert empty.state == parameters


def test_provenance_records_parameter_selection_and_snapshot_relationship() -> None:
    parameters = _model_parameters()
    realization = apply_uniform_parameter_perturbation(
        parameters,
        widths={"pairing": 0.4, "hopping": 0.2},
        seed=17,
    )

    assert realization.provenance.disorder_key == (
        UNIFORM_PARAMETER_PERTURBATION_KEY
    )
    assert realization.provenance.disorder_version == (
        UNIFORM_PARAMETER_PERTURBATION_VERSION
    )
    assert realization.provenance.seed == 17
    assert realization.provenance.rng_algorithm == DISORDER_RNG_ALGORITHM
    assert realization.provenance.parameters == {
        "constraint_policy": "no_clipping_resampling_or_model_inference",
        "distribution": "uniform",
        "key_policy": "preserve_exact_top_level_parameter_keys",
        "parameter_count": 7,
        "parameter_iteration_order": "lexicographic_parameter_name",
        "perturbed_parameters": ("hopping", "pairing"),
        "sampling_rule": "independent_additive_offset_per_selected_parameter",
        "selection_rule": "explicit_top_level_continuous_float_parameters_only",
        "source_parameter_set_id": exact_model_parameter_set_id(parameters),
        "support": "[-width/2, width/2]",
        "widths": {"hopping": 0.2, "pairing": 0.4},
    }
    assert realization.provenance.source.target is DisorderTarget.MODEL_PARAMETERS
    assert realization.provenance.source.scheme == MODEL_PARAMETER_SET_ID_SCHEME
    assert realization.provenance.source.identifier == (
        exact_model_parameter_set_id(parameters)
    )
    assert isinstance(realization.state, Mapping)
    assert realization.provenance.result.identifier == (
        exact_model_parameter_set_id(realization.state)
    )


def test_execution_does_not_read_or_modify_global_numpy_random_state() -> None:
    np.random.seed(9182)
    expected = np.random.random(4)
    np.random.seed(9182)

    apply_uniform_parameter_perturbation(
        {"hopping": 1.0},
        widths={"hopping": 0.2},
        seed=7,
    )

    assert np.array_equal(np.random.random(4), expected)


def test_common_executor_rejects_parameter_key_changes_and_wrong_state_kind() -> None:
    def drop_parameter(
        source: DisorderState,
        parameters: Mapping[str, DisorderParameterValue],
        rng: np.random.Generator,
    ) -> DisorderState:
        del parameters, rng
        assert isinstance(source, Mapping)
        return {"hopping": source["hopping"]}

    transform = FunctionDisorderTransform(
        key="invalid_parameter_key_change",
        target=DisorderTarget.MODEL_PARAMETERS,
        function=drop_parameter,
    )
    with pytest.raises(ValueError, match="preserve the parameter keys"):
        realize_disorder(
            {"hopping": 1.0, "pairing": 0.5},
            transform=transform,
            request=DisorderRequest(seed=1),
        )
    with pytest.raises(TypeError, match="requires and returns a parameter mapping"):
        realize_disorder(
            np.eye(2),
            transform=transform,
            request=DisorderRequest(seed=1),
        )


@pytest.mark.parametrize(
    ("parameter", "value"),
    [("chirality", 1), ("enabled", True), ("boundary", "open")],
)
def test_discrete_or_nonnumeric_parameters_cannot_be_selected(
    parameter: str,
    value: DisorderParameterValue,
) -> None:
    with pytest.raises(TypeError, match="must be a continuous float"):
        apply_uniform_parameter_perturbation(
            {parameter: value},
            widths={parameter: 0.2},
            seed=1,
        )


def test_unknown_parameter_width_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown model parameter"):
        apply_uniform_parameter_perturbation(
            {"hopping": 1.0},
            widths={"pairing": 0.2},
            seed=1,
        )


@pytest.mark.parametrize("width", [-0.1, np.inf, np.nan])
def test_invalid_width_is_rejected(width: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        apply_uniform_parameter_perturbation(
            {"hopping": 1.0},
            widths={"hopping": width},
            seed=1,
        )


@pytest.mark.parametrize("width", [True, "wide"])
def test_nonreal_width_is_rejected(width: object) -> None:
    with pytest.raises(TypeError, match="must be a real number"):
        apply_uniform_parameter_perturbation(
            {"hopping": 1.0},
            widths={"hopping": width},  # type: ignore[dict-item]
            seed=1,
        )
