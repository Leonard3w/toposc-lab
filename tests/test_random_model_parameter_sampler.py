from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

import toposc_lab.search.random_model_parameters as sampler_module
from toposc_lab.models.chiral_p_wave import ChiralPWaveParameters
from toposc_lab.robustness import (
    DisorderParameterValue,
    exact_model_parameter_set_id,
)
from toposc_lab.search import (
    RANDOM_MODEL_PARAMETER_SAMPLER_RNG_ALGORITHM,
    RANDOM_MODEL_PARAMETER_SAMPLER_VERSION,
    CategoricalDistribution,
    RandomModelParameterSamplingConfig,
    RandomModelParameterSamplingError,
    UniformFloatDistribution,
    UniformIntegerDistribution,
    sample_random_model_parameters,
)


def _mixed_config(*, sample_count: int = 5) -> RandomModelParameterSamplingConfig:
    return RandomModelParameterSamplingConfig(
        base_parameters={
            "hopping": 1.0,
            "chemical_potential": 0.0,
            "pairing": 0.5,
            "chirality": 1,
            "copies": 2,
            "plane_axes": (0, 1),
            "label": "reference",
        },
        distributions={
            "pairing": UniformFloatDistribution(0.2, 0.9),
            "copies": UniformIntegerDistribution(2, 6),
            "chirality": CategoricalDistribution((-1, 1)),
            "chemical_potential": UniformFloatDistribution(-2.0, 2.0),
        },
        sample_count=sample_count,
    )


def test_fixed_pcg64_sample_and_lexicographic_draw_order() -> None:
    result = sample_random_model_parameters(
        _mixed_config(sample_count=4),
        seed=20260901,
    )

    assert tuple(dict(sample.parameters) for sample in result.samples) == (
        {
            "chemical_potential": 0.5287207363037645,
            "chirality": 1,
            "copies": 4,
            "hopping": 1.0,
            "label": "reference",
            "pairing": 0.4479560985815304,
            "plane_axes": (0, 1),
        },
        {
            "chemical_potential": -0.8375762003091589,
            "chirality": -1,
            "copies": 2,
            "hopping": 1.0,
            "label": "reference",
            "pairing": 0.20384048583356085,
            "plane_axes": (0, 1),
        },
        {
            "chemical_potential": -1.1955271805228027,
            "chirality": 1,
            "copies": 3,
            "hopping": 1.0,
            "label": "reference",
            "pairing": 0.23649045343071418,
            "plane_axes": (0, 1),
        },
        {
            "chemical_potential": 1.8746490466612222,
            "chirality": 1,
            "copies": 4,
            "hopping": 1.0,
            "label": "reference",
            "pairing": 0.848777998699618,
            "plane_axes": (0, 1),
        },
    )


def test_same_configuration_and_seed_reproduce_exact_ordered_parameter_sets() -> None:
    config = _mixed_config()

    first = sample_random_model_parameters(config, seed=123456)
    second = sample_random_model_parameters(config, seed=123456)

    assert first.seed == second.seed == 123456
    assert first.rng_algorithm == second.rng_algorithm
    assert first.sampler_version == second.sampler_version
    assert tuple(sample.sample_index for sample in first.samples) == tuple(
        sample.sample_index for sample in second.samples
    )
    assert tuple(
        exact_model_parameter_set_id(sample.parameters) for sample in first.samples
    ) == tuple(
        exact_model_parameter_set_id(sample.parameters) for sample in second.samples
    )


def test_reordered_input_mappings_have_identical_sampling_semantics() -> None:
    first_config = RandomModelParameterSamplingConfig(
        base_parameters={"fixed": "x", "b": 0.0, "a": 0.0},
        distributions={
            "b": UniformFloatDistribution(10.0, 20.0),
            "a": UniformFloatDistribution(-1.0, 1.0),
        },
        sample_count=4,
    )
    second_config = RandomModelParameterSamplingConfig(
        base_parameters={"a": 0.0, "b": 0.0, "fixed": "x"},
        distributions={
            "a": UniformFloatDistribution(-1.0, 1.0),
            "b": UniformFloatDistribution(10.0, 20.0),
        },
        sample_count=4,
    )

    first = sample_random_model_parameters(first_config, seed=314)
    second = sample_random_model_parameters(second_config, seed=314)

    assert tuple(first_config.base_parameters) == ("a", "b", "fixed")
    assert tuple(first_config.distributions) == ("a", "b")
    assert tuple(
        exact_model_parameter_set_id(sample.parameters) for sample in first.samples
    ) == tuple(
        exact_model_parameter_set_id(sample.parameters) for sample in second.samples
    )


def test_result_records_version_rng_and_strict_sample_order() -> None:
    result = sample_random_model_parameters(_mixed_config(sample_count=3), seed=9)

    assert result.rng_algorithm == RANDOM_MODEL_PARAMETER_SAMPLER_RNG_ALGORITHM
    assert result.sampler_version == RANDOM_MODEL_PARAMETER_SAMPLER_VERSION
    assert tuple(sample.sample_index for sample in result.samples) == (0, 1, 2)
    assert all(
        tuple(sample.parameters) == tuple(result.config.base_parameters)
        for sample in result.samples
    )


def test_distribution_outputs_have_plain_exact_parameter_types() -> None:
    result = sample_random_model_parameters(_mixed_config(sample_count=10), seed=42)

    for sample in result.samples:
        assert type(sample.parameters["chemical_potential"]) is float
        assert type(sample.parameters["pairing"]) is float
        assert type(sample.parameters["copies"]) is int
        assert type(sample.parameters["chirality"]) is int
        assert sample.parameters["chirality"] in (-1, 1)
        assert sample.parameters["plane_axes"] == (0, 1)
        assert sample.parameters["hopping"] == 1.0
        assert sample.parameters["label"] == "reference"


def test_sampled_mapping_can_be_explicitly_validated_by_a_parameter_model() -> None:
    config = RandomModelParameterSamplingConfig(
        base_parameters={
            "hopping": 1.0,
            "chemical_potential": 0.0,
            "pairing": 0.5,
            "chirality": 1,
            "plane_axes": (0, 1),
        },
        distributions={
            "chemical_potential": UniformFloatDistribution(-2.0, 2.0),
            "pairing": UniformFloatDistribution(0.1, 1.0),
            "chirality": CategoricalDistribution((-1, 1)),
        },
        sample_count=5,
    )

    result = sample_random_model_parameters(config, seed=7)
    validated = tuple(
        ChiralPWaveParameters.model_validate(dict(sample.parameters))
        for sample in result.samples
    )

    assert len(validated) == 5
    assert all(parameters.plane_axes == (0, 1) for parameters in validated)
    assert all(parameters.chirality in (-1, 1) for parameters in validated)


def test_configuration_and_samples_are_deeply_immutable() -> None:
    nested: dict[str, Any] = {"axes": [0, 1]}
    choices: list[Any] = [{"mode": ["a", "b"]}, {"mode": ["c"]}]
    config = RandomModelParameterSamplingConfig(
        base_parameters={
            "fixed": nested,
            "selected": {"mode": ["base"]},  # type: ignore[dict-item]
        },
        distributions={
            "selected": CategoricalDistribution(choices),  # type: ignore[arg-type]
        },
        sample_count=1,
    )
    nested["axes"].append(2)
    choices[0]["mode"].append("changed")

    sample = sample_random_model_parameters(config, seed=3).samples[0]

    assert config.base_parameters["fixed"] == {"axes": (0, 1)}
    assert sample.parameters["fixed"] == {"axes": (0, 1)}
    with pytest.raises(TypeError):
        config.base_parameters["new"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        sample.parameters["new"] = 1  # type: ignore[index]
    selected = sample.parameters["selected"]
    assert isinstance(selected, Mapping)
    with pytest.raises(TypeError):
        selected["mode"] = ()  # type: ignore[index]


def test_categorical_duplicates_are_exact_and_type_sensitive() -> None:
    typed_distinct = CategoricalDistribution((1, 1.0, True))

    assert typed_distinct.choices == (1, 1.0, True)
    with pytest.raises(ValueError, match="exact duplicate"):
        CategoricalDistribution(
            ({"value": [1]}, {"value": (1,)})  # type: ignore[dict-item]
        )


def test_sampler_does_not_read_or_modify_global_numpy_random_state() -> None:
    np.random.seed(9182)
    expected = np.random.random(4)
    np.random.seed(9182)

    sample_random_model_parameters(_mixed_config(sample_count=4), seed=7)

    assert np.array_equal(np.random.random(4), expected)


def test_different_root_seed_changes_sampled_values() -> None:
    first = sample_random_model_parameters(_mixed_config(), seed=1)
    second = sample_random_model_parameters(_mixed_config(), seed=2)

    assert tuple(
        exact_model_parameter_set_id(sample.parameters) for sample in first.samples
    ) != tuple(
        exact_model_parameter_set_id(sample.parameters) for sample in second.samples
    )


@pytest.mark.parametrize(
    ("low", "high", "error_type", "message"),
    (
        (0.0, 0.0, ValueError, "smaller than high"),
        (1.0, -1.0, ValueError, "smaller than high"),
        (0.0, np.inf, ValueError, "must be finite"),
        (-1.0e308, 1.0e308, ValueError, "interval width must be finite"),
        (True, 1.0, TypeError, "real number"),
        ("0", 1.0, TypeError, "real number"),
    ),
)
def test_uniform_float_distribution_rejects_invalid_bounds(
    low: object,
    high: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        UniformFloatDistribution(low, high)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("low", "high", "error_type", "message"),
    (
        (0, 0, ValueError, "smaller than high"),
        (2, 1, ValueError, "smaller than high"),
        (-(2**63) - 1, 0, ValueError, "int64"),
        (0, 2**63 + 1, ValueError, "int64"),
        (True, 2, TypeError, "integer"),
        (0.0, 2, TypeError, "integer"),
    ),
)
def test_uniform_integer_distribution_rejects_invalid_bounds(
    low: object,
    high: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        UniformIntegerDistribution(low, high)  # type: ignore[arg-type]


def test_categorical_distribution_rejects_empty_or_nonportable_choices() -> None:
    with pytest.raises(ValueError, match="at least one"):
        CategoricalDistribution(())
    with pytest.raises(TypeError, match="unsupported type"):
        CategoricalDistribution((object(),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be finite"):
        CategoricalDistribution((np.nan,))


@pytest.mark.parametrize("sample_count", (0, -1, -10))
def test_config_rejects_nonpositive_sample_count(sample_count: int) -> None:
    with pytest.raises(ValueError, match="sample_count must be positive"):
        RandomModelParameterSamplingConfig(
            base_parameters={"value": 0.0},
            distributions={"value": UniformFloatDistribution(0.0, 1.0)},
            sample_count=sample_count,
        )


@pytest.mark.parametrize("sample_count", (True, 1.5, "2"))
def test_config_rejects_noninteger_sample_count(sample_count: object) -> None:
    with pytest.raises(TypeError, match="sample_count must be an integer"):
        RandomModelParameterSamplingConfig(
            base_parameters={"value": 0.0},
            distributions={"value": UniformFloatDistribution(0.0, 1.0)},
            sample_count=sample_count,  # type: ignore[arg-type]
        )


def test_config_rejects_empty_unknown_or_unsupported_distributions() -> None:
    with pytest.raises(ValueError, match="at least one parameter"):
        RandomModelParameterSamplingConfig(
            base_parameters={"value": 0.0},
            distributions={},
            sample_count=1,
        )
    with pytest.raises(ValueError, match="unknown model parameters.*missing"):
        RandomModelParameterSamplingConfig(
            base_parameters={"value": 0.0},
            distributions={"missing": UniformFloatDistribution(0.0, 1.0)},
            sample_count=1,
        )
    with pytest.raises(TypeError, match="unsupported type object"):
        RandomModelParameterSamplingConfig(
            base_parameters={"value": 0.0},
            distributions={"value": object()},  # type: ignore[dict-item]
            sample_count=1,
        )


def test_config_rejects_invalid_parameter_keys_and_values() -> None:
    with pytest.raises(ValueError, match="Python-style parameter names"):
        RandomModelParameterSamplingConfig(
            base_parameters={"not-valid": 0.0},
            distributions={"not-valid": UniformFloatDistribution(0.0, 1.0)},
            sample_count=1,
        )
    with pytest.raises(TypeError, match="unsupported type object"):
        RandomModelParameterSamplingConfig(
            base_parameters={"value": object()},  # type: ignore[dict-item]
            distributions={"value": UniformFloatDistribution(0.0, 1.0)},
            sample_count=1,
        )


@pytest.mark.parametrize("seed", (-1, -10))
def test_sampler_rejects_negative_root_seed(seed: int) -> None:
    with pytest.raises(ValueError, match="seed must be nonnegative"):
        sample_random_model_parameters(_mixed_config(sample_count=1), seed=seed)


@pytest.mark.parametrize("seed", (True, 1.5, "1"))
def test_sampler_rejects_noninteger_root_seed(seed: object) -> None:
    with pytest.raises(TypeError, match="seed must be an integer"):
        sample_random_model_parameters(
            _mixed_config(sample_count=1),
            seed=seed,  # type: ignore[arg-type]
        )


def test_distribution_failure_stops_and_reports_exact_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = sampler_module._sample_distribution

    def fail_on_second_draw(
        distribution: sampler_module.ModelParameterDistribution,
        *,
        rng: np.random.Generator,
    ) -> DisorderParameterValue:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ArithmeticError("synthetic draw failure")
        return original(distribution, rng=rng)

    monkeypatch.setattr(sampler_module, "_sample_distribution", fail_on_second_draw)
    config = RandomModelParameterSamplingConfig(
        base_parameters={"value": 0.0},
        distributions={"value": UniformFloatDistribution(0.0, 1.0)},
        sample_count=4,
    )

    with pytest.raises(RandomModelParameterSamplingError) as captured:
        sample_random_model_parameters(config, seed=8)

    error = captured.value
    assert calls == 2
    assert error.sample_index == 1
    assert error.parameter_name == "value"
    assert error.distribution_name == "UniformFloatDistribution"
    assert error.error_type == "ArithmeticError"
    assert error.detail == "synthetic draw failure"
    assert isinstance(error.__cause__, ArithmeticError)
