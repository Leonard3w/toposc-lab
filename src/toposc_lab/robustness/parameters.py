"""Reproducible additive perturbation of explicit physical model parameters."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np

from toposc_lab.robustness.disorder import (
    DisorderParameterValue,
    DisorderRealization,
    DisorderRequest,
    DisorderState,
    DisorderTarget,
    FunctionDisorderTransform,
    ModelParameterSet,
    exact_model_parameter_set_id,
    realize_disorder,
)

UNIFORM_PARAMETER_PERTURBATION_KEY = "uniform_parameter_perturbation"
UNIFORM_PARAMETER_PERTURBATION_VERSION = 1


def apply_uniform_parameter_perturbation(
    parameters: ModelParameterSet,
    *,
    widths: Mapping[str, float],
    seed: int,
) -> DisorderRealization:
    r"""Perturb selected continuous parameters within symmetric uniform widths.

    Only top-level floating-point parameters named by ``widths`` are changed.
    Discrete and structural parameter values remain part of the immutable
    parameter snapshot but cannot be selected by this additive transform.
    """
    source_id = exact_model_parameter_set_id(parameters)
    normalized_widths = _validated_widths(widths, parameters=parameters)
    parameter_names = tuple(normalized_widths)
    request = DisorderRequest(
        seed=seed,
        parameters={
            "distribution": "uniform",
            "support": "[-width/2, width/2]",
            "widths": normalized_widths,
            "source_parameter_set_id": source_id,
            "parameter_count": len(parameters),
            "perturbed_parameters": parameter_names,
            "parameter_iteration_order": "lexicographic_parameter_name",
            "sampling_rule": "independent_additive_offset_per_selected_parameter",
            "selection_rule": "explicit_top_level_continuous_float_parameters_only",
            "key_policy": "preserve_exact_top_level_parameter_keys",
            "constraint_policy": "no_clipping_resampling_or_model_inference",
        },
    )

    def transform(
        source: DisorderState,
        recorded_parameters: Mapping[str, DisorderParameterValue],
        rng: np.random.Generator,
    ) -> DisorderState:
        assert isinstance(source, Mapping)
        recorded_widths = recorded_parameters["widths"]
        if not isinstance(recorded_widths, Mapping):
            raise TypeError("recorded parameter widths must be a mapping")
        return _apply_parameter_offsets(
            source,
            widths=recorded_widths,
            rng=rng,
        )

    disorder_transform = FunctionDisorderTransform(
        key=UNIFORM_PARAMETER_PERTURBATION_KEY,
        version=UNIFORM_PARAMETER_PERTURBATION_VERSION,
        target=DisorderTarget.MODEL_PARAMETERS,
        function=transform,
    )
    return realize_disorder(
        parameters,
        transform=disorder_transform,
        request=request,
    )


def _validated_widths(
    widths: Mapping[str, float],
    *,
    parameters: ModelParameterSet,
) -> dict[str, float]:
    if not isinstance(widths, Mapping):
        raise TypeError("widths must be a mapping")
    if any(not isinstance(name, str) or not name.isidentifier() for name in widths):
        raise ValueError(
            "width keys must be non-empty Python-style parameter names"
        )

    normalized: dict[str, float] = {}
    for name in sorted(widths):
        if name not in parameters:
            raise ValueError(f"width refers to unknown model parameter {name!r}")
        value = parameters[name]
        if (
            isinstance(value, bool)
            or isinstance(value, Integral)
            or not isinstance(value, Real)
        ):
            raise TypeError(
                f"selected model parameter {name!r} must be a continuous float"
            )
        normalized[name] = _nonnegative_finite_real(
            widths[name],
            name=f"widths[{name!r}]",
        )
    return normalized


def _apply_parameter_offsets(
    source: ModelParameterSet,
    *,
    widths: Mapping[str, DisorderParameterValue],
    rng: np.random.Generator,
) -> ModelParameterSet:
    result = dict(source)
    for name in sorted(widths):
        width = widths[name]
        if isinstance(width, bool) or not isinstance(width, Real):
            raise TypeError("recorded parameter widths must contain real numbers")
        source_value = source[name]
        if (
            isinstance(source_value, bool)
            or isinstance(source_value, Integral)
            or not isinstance(source_value, Real)
        ):
            raise TypeError(
                f"selected model parameter {name!r} must be a continuous float"
            )
        offset = float(
            rng.uniform(
                low=-0.5 * float(width),
                high=0.5 * float(width),
            )
        )
        with np.errstate(over="ignore", invalid="ignore"):
            perturbed = float(source_value) + offset
        if not np.isfinite(perturbed):
            raise ValueError(
                f"perturbation produced a non-finite value for parameter {name!r}"
            )
        result[name] = perturbed
    return result


def _nonnegative_finite_real(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result
