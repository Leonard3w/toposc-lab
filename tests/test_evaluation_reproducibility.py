from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from toposc_lab.core.model import BaseModel
from toposc_lab.core.results import BasisLayout
from toposc_lab.evaluation import (
    GEOMETRY_ID_SCHEME,
    GeometryEvaluationConfig,
    GeometryModelAdapter,
    create_reproducibility_record,
    evaluate_geometry,
    exact_geometry_id,
)
from toposc_lab.geometry import Geometry, GeometryEdge, chain


class _ParameterModel(BaseModel):
    def __init__(self, geometry: Geometry) -> None:
        self.geometry = geometry

    @property
    def parameters(self) -> dict[str, object]:
        return {
            "hopping": 1.25,
            "boundary": {"axes": [0], "periodic": False},
        }

    @property
    def basis_layout(self) -> BasisLayout:
        return BasisLayout(spatial_shape=(self.geometry.n_sites,))

    def hamiltonian(self) -> np.ndarray:
        return np.diag([-1.0, 1.0])


def _adapter() -> GeometryModelAdapter:
    return GeometryModelAdapter(model_factory=_ParameterModel)


def test_pipeline_records_complete_reproducibility_inputs() -> None:
    run = evaluate_geometry(
        chain(2),
        adapter=_adapter(),
        config=GeometryEvaluationConfig(low_energy_count=2),
        seed=1729,
        code_version="commit:abc123",
    )

    assert run.is_valid
    assert run.reproducibility is not None
    record = run.reproducibility
    assert record.seed == 1729
    assert record.model_name == "_ParameterModel"
    assert record.model_parameters == {
        "boundary": {"axes": (0,), "periodic": False},
        "hopping": 1.25,
    }
    assert record.geometry_id_scheme == GEOMETRY_ID_SCHEME
    assert record.geometry_id == exact_geometry_id(chain(2))
    assert record.solver_name.endswith(".ExactDiagonalizationSolver")
    assert record.solver_settings["backend"] == "numpy.linalg.eigh"
    assert record.evaluation_settings["low_energy_count"] == 2
    assert record.code_version == "commit:abc123"
    assert record.code_version_source == "explicit"
    assert any("not a canonical isomorphism" in item for item in record.warnings)


def test_exact_geometry_id_is_repeatable_and_orientation_sensitive() -> None:
    forward = Geometry(n_sites=2, edges=(GeometryEdge(0, 1),))
    reverse = Geometry(n_sites=2, edges=(GeometryEdge(1, 0),))

    first = exact_geometry_id(forward)

    assert first == exact_geometry_id(forward)
    assert first != exact_geometry_id(reverse)


def test_record_deeply_freezes_parameter_and_setting_mappings() -> None:
    parameters: dict[str, object] = {"nested": {"values": [1, 2]}}
    record = create_reproducibility_record(
        chain(2),
        seed=None,
        model_name="ExampleModel",
        model_parameters=parameters,
        solver_name="ExampleSolver",
        solver_settings={},
        evaluation_settings={},
        code_version="v1",
    )
    parameters["nested"] = "changed"

    assert record.model_parameters["nested"] == {"values": (1, 2)}
    with pytest.raises(TypeError):
        record.model_parameters["new"] = 1  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        record.seed = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        ({"seed": -1}, ValueError, "non-negative"),
        ({"seed": True}, TypeError, "integer or None"),
        ({"code_version": ""}, ValueError, "non-empty"),
    ],
)
def test_pipeline_rejects_invalid_reproducibility_inputs_before_execution(
    kwargs: dict[str, object],
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        evaluate_geometry(
            chain(2),
            adapter=_adapter(),
            **kwargs,  # type: ignore[arg-type]
        )
