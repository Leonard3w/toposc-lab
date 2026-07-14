from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.app.study_workspace import (
    common_scalar_observables,
    scan_parameter_name,
    study_summary,
)
from toposc_lab.data.studies import StudyData, StudyMetadata


def _study(*, parameter_name: str = "mass", extra_array: bool = False) -> StudyData:
    arrays: dict[str, np.ndarray] = {
        "parameter_values": np.array([-1.0, 0.0, 1.0]),
        "spectra": np.array([[-1.0, 1.0], [-0.1, 0.1], [-1.0, 1.0]]),
        "bulk_gaps": np.array([1.0, 0.1, 1.0]),
    }
    if extra_array:
        arrays["edge_weights"] = np.array([0.8, 0.2, 0.8])

    return StudyData(
        metadata=StudyMetadata(
            study_name="test study",
            model_name="TestModel",
            scan_parameters={"parameter_name": parameter_name},
            package_version="test",
        ),
        arrays=arrays,
    )


def test_study_explorer_only_offers_shared_scalar_observables() -> None:
    studies = {"first": _study(), "second": _study(extra_array=True)}

    assert scan_parameter_name(studies) == "mass"
    assert common_scalar_observables(studies) == ("bulk_gaps",)


def test_study_explorer_rejects_different_scan_axes() -> None:
    studies = {"mass": _study(), "field": _study(parameter_name="field")}

    with pytest.raises(ValueError, match="different scan parameters"):
        scan_parameter_name(studies)


def test_study_summary_reports_reproducibility_context() -> None:
    summary = study_summary(_study())

    assert summary["model"] == "TestModel"
    assert summary["scan parameter"] == "mass"
    assert summary["scan points"] == 3
