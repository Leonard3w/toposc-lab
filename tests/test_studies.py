from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from toposc_lab.core.results import (
    BasisLayout,
    ParameterScanResult,
    SimulationResult,
)
from toposc_lab.data.studies import (
    StudyData,
    StudyMetadata,
    load_study,
    save_study,
    study_from_bytes,
    study_from_parameter_scan,
    study_to_bytes,
)


def _metadata() -> StudyMetadata:
    return StudyMetadata(
        study_name="haldane mass scan",
        model_name="HaldaneModel",
        model_parameters={"t1": 1.0, "t2": 0.2, "phase": np.pi / 2.0},
        random_seed=17,
        created_at=datetime(2026, 7, 13, tzinfo=timezone.utc),
        package_version="test-version",
        tags=("haldane", "phase-diagram"),
    )


def test_study_round_trip_preserves_metadata_and_arrays(tmp_path) -> None:
    study = StudyData(
        metadata=_metadata(),
        arrays={
            "mass_values": np.array([-1.0, 0.0, 1.0]),
            "bulk_gaps": np.array([0.5, 0.0, 0.5]),
            "chern_numbers": np.array([0, -1, 0]),
        },
    )

    path = save_study(tmp_path / "haldane_phase_diagram", study)
    loaded = load_study(path)

    assert path.name == "haldane_phase_diagram.npz"
    assert loaded.metadata == study.metadata
    assert set(loaded.arrays) == set(study.arrays)

    for name, values in study.arrays.items():
        assert np.array_equal(loaded.arrays[name], values)


def test_study_rejects_unsafe_object_arrays() -> None:
    with pytest.raises(ValueError, match="object dtype"):
        StudyData(
            metadata=_metadata(),
            arrays={"not_safe": np.array([{"value": 1}], dtype=object)},
        )


def test_study_bytes_round_trip_preserves_arrays() -> None:
    study = StudyData(
        metadata=_metadata(),
        arrays={"bulk_gaps": np.array([0.5, 0.0, 0.5])},
    )

    loaded = study_from_bytes(study_to_bytes(study))

    assert loaded.metadata == study.metadata
    assert np.array_equal(loaded.arrays["bulk_gaps"], study.arrays["bulk_gaps"])


def test_study_from_parameter_scan_stores_standard_and_extra_observables() -> None:
    layout = BasisLayout(spatial_shape=(1,))
    first_result = SimulationResult(
        model_name="ToyModel",
        eigenvalues=np.array([-1.0]),
        eigenvectors=np.array([[1.0]]),
        basis_layout=layout,
    )
    second_result = SimulationResult(
        model_name="ToyModel",
        eigenvalues=np.array([0.25]),
        eigenvectors=np.array([[1.0]]),
        basis_layout=layout,
    )
    scan = ParameterScanResult(
        parameter_name="mass",
        parameter_values=np.array([-1.0, 1.0]),
        results=(first_result, second_result),
        metadata={"solver": "exact"},
    )

    study = study_from_parameter_scan(
        scan,
        _metadata(),
        observables={"bulk_gaps": np.array([1.0, 0.25])},
    )

    assert np.array_equal(study.arrays["parameter_values"], [-1.0, 1.0])
    assert np.array_equal(study.arrays["spectra"], [[-1.0], [0.25]])
    assert np.array_equal(study.arrays["minimum_abs_energy"], [1.0, 0.25])
    assert np.array_equal(study.arrays["bulk_gaps"], [1.0, 0.25])
    assert study.metadata.scan_parameters == {
        "parameter_name": "mass",
        "solver": "exact",
    }
