from collections.abc import Iterable

import numpy as np
import pytest

from toposc_lab.data.studies import StudyData, StudyMetadata, study_from_bytes, study_to_bytes
from toposc_lab.observables.berry import BerryCurvatureResult
from toposc_lab.observables.ldos import LocalDensityOfStates
from toposc_lab.observables.localization import LocalizationProfile, SiteProbabilityDensity
from toposc_lab.observables.majorana import (
    FiniteSizeSplittingDiagnostics,
    MajoranaDiagnostics,
)
from toposc_lab.observables.results import (
    ObservableRecord,
    StandardizedObservable,
    stack_observable_records,
)
from toposc_lab.observables.symmetries import SymmetryCheckResult


def _specialized_results() -> Iterable[StandardizedObservable]:
    yield BerryCurvatureResult(
        k_x=np.array([0.0]),
        k_y=np.array([0.0]),
        berry_flux=np.array([[0.0]]),
        berry_curvature=np.array([[0.0]]),
        chern_number=0.0,
    )
    yield LocalDensityOfStates(
        energy_values=np.array([0.0]),
        values=np.array([[1.0]]),
        component_values=np.array([[[1.0]]]),
        component_labels=("orbital",),
    )
    yield SiteProbabilityDensity(
        probability=np.array([1.0]),
        component_probabilities=np.array([[1.0]]),
        component_labels=("orbital",),
    )
    yield LocalizationProfile(
        probability=np.array([1.0]),
        component_probabilities=np.array([[1.0]]),
        center_of_mass=np.array([0.0]),
        inverse_participation_ratio=1.0,
        participation_ratio=1.0,
        edge_weight=1.0,
        bulk_weight=0.0,
        is_edge_localized=True,
        component_labels=("orbital",),
    )
    yield MajoranaDiagnostics(
        site_probability=np.array([1.0]),
        particle_probability=np.array([0.5]),
        hole_probability=np.array([0.5]),
        polarization=np.array([1.0 + 0.0j]),
        polarization_magnitude=np.array([1.0]),
        total_polarization=1.0 + 0.0j,
        self_conjugacy=1.0,
        polarization_norm=1.0,
        particle_weight=0.5,
        hole_weight=0.5,
    )
    yield FiniteSizeSplittingDiagnostics(
        classification="split_pair_candidate",
        zero_mode_indices=(),
        negative_index=0,
        positive_index=1,
        negative_energy=-1.0e-5,
        positive_energy=1.0e-5,
        quasiparticle_energy=1.0e-5,
        pair_level_separation=2.0e-5,
        pair_center_offset=0.0,
        particle_hole_mismatch=0.0,
        is_particle_hole_pair=True,
        is_split_pair_candidate=True,
        next_excitation_energy=None,
        isolation_gap=None,
        isolation_ratio=None,
    )
    yield SymmetryCheckResult(
        name="Hermitian",
        satisfied=True,
        residual=0.0,
        tolerance=1.0e-10,
    )


def test_all_observable_dataclasses_share_standard_record_interface() -> None:
    records = []
    for result in _specialized_results():
        assert isinstance(result, StandardizedObservable)
        record = result.to_observable_record()
        assert isinstance(record, ObservableRecord)
        assert record.kind.isidentifier()
        records.append(record)

    assert {record.kind for record in records} == {
        "berry_curvature",
        "finite_size_splitting",
        "local_density_of_states",
        "localization_profile",
        "majorana_diagnostics",
        "site_probability_density",
        "symmetry_check",
    }


def test_observable_record_defensively_copies_arrays_and_metadata() -> None:
    values = np.array([1.0, 2.0])
    metadata = {"labels": ["left", "right"]}

    record = ObservableRecord(
        kind="test_observable",
        arrays={"values": values},
        metadata=metadata,
    )
    values[0] = 99.0
    metadata["labels"].append("changed")

    assert np.array_equal(record.arrays["values"], [1.0, 2.0])
    assert not record.arrays["values"].flags.writeable
    assert record.metadata["labels"] == ["left", "right"]


def test_numerical_arrays_are_npz_safe_and_encode_missing_scalars() -> None:
    record = ObservableRecord(
        kind="splitting",
        scalars={"energy": None, "detected": False},
        arrays={"indices": np.array([], dtype=np.int64)},
    )

    payload = record.numerical_arrays()

    assert set(payload) == {
        "splitting_energy",
        "splitting_detected",
        "splitting_indices",
    }
    assert np.isnan(payload["splitting_energy"])
    assert not any(values.dtype.hasobject for values in payload.values())


def test_scalar_features_have_deterministic_order() -> None:
    record = ObservableRecord(
        kind="test",
        scalars={"z_value": 2.0, "a_flag": True, "missing": None},
    )

    names, values = record.scalar_features()

    assert names == ("a_flag", "missing", "z_value")
    assert values[0] == 1.0
    assert np.isnan(values[1])
    assert values[2] == 2.0


def test_compatible_observable_records_stack_on_sample_axis() -> None:
    first = ObservableRecord(
        kind="localization",
        scalars={"ipr": 1.0, "missing": None},
        arrays={"probability": np.array([1.0, 0.0])},
        metadata={"component_labels": ["orbital"]},
    )
    second = ObservableRecord(
        kind="localization",
        scalars={"ipr": 0.5, "missing": 2.0},
        arrays={"probability": np.array([0.5, 0.5])},
        metadata={"component_labels": ["orbital"]},
    )

    payload = stack_observable_records([first, second])

    assert np.allclose(payload["localization_ipr"], [1.0, 0.5])
    assert np.isnan(payload["localization_missing"][0])
    assert np.array_equal(
        payload["localization_probability"],
        [[1.0, 0.0], [0.5, 0.5]],
    )


def test_stacking_rejects_incompatible_record_schema() -> None:
    first = ObservableRecord(kind="first", scalars={"value": 1.0})
    second = ObservableRecord(kind="second", scalars={"value": 2.0})

    with pytest.raises(ValueError, match="same kind"):
        stack_observable_records([first, second])


def test_observable_payload_round_trips_through_study_storage() -> None:
    diagnostics = next(
        result
        for result in _specialized_results()
        if isinstance(result, MajoranaDiagnostics)
    )
    record = diagnostics.to_observable_record()
    study = StudyData(
        metadata=StudyMetadata(study_name="observable", model_name="test"),
        arrays=record.numerical_arrays(),
    )

    loaded = study_from_bytes(study_to_bytes(study))

    assert set(loaded.arrays) == set(study.arrays)
    assert loaded.arrays["majorana_diagnostics_self_conjugacy"] == pytest.approx(1.0)
    assert np.array_equal(
        loaded.arrays["majorana_diagnostics_polarization"],
        np.array([1.0 + 0.0j]),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "not valid"},
        {"kind": "valid", "arrays": {"labels": np.array(["A", "B"])}},
        {"kind": "valid", "arrays": {"unsafe": np.array([object()])}},
        {"kind": "valid", "scalars": {"value": np.inf}},
        {"kind": "valid", "metadata": {"value": object()}},
    ],
)
def test_observable_record_rejects_non_dataset_safe_values(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        ObservableRecord(**kwargs)  # type: ignore[arg-type]
