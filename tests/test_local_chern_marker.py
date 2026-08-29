import numpy as np
import pytest

from toposc_lab.models.qwz_model import QWZModel, QWZModelParameters
from toposc_lab.observables.results import StandardizedObservable
from toposc_lab.topology import LocalChernMarkerResult, local_chern_marker
from toposc_lab.topology.symmetry import SymmetryClassification


def _class_a() -> SymmetryClassification:
    return SymmetryClassification.from_signature(
        time_reversal_square=None,
        particle_hole_square=None,
        chiral_symmetry=False,
    )


def _qwz_marker(
    mass: float,
    *,
    coordinate_scale: float = 1.0,
    swap_coordinates: bool = False,
) -> LocalChernMarkerResult:
    size = 10
    model = QWZModel(
        QWZModelParameters(
            n_x=size,
            n_y=size,
            mass=mass,
            boundary_x="open",
            boundary_y="open",
        )
    )
    site_coordinates = coordinate_scale * model.lattice.coordinates.astype(float)
    basis_coordinates = np.repeat(site_coordinates, 2, axis=0)
    if swap_coordinates:
        basis_coordinates = basis_coordinates[:, ::-1]
    bulk_mask = np.zeros((size, size), dtype=bool)
    bulk_mask[3:7, 3:7] = True
    return local_chern_marker(
        model.hamiltonian(),
        basis_coordinates,
        np.full(size * size, coordinate_scale**2),
        bulk_mask.ravel(),
        _class_a(),
    )


@pytest.mark.parametrize(("mass", "expected"), [(-1.0, -1), (1.0, 1)])
def test_open_qwz_bulk_marker_matches_oriented_chern_phase(
    mass: float,
    expected: int,
) -> None:
    result = _qwz_marker(mass)

    assert result.is_quantized
    assert result.chern_number == expected
    assert result.bulk_chern_estimate == pytest.approx(expected, abs=2.0e-3)
    assert result.occupied_state_count == 100
    assert result.unoccupied_state_count == 100
    assert result.minimum_fermi_distance > 0.07
    assert result.finite_sample_trace_residual < 1.0e-10


def test_open_qwz_trivial_phase_has_zero_bulk_marker() -> None:
    result = _qwz_marker(3.0)

    assert result.chern_number == 0
    assert result.bulk_chern_estimate == pytest.approx(0.0, abs=2.0e-3)


def test_marker_is_invariant_under_consistent_coordinate_rescaling() -> None:
    reference = _qwz_marker(1.0)
    rescaled = _qwz_marker(1.0, coordinate_scale=2.5)

    assert rescaled.bulk_chern_estimate == pytest.approx(
        reference.bulk_chern_estimate
    )
    assert np.allclose(rescaled.local_marker, reference.local_marker)


def test_swapping_coordinate_orientation_reverses_marker_sign() -> None:
    result = _qwz_marker(1.0, swap_coordinates=True)

    assert result.chern_number == -1
    assert result.bulk_chern_estimate == pytest.approx(-1.0, abs=2.0e-3)


def test_local_chern_result_has_standardized_output() -> None:
    result = _qwz_marker(1.0)

    assert isinstance(result, StandardizedObservable)
    record = result.to_observable_record()
    assert record.kind == "local_chern_marker"
    assert record.scalars["chern_number"] == 1
    assert record.metadata["requires_explicit_position_areas"] is True
    assert record.metadata["requires_explicit_bulk_mask"] is True


def test_local_chern_rejects_unsupported_symmetry_class() -> None:
    class_aii = SymmetryClassification.from_signature(
        time_reversal_square=-1,
        particle_hole_square=None,
        chiral_symmetry=False,
    )

    with pytest.raises(ValueError, match="class A, C, or D"):
        local_chern_marker(
            np.diag([-2.0, -1.0, 1.0, 2.0]),
            np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]),
            1.0,
            np.ones(4, dtype=bool),
            class_aii,
        )


def test_local_chern_rejects_state_at_fermi_energy() -> None:
    with pytest.raises(ValueError, match="Fermi-level"):
        local_chern_marker(
            np.diag([-1.0, 0.0, 1.0, 2.0]),
            np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]),
            1.0,
            np.ones(4, dtype=bool),
            _class_a(),
        )


def test_local_chern_validates_position_areas_and_bulk_mask() -> None:
    hamiltonian = np.diag([-2.0, -1.0, 1.0, 2.0])
    coordinates = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    )

    with pytest.raises(ValueError, match="one value per unique position"):
        local_chern_marker(
            hamiltonian,
            coordinates,
            np.ones(3),
            np.ones(4, dtype=bool),
            _class_a(),
        )
    with pytest.raises(ValueError, match="finite and positive"):
        local_chern_marker(
            hamiltonian,
            coordinates,
            np.array([1.0, 1.0, 1.0, 0.0]),
            np.ones(4, dtype=bool),
            _class_a(),
        )
    with pytest.raises(TypeError, match="boolean"):
        local_chern_marker(
            hamiltonian,
            coordinates,
            1.0,
            np.ones(4),
            _class_a(),
        )
