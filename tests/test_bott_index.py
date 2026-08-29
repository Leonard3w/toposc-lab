import numpy as np
import pytest

from toposc_lab.models.qwz_model import QWZModel, QWZModelParameters
from toposc_lab.observables.results import StandardizedObservable
from toposc_lab.topology import BottIndexResult, bott_index
from toposc_lab.topology.symmetry import SymmetryClassification


def _class_a() -> SymmetryClassification:
    return SymmetryClassification.from_signature(
        time_reversal_square=None,
        particle_hole_square=None,
        chiral_symmetry=False,
    )


def _qwz_bott(mass: float, *, swap_coordinates: bool = False) -> BottIndexResult:
    size = 6
    model = QWZModel(
        QWZModelParameters(
            n_x=size,
            n_y=size,
            mass=mass,
            boundary_x="open",
            boundary_y="open",
        )
    )
    coordinates = np.repeat(model.lattice.coordinates.astype(float), 2, axis=0)
    if swap_coordinates:
        coordinates = coordinates[:, ::-1]
    return bott_index(
        model.hamiltonian(),
        coordinates,
        np.array([size, size], dtype=float),
        _class_a(),
    )


@pytest.mark.parametrize(("mass", "expected"), [(-1.0, -1), (1.0, 1)])
def test_open_qwz_topological_phases_have_oriented_bott_index(
    mass: float,
    expected: int,
) -> None:
    result = _qwz_bott(mass)

    assert result.is_quantized
    assert result.bott_index == expected
    assert result.bott_estimate == pytest.approx(float(expected))
    assert result.occupied_state_count == 36
    assert result.unoccupied_state_count == 36
    assert result.minimum_fermi_distance > 0.1
    assert result.minimum_projected_position_singular_value > 0.2


def test_open_qwz_trivial_phase_has_zero_bott_index() -> None:
    result = _qwz_bott(3.0)

    assert result.bott_index == 0
    assert result.bott_estimate == pytest.approx(0.0, abs=1.0e-12)


def test_swapping_coordinate_orientation_reverses_bott_sign() -> None:
    result = _qwz_bott(1.0, swap_coordinates=True)

    assert result.bott_index == -1
    assert result.bott_estimate == pytest.approx(-1.0)


def test_bott_result_has_standardized_output() -> None:
    result = _qwz_bott(1.0)

    assert isinstance(result, StandardizedObservable)
    record = result.to_observable_record()
    assert record.kind == "bott_index"
    assert record.scalars["bott_index"] == 1
    assert record.metadata["requires_explicit_coordinate_periods"] is True
    assert record.metadata["requires_mobility_or_spectral_gap_assumption"] is True


def test_bott_rejects_unsupported_symmetry_class() -> None:
    class_aii = SymmetryClassification.from_signature(
        time_reversal_square=-1,
        particle_hole_square=None,
        chiral_symmetry=False,
    )

    with pytest.raises(ValueError, match="class A, C, or D"):
        bott_index(
            np.diag([-1.0, 1.0, -2.0, 2.0]),
            np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]),
            np.array([2.0, 2.0]),
            class_aii,
        )


def test_bott_rejects_state_at_fermi_energy() -> None:
    with pytest.raises(ValueError, match="Fermi-level"):
        bott_index(
            np.diag([-1.0, 0.0, 1.0, 2.0]),
            np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]),
            np.array([2.0, 2.0]),
            _class_a(),
        )


def test_bott_rejects_singular_projected_position_operator() -> None:
    occupied = np.full(4, 0.5, dtype=complex)
    hamiltonian = np.eye(4) - 2.0 * np.outer(occupied, occupied.conj())

    with pytest.raises(ValueError, match="numerically singular"):
        bott_index(
            hamiltonian,
            np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]),
            np.array([2.0, 2.0]),
            _class_a(),
        )


def test_bott_validates_hermiticity_coordinates_and_periods() -> None:
    hamiltonian = np.diag([-2.0, -1.0, 1.0, 2.0]).astype(complex)
    coordinates = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    )

    broken = hamiltonian.copy()
    broken[0, 1] = 1.0
    with pytest.raises(ValueError, match="Hermitian"):
        bott_index(broken, coordinates, np.array([2.0, 2.0]), _class_a())
    with pytest.raises(ValueError, match=r"shape \(dimension, 2\)"):
        bott_index(hamiltonian, coordinates[:, 0], np.array([2.0, 2.0]), _class_a())
    with pytest.raises(ValueError, match="finite and positive"):
        bott_index(hamiltonian, coordinates, np.array([2.0, 0.0]), _class_a())
