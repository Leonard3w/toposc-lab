import numpy as np
import pytest

from toposc_lab.models.qwz_model import QWZModel, QWZModelParameters
from toposc_lab.observables.results import StandardizedObservable
from toposc_lab.topology import SpectralLocalizerResult, spectral_localizer
from toposc_lab.topology.symmetry import SymmetryClassification


def _class_a() -> SymmetryClassification:
    return SymmetryClassification.from_signature(
        time_reversal_square=None,
        particle_hole_square=None,
        chiral_symmetry=False,
    )


def _qwz_localizer(
    mass: float,
    *,
    probe_position: tuple[float, float] = (2.5, 2.5),
    coordinate_scale: float = 1.0,
    swap_coordinates: bool = False,
) -> SpectralLocalizerResult:
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
    coordinates = coordinate_scale * np.repeat(
        model.lattice.coordinates.astype(float),
        2,
        axis=0,
    )
    probe = coordinate_scale * np.asarray(probe_position)
    if swap_coordinates:
        coordinates = coordinates[:, ::-1]
        probe = probe[::-1]
    return spectral_localizer(
        model.hamiltonian(),
        coordinates,
        probe,
        _class_a(),
        kappa=0.2 / coordinate_scale,
    )


@pytest.mark.parametrize(("mass", "expected"), [(-1.0, -1), (1.0, 1)])
def test_qwz_center_localizer_matches_oriented_chern_phase(
    mass: float,
    expected: int,
) -> None:
    result = _qwz_localizer(mass)

    assert result.is_invertible
    assert result.signature == 2 * expected
    assert result.local_chern_number == expected
    assert result.localizer_gap > 0.4
    assert result.zero_eigenvalue_count == 0


@pytest.mark.parametrize("mass", [-3.0, 3.0])
def test_qwz_trivial_phase_has_zero_localizer_index(mass: float) -> None:
    result = _qwz_localizer(mass)

    assert result.local_chern_number == 0
    assert result.signature == 0
    assert result.localizer_gap > 1.0


def test_probe_outside_sample_has_trivial_localizer_index() -> None:
    result = _qwz_localizer(1.0, probe_position=(20.0, 20.0))

    assert result.local_chern_number == 0
    assert result.localizer_gap > 4.0


def test_swapping_coordinate_orientation_reverses_localizer_index() -> None:
    result = _qwz_localizer(1.0, swap_coordinates=True)

    assert result.local_chern_number == -1
    assert result.signature == -2


def test_consistent_coordinate_and_kappa_rescaling_is_invariant() -> None:
    reference = _qwz_localizer(1.0)
    rescaled = _qwz_localizer(1.0, coordinate_scale=2.5)

    assert np.allclose(
        rescaled.localizer_eigenvalues,
        reference.localizer_eigenvalues,
    )
    assert rescaled.local_chern_number == reference.local_chern_number
    assert rescaled.localizer_gap == pytest.approx(reference.localizer_gap)


def test_zero_localizer_gap_leaves_index_undefined() -> None:
    hamiltonian = np.diag([-1.0, 1.0, 2.0, 3.0])
    coordinates = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    )

    result = spectral_localizer(
        hamiltonian,
        coordinates,
        np.array([0.0, 0.0]),
        _class_a(),
        energy=-1.0,
    )

    assert not result.is_invertible
    assert result.local_chern_number is None
    assert result.localizer_gap == pytest.approx(0.0)
    assert result.zero_eigenvalue_count == 2


def test_spectral_localizer_result_has_standardized_output() -> None:
    result = _qwz_localizer(1.0)

    assert isinstance(result, StandardizedObservable)
    record = result.to_observable_record()
    assert record.kind == "spectral_localizer_2d"
    assert record.scalars["local_chern_number"] == 1
    assert record.metadata["index_convention"] == "half_signature"


def test_spectral_localizer_rejects_unsupported_symmetry_class() -> None:
    class_aii = SymmetryClassification.from_signature(
        time_reversal_square=-1,
        particle_hole_square=None,
        chiral_symmetry=False,
    )

    with pytest.raises(ValueError, match="class A, C, or D"):
        spectral_localizer(
            np.diag([-2.0, -1.0, 1.0, 2.0]),
            np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]),
            np.array([0.5, 0.5]),
            class_aii,
        )


def test_spectral_localizer_validates_inputs() -> None:
    hamiltonian = np.diag([-2.0, -1.0, 1.0, 2.0]).astype(complex)
    coordinates = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    )

    broken = hamiltonian.copy()
    broken[0, 1] = 1.0
    with pytest.raises(ValueError, match="Hermitian"):
        spectral_localizer(broken, coordinates, (0.5, 0.5), _class_a())
    with pytest.raises(ValueError, match=r"shape \(dimension, 2\)"):
        spectral_localizer(
            hamiltonian,
            coordinates[:, 0],
            (0.5, 0.5),
            _class_a(),
        )
    with pytest.raises(ValueError, match="two finite coordinates"):
        spectral_localizer(
            hamiltonian,
            coordinates,
            np.array([0.5]),
            _class_a(),
        )
    with pytest.raises(ValueError, match="positive"):
        spectral_localizer(
            hamiltonian,
            coordinates,
            (0.5, 0.5),
            _class_a(),
            kappa=0.0,
        )
