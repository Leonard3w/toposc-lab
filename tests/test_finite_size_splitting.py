import numpy as np
import pytest

from toposc_lab.models.geometry_kitaev_chain import GeometryKitaevChain
from toposc_lab.models.kitaev_chain import KitaevChainParameters
from toposc_lab.observables.majorana import finite_size_splitting_diagnostics
from toposc_lab.solvers.exact_diagonalization import ExactDiagonalizationSolver


def test_numerical_zero_modes_are_not_classified_as_split_pair() -> None:
    diagnostics = finite_size_splitting_diagnostics(
        np.array([-2.0, 0.0, 1.0e-12, 2.0]),
        zero_tolerance=1.0e-10,
    )

    assert diagnostics.classification == "numerical_zero_modes"
    assert diagnostics.zero_mode_indices == (1, 2)
    assert diagnostics.quasiparticle_energy == 0.0
    assert diagnostics.next_excitation_energy == pytest.approx(2.0)
    assert not diagnostics.is_split_pair_candidate


def test_symmetric_near_zero_levels_are_split_pair_candidate() -> None:
    diagnostics = finite_size_splitting_diagnostics(
        np.array([2.0, -2.0e-5, -2.0, 2.0e-5]),
        splitting_tolerance=1.0e-3,
    )

    assert diagnostics.classification == "split_pair_candidate"
    assert diagnostics.negative_index == 1
    assert diagnostics.positive_index == 3
    assert diagnostics.quasiparticle_energy == pytest.approx(2.0e-5)
    assert diagnostics.pair_level_separation == pytest.approx(4.0e-5)
    assert diagnostics.pair_center_offset == pytest.approx(0.0)
    assert diagnostics.particle_hole_mismatch == pytest.approx(0.0)
    assert diagnostics.next_excitation_energy == pytest.approx(2.0)
    assert diagnostics.isolation_gap == pytest.approx(2.0 - 2.0e-5)
    assert diagnostics.isolation_ratio == pytest.approx(1.0e5)


def test_asymmetric_near_zero_levels_are_not_particle_hole_pair() -> None:
    diagnostics = finite_size_splitting_diagnostics(
        np.array([-2.0e-5, 3.0e-5]),
        splitting_tolerance=1.0e-3,
        particle_hole_tolerance=1.0e-6,
    )

    assert diagnostics.classification == "no_near_zero_structure"
    assert diagnostics.particle_hole_mismatch == pytest.approx(1.0e-5)
    assert not diagnostics.is_particle_hole_pair


def test_symmetric_pair_outside_splitting_window_is_not_candidate() -> None:
    diagnostics = finite_size_splitting_diagnostics(
        np.array([-0.2, 0.2]),
        splitting_tolerance=0.1,
    )

    assert diagnostics.is_particle_hole_pair
    assert not diagnostics.is_split_pair_candidate
    assert diagnostics.classification == "no_near_zero_structure"


def test_one_sided_spectrum_has_no_pair_diagnostics() -> None:
    diagnostics = finite_size_splitting_diagnostics(np.array([0.1, 0.2]))

    assert diagnostics.classification == "no_near_zero_structure"
    assert diagnostics.negative_energy is None
    assert diagnostics.positive_energy is None


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("zero_tolerance", -1.0),
        ("splitting_tolerance", np.inf),
        ("particle_hole_tolerance", np.nan),
    ],
)
def test_splitting_diagnostics_rejects_invalid_tolerances(
    name: str,
    value: float,
) -> None:
    tolerances = {
        "zero_tolerance": 1.0e-10,
        "splitting_tolerance": 1.0e-3,
        "particle_hole_tolerance": 1.0e-8,
    }
    tolerances[name] = value

    with pytest.raises(ValueError, match=name):
        finite_size_splitting_diagnostics(
            np.array([-1.0, 1.0]),
            **tolerances,
        )


def test_open_kitaev_chain_exhibits_isolated_finite_size_split_pair() -> None:
    model = GeometryKitaevChain(
        KitaevChainParameters(
            n_sites=8,
            hopping=1.0,
            chemical_potential=0.5,
            pairing=0.8,
            boundary="open",
        )
    )
    result = ExactDiagonalizationSolver().solve_model(model)

    diagnostics = finite_size_splitting_diagnostics(
        result.eigenvalues,
        splitting_tolerance=1.0e-3,
    )

    assert diagnostics.classification == "split_pair_candidate"
    assert diagnostics.quasiparticle_energy == pytest.approx(2.089026e-4, rel=1.0e-5)
    assert diagnostics.isolation_ratio is not None
    assert diagnostics.isolation_ratio > 1_000.0
