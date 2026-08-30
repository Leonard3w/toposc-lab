from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.evaluation import GeometryEvaluation, evaluate_spectrum
from toposc_lab.solvers.exact_diagonalization import ExactDiagonalizationSolver


def test_basic_spectral_evaluation_populates_only_spectral_fields() -> None:
    result = evaluate_spectrum(
        np.asarray([-2.0, -0.3, 0.3, 2.0]),
        low_energy_count=2,
    )

    assert isinstance(result, GeometryEvaluation)
    assert result.gap == pytest.approx(0.6)
    assert result.low_energy_states == {1: -0.3, 2: 0.3}
    assert result.zero_mode_count == 0
    assert result.ipr == {}
    assert result.localization == {}
    assert result.majorana_metrics == {}
    assert result.topology == ()
    assert result.geometry_descriptors == {}


def test_matrix_level_solver_output_enters_spectral_evaluation_directly() -> None:
    eigensystem = ExactDiagonalizationSolver().solve(
        np.diag([-2.0, -0.25, 0.25, 2.0])
    )

    result = evaluate_spectrum(eigensystem.eigenvalues, low_energy_count=2)

    assert result.gap == pytest.approx(0.5)
    assert result.low_energy_states == {1: -0.25, 2: 0.25}


def test_gap_is_full_separation_and_not_positive_excitation_energy() -> None:
    result = evaluate_spectrum(
        np.asarray([-2.0, -0.4, 0.4, 2.0]),
        low_energy_count=2,
    )

    assert result.gap == pytest.approx(0.8)
    assert "full finite-spectrum separation" in result.warnings[0]
    assert "not a zero-mode-filtered positive excitation gap" in result.warnings[0]


def test_zero_modes_close_gap_and_are_numerically_counted() -> None:
    result = evaluate_spectrum(
        np.asarray([-1.0, -1.0e-12, 0.0, 1.0e-12, 1.0]),
        zero_mode_tolerance=1.0e-10,
        low_energy_count=2,
    )

    assert result.gap == 0.0
    assert result.zero_mode_count == 3
    assert result.low_energy_states == {2: 0.0, 1: -1.0e-12, 3: 1.0e-12}
    assert "does not establish Majorana character" in result.warnings[1]


def test_low_energy_selection_preserves_all_boundary_ties() -> None:
    result = evaluate_spectrum(
        np.asarray([-3.0, -1.0, 1.0, 1.0, 4.0]),
        low_energy_count=2,
    )

    assert result.low_energy_states == {1: -1.0, 2: 1.0, 3: 1.0}


def test_low_energy_selection_retains_solver_column_indices() -> None:
    result = evaluate_spectrum(
        np.asarray([3.0, -0.2, -4.0, 0.1]),
        low_energy_count=2,
    )

    assert tuple(result.low_energy_states) == (3, 1)
    assert result.low_energy_states == {3: 0.1, 1: -0.2}


def test_nonzero_reference_energy_controls_all_spectral_quantities() -> None:
    result = evaluate_spectrum(
        np.asarray([-1.0, 0.2, 0.5, 0.8, 2.0]),
        reference_energy=0.5,
        zero_mode_tolerance=1.0e-12,
        low_energy_count=1,
    )

    assert result.gap == 0.0
    assert result.zero_mode_count == 1
    assert result.low_energy_states == {2: 0.5}
    assert "reference_energy=0.5" in result.warnings[0]


def test_reference_without_states_on_both_sides_is_rejected() -> None:
    with pytest.raises(ValueError, match="both sides"):
        evaluate_spectrum(np.asarray([0.5, 1.0, 2.0]))


@pytest.mark.parametrize(
    ("eigenvalues", "message"),
    [
        (np.asarray([]), "must not be empty"),
        (np.zeros((2, 2)), "one-dimensional"),
        (np.asarray([-1.0, np.nan, 1.0]), "finite values"),
        (np.asarray([-1.0, np.inf, 1.0]), "finite values"),
    ],
)
def test_invalid_spectra_are_rejected(
    eigenvalues: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_spectrum(eigenvalues)


@pytest.mark.parametrize("low_energy_count", [0, -1])
def test_low_energy_count_must_be_positive(low_energy_count: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        evaluate_spectrum(
            np.asarray([-1.0, 1.0]),
            low_energy_count=low_energy_count,
        )


@pytest.mark.parametrize("low_energy_count", [True, 1.5, "two"])
def test_low_energy_count_must_be_an_integer(low_energy_count: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        evaluate_spectrum(
            np.asarray([-1.0, 1.0]),
            low_energy_count=low_energy_count,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("zero_mode_tolerance", [-1.0, np.inf, np.nan])
def test_zero_mode_tolerance_must_be_nonnegative_and_finite(
    zero_mode_tolerance: float,
) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        evaluate_spectrum(
            np.asarray([-1.0, 1.0]),
            zero_mode_tolerance=zero_mode_tolerance,
        )


@pytest.mark.parametrize("reference_energy", [np.inf, np.nan])
def test_reference_energy_must_be_finite(reference_energy: float) -> None:
    with pytest.raises(ValueError, match="reference_energy"):
        evaluate_spectrum(
            np.asarray([-1.0, 1.0]),
            reference_energy=reference_energy,
        )
