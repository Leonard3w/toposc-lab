from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.observables.spectrum import (
    bulk_gap,
    count_zero_modes,
    edge_gap,
    energy_gap,
    lowest_abs_energy,
    positive_energies,
    spectral_gap,
)


def test_positive_energies_returns_only_positive_values_above_tolerance() -> None:
    eigenvalues = np.array([-2.0, -1e-12, 0.0, 1e-12, 0.5, 2.0])

    result = positive_energies(eigenvalues, tolerance=1e-10)

    assert np.allclose(result, np.array([0.5, 2.0]))


def test_energy_gap_keeps_old_behavior() -> None:
    eigenvalues = np.array([-3.0, -0.2, 0.0, 0.2, 2.0])

    assert energy_gap(eigenvalues, tolerance=1e-10) == pytest.approx(0.2)


def test_energy_gap_returns_zero_if_no_positive_energy_exists() -> None:
    eigenvalues = np.array([-3.0, -0.2, 0.0])

    assert energy_gap(eigenvalues, tolerance=1e-10) == pytest.approx(0.0)


def test_lowest_abs_energy_returns_energy_closest_to_zero() -> None:
    eigenvalues = np.array([-3.0, -0.2, 0.5, 2.0])

    assert lowest_abs_energy(eigenvalues) == pytest.approx(0.2)


def test_lowest_abs_energy_returns_exact_zero_when_present() -> None:
    assert lowest_abs_energy(np.asarray([-1.0, 0.0, 2.0])) == 0.0


@pytest.mark.parametrize(
    ("eigenvalues", "message"),
    [
        (np.asarray([]), "must not be empty"),
        (np.zeros((2, 2)), "one-dimensional"),
        (np.asarray([0.0, np.nan]), "finite values"),
        (np.asarray([0.0, np.inf]), "finite values"),
    ],
)
def test_lowest_abs_energy_rejects_invalid_spectra(
    eigenvalues: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        lowest_abs_energy(eigenvalues)


def test_count_zero_modes_counts_energies_below_tolerance() -> None:
    eigenvalues = np.array([-2.0, -1e-12, 0.0, 1e-12, 2.0])

    assert count_zero_modes(eigenvalues, tolerance=1e-10) == 3


def test_count_zero_modes_includes_tolerance_boundary() -> None:
    eigenvalues = np.array([-1e-3, -1e-4, 0.0, 1e-4, 1e-3])

    assert count_zero_modes(eigenvalues, tolerance=1e-4) == 3


def test_count_zero_modes_uses_configurable_tolerance() -> None:
    eigenvalues = np.array([-1e-5, 0.0, 1e-5])

    assert count_zero_modes(eigenvalues, tolerance=1e-6) == 1
    assert count_zero_modes(eigenvalues, tolerance=1e-4) == 3


@pytest.mark.parametrize(
    ("eigenvalues", "message"),
    [
        (np.asarray([]), "must not be empty"),
        (np.zeros((2, 2)), "one-dimensional"),
        (np.asarray([0.0, np.nan]), "finite values"),
        (np.asarray([0.0, np.inf]), "finite values"),
    ],
)
def test_count_zero_modes_rejects_invalid_spectra(
    eigenvalues: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        count_zero_modes(eigenvalues)


@pytest.mark.parametrize("tolerance", [-1.0, np.inf, np.nan])
def test_count_zero_modes_rejects_invalid_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        count_zero_modes(np.asarray([-1.0, 0.0, 1.0]), tolerance=tolerance)


@pytest.mark.parametrize("tolerance", [True, "loose"])
def test_count_zero_modes_rejects_non_real_tolerance(tolerance: object) -> None:
    with pytest.raises(TypeError, match="tolerance"):
        count_zero_modes(np.asarray([-1.0, 0.0, 1.0]), tolerance=tolerance)  # type: ignore[arg-type]


def test_edge_gap_is_lowest_abs_energy() -> None:
    eigenvalues = np.array([-2.0, -1e-4, 1e-4, 2.0])

    assert edge_gap(eigenvalues) == pytest.approx(1e-4)


def test_bulk_gap_ignores_zero_modes() -> None:
    eigenvalues = np.array([-2.0, -1e-12, 0.0, 1e-12, 0.5, 2.0])

    assert bulk_gap(eigenvalues, tolerance=1e-10) == pytest.approx(0.5)


def test_spectral_gap_measures_full_separation_across_zero() -> None:
    eigenvalues = np.asarray([-3.0, -0.2, 0.5, 2.0])

    assert spectral_gap(eigenvalues) == pytest.approx(0.7)


def test_bdg_spectral_gap_is_twice_positive_excitation_gap() -> None:
    eigenvalues = np.asarray([-2.0, -0.4, 0.4, 2.0])

    assert spectral_gap(eigenvalues) == pytest.approx(0.8)
    assert bulk_gap(eigenvalues) == pytest.approx(0.4)


def test_spectral_gap_closes_for_state_at_reference_within_tolerance() -> None:
    eigenvalues = np.asarray([-2.0, 1.0e-12, 0.5, 2.0])

    assert spectral_gap(eigenvalues, tolerance=1.0e-10) == 0.0


def test_spectral_gap_supports_nonzero_reference_energy() -> None:
    eigenvalues = np.asarray([-1.0, 0.2, 0.8, 2.0])

    assert spectral_gap(eigenvalues, reference_energy=0.5) == pytest.approx(0.6)


def test_spectral_gap_requires_states_on_both_sides_of_reference() -> None:
    with pytest.raises(ValueError, match="both sides"):
        spectral_gap(np.asarray([0.5, 1.0, 2.0]))


@pytest.mark.parametrize("tolerance", [-1.0, np.inf, np.nan])
def test_spectral_gap_rejects_invalid_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        spectral_gap(np.asarray([-1.0, 1.0]), tolerance=tolerance)
