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


def test_count_zero_modes_counts_energies_below_tolerance() -> None:
    eigenvalues = np.array([-2.0, -1e-12, 0.0, 1e-12, 2.0])

    assert count_zero_modes(eigenvalues, tolerance=1e-10) == 3


def test_edge_gap_is_lowest_abs_energy() -> None:
    eigenvalues = np.array([-2.0, -1e-4, 1e-4, 2.0])

    assert edge_gap(eigenvalues) == pytest.approx(1e-4)


def test_bulk_gap_ignores_zero_modes() -> None:
    eigenvalues = np.array([-2.0, -1e-12, 0.0, 1e-12, 0.5, 2.0])

    assert bulk_gap(eigenvalues, tolerance=1e-10) == pytest.approx(0.5)