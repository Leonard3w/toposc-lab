import numpy as np

from toposc_lab.scans.haldane_scan import scan_haldane_mass


def test_haldane_mass_scan_returns_standardized_results() -> None:
    result = scan_haldane_mass(
        mass_values=np.array([-1.5, 0.0, 1.5]),
        n_k=21,
    )

    assert result.mass_values.shape == (3,)
    assert result.bulk_gaps.shape == (3,)
    assert result.chern_numbers.shape == (3,)
    assert len(result.scan.results) == 3


def test_haldane_mass_scan_finds_topological_and_trivial_regions() -> None:
    result = scan_haldane_mass(
        mass_values=np.array([-1.5, 0.0, 1.5]),
        n_k=21,
    )

    # Außerhalb des kritischen Bereichs ist die Phase trivial.
    assert result.chern_numbers[0] == 0
    assert result.chern_numbers[2] == 0

    # Bei M = 0 und phi = pi/2 liegt die Haldane-Chern-Phase vor.
    assert abs(result.chern_numbers[1]) == 1


def test_haldane_gap_becomes_small_near_the_analytic_transition() -> None:
    next_nearest_hopping = 0.2
    critical_mass = 3.0 * np.sqrt(3.0) * next_nearest_hopping

    result = scan_haldane_mass(
        mass_values=np.array(
            [critical_mass - 0.03, critical_mass + 0.03]
        ),
        next_nearest_neighbor_hopping=next_nearest_hopping,
        n_k=21,
    )

    assert np.min(result.bulk_gaps) < 0.1
