import numpy as np

from toposc_lab.scans.kitaev_scan import KitaevScanResult, scan_kitaev_mu


def test_scan_kitaev_mu_returns_result_object() -> None:
    mu_values = np.array([-1.0, 0.0, 1.0])

    result = scan_kitaev_mu(mu_values=mu_values, L=10)

    assert isinstance(result, KitaevScanResult)


def test_scan_kitaev_mu_has_correct_shapes() -> None:
    mu_values = np.array([-1.0, 0.0, 1.0])
    L = 10

    result = scan_kitaev_mu(mu_values=mu_values, L=L)

    assert result.mu_values.shape == (3,)
    assert result.spectra.shape == (3, 2 * L)
    assert result.gaps.shape == (3,)


def test_scan_kitaev_mu_preserves_mu_values() -> None:
    mu_values = np.linspace(-2.0, 2.0, 5)

    result = scan_kitaev_mu(mu_values=mu_values, L=8)

    assert np.allclose(result.mu_values, mu_values)


def test_scan_kitaev_mu_gaps_are_non_negative() -> None:
    mu_values = np.linspace(-3.0, 3.0, 7)

    result = scan_kitaev_mu(mu_values=mu_values, L=12)

    assert np.all(result.gaps >= 0.0)

    