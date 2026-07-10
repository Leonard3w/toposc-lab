import numpy as np

from toposc_lab.scans.qwz_scan import QWZScanResult, scan_qwz_mass


def test_scan_qwz_mass_returns_result_object() -> None:
    result = scan_qwz_mass(np.array([-1.0, 0.0, 1.0]), n_x=4, n_y=4)

    assert isinstance(result, QWZScanResult)


def test_scan_qwz_mass_has_correct_shapes() -> None:
    mass_values = np.array([-1.0, 0.0, 1.0])
    result = scan_qwz_mass(mass_values, n_x=4, n_y=3)

    assert result.mass_values.shape == (3,)
    assert result.spectra.shape == (3, 2 * 4 * 3)
    assert result.gaps.shape == (3,)


def test_scan_qwz_mass_preserves_mass_values() -> None:
    mass_values = np.array([-2.0, 0.0, 2.0])
    result = scan_qwz_mass(mass_values, n_x=4, n_y=4)

    assert np.allclose(result.mass_values, mass_values)


def test_scan_qwz_mass_gaps_are_non_negative() -> None:
    result = scan_qwz_mass(np.linspace(-3.0, 3.0, 7), n_x=4, n_y=4)

    assert np.all(result.gaps >= 0.0)