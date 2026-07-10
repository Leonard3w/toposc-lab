from __future__ import annotations

import numpy as np

from toposc_lab.scans.ssh_scan import scan_ssh_ratio, scan_ssh_v, scan_ssh_w


def test_scan_ssh_v_has_correct_shapes() -> None:
    v_values = np.array([0.3, 1.0, 1.5])
    n_cells = 10

    result = scan_ssh_v(v_values, n_cells=n_cells, w=1.0)

    assert result.parameter_values.shape == (3,)
    assert result.spectra.shape == (3, 2 * n_cells)
    assert result.bulk_gaps.shape == (3,)
    assert result.edge_gaps.shape == (3,)
    assert result.zero_mode_counts.shape == (3,)


def test_scan_ssh_w_preserves_parameter_values() -> None:
    w_values = np.array([0.3, 1.0, 1.5])

    result = scan_ssh_w(w_values, n_cells=10, v=1.0)

    assert np.allclose(result.parameter_values, w_values)


def test_scan_ssh_ratio_has_edge_states_in_topological_phase() -> None:
    # v/w = 0.3 liegt klar in der topologischen Phase.
    result = scan_ssh_ratio(
        ratio_values=np.array([0.3]),
        n_cells=30,
        w=1.0,
        boundary="open",
    )

    assert result.zero_mode_counts[0] == 2


def test_scan_ssh_ratio_has_no_edge_states_in_trivial_phase() -> None:
    # v/w = 1.5 liegt in der trivialen Phase.
    result = scan_ssh_ratio(
        ratio_values=np.array([1.5]),
        n_cells=30,
        w=1.0,
        boundary="open",
    )

    assert result.zero_mode_counts[0] == 0