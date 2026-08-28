from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from toposc_lab.quantum_hall.integer_quantum_hall import IQHEParameters
from toposc_lab.visualization.integer_quantum_hall_plots import (
    integer_quantum_hall_dashboard,
    integer_quantum_hall_edge_figure,
)


def test_iqhe_dashboard_synchronizes_four_scientific_panels() -> None:
    parameters = IQHEParameters(
        magnetic_field_tesla=5.0,
        electron_density_m2=2.4e15,
        maximum_level=6,
        spin_mode="zeeman",
        field_scan_points=151,
    )
    figure, state, scan = integer_quantum_hall_dashboard(parameters)

    assert len(figure.axes) == 4
    assert len(figure.axes[0].lines) >= 1
    assert len(figure.axes[1].lines) >= 2
    assert len(figure.axes[2].lines) >= 3
    assert len(figure.axes[3].lines) >= 2
    assert state.energies_joule.size >= 2 * (parameters.maximum_level + 1)
    assert scan.magnetic_field_tesla.shape == (151,)
    assert np.all(scan.longitudinal_resistivity_ohm >= 0.0)
    plt.close(figure)


def test_edge_figure_contains_confinement_dispersion_velocity_and_current() -> None:
    figure, spectrum = integer_quantum_hall_edge_figure(
        IQHEParameters(temperature_kelvin=0.0)
    )

    assert len(figure.axes) == 4
    assert len(figure.axes[0].lines) >= 2
    assert len(figure.axes[1].lines) >= 3
    assert len(figure.axes[2].lines) >= 2
    assert len(figure.axes[3].lines) >= 3
    assert spectrum.left_mode_count == spectrum.right_mode_count
    plt.close(figure)
