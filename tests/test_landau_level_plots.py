from __future__ import annotations

import matplotlib.pyplot as plt

from toposc_lab.quantum_hall.landau_levels import LandauLevelParameters
from toposc_lab.visualization.landau_level_plots import (
    cyclotron_drift_animation,
    degeneracy_figure,
    landau_gauge_state_figure,
    landau_spectrum_figure,
    symmetric_gauge_state_figure,
)


def test_spectrum_figure_contains_field_scan_and_k_dispersion() -> None:
    parameters = LandauLevelParameters(
        maximum_level=3,
        electric_field_v_per_m=1500.0,
        include_zeeman=True,
    )
    figure, spectra = landau_spectrum_figure(parameters)

    assert len(figure.axes) == 2
    assert spectra.shape == (4, 241)
    assert len(figure.axes[0].lines) >= 4
    assert len(figure.axes[1].lines) >= 4
    plt.close(figure)


def test_landau_gauge_plot_contains_strip_and_normalized_cross_section() -> None:
    parameters = LandauLevelParameters(maximum_level=3, selected_level=1)
    figure, density = landau_gauge_state_figure(parameters)

    assert density.shape == (241, 241)
    assert len(figure.axes) == 3
    assert len(figure.axes[1].lines) >= 2
    plt.close(figure)


def test_symmetric_gauge_plot_contains_ring_density_and_comparison_curves() -> None:
    parameters = LandauLevelParameters(angular_momentum=4)
    figure, density = symmetric_gauge_state_figure(parameters)

    assert density.shape == (241, 241)
    assert len(figure.axes) == 3
    assert len(figure.axes[1].lines) >= 3
    plt.close(figure)


def test_degeneracy_plot_caps_representative_centers() -> None:
    parameters = LandauLevelParameters(
        magnetic_field_tesla=10.0,
        sample_width_m=10.0e-6,
        sample_height_m=10.0e-6,
    )
    figure, centers = degeneracy_figure(parameters)

    assert centers.shape == (500, 2)
    assert len(figure.axes) == 2
    plt.close(figure)


def test_cyclotron_animation_has_playable_frames_and_two_moving_markers() -> None:
    figure = cyclotron_drift_animation(
        LandauLevelParameters(electric_field_v_per_m=1000.0),
        samples=41,
    )

    assert len(figure.data) == 3
    assert len(figure.frames) == 41
    assert tuple(figure.frames[1].traces) == (1, 2)
    assert len(figure.layout.updatemenus[0].buttons) == 2
