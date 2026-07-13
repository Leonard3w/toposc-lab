import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from toposc_lab.visualization.plots import (
    plot_eigenvalue_spectrum,
    plot_gap_vs_parameter,
    plot_spectrum_vs_parameter,
)


def test_plot_eigenvalue_spectrum_returns_figure_and_axes() -> None:
    figure, axes = plot_eigenvalue_spectrum(
        np.array([-1.0, 0.0, 1.0]),
        show=False,
    )

    assert axes.get_xlabel() == "Eigenstate index"
    assert axes.get_ylabel() == "Energy"
    assert len(axes.collections) == 1

    plt.close(figure)


def test_plot_spectrum_vs_parameter_returns_figure_and_axes() -> None:
    parameter_values = np.array([-1.0, 0.0, 1.0])
    spectra = np.array(
        [
            [-2.0, -1.0, 1.0, 2.0],
            [-1.5, -0.5, 0.5, 1.5],
            [-2.0, -1.0, 1.0, 2.0],
        ],
    )

    fig, ax = plot_spectrum_vs_parameter(
        parameter_values=parameter_values,
        spectra=spectra,
        show=False,
    )

    assert fig is not None
    assert ax is not None
    assert ax.get_xlabel() == "Parameter"
    assert ax.get_ylabel() == "Energy"

    plt.close(fig)


def test_plot_gap_vs_parameter_returns_figure_and_axes() -> None:
    parameter_values = np.array([-1.0, 0.0, 1.0])
    gaps = np.array([0.5, 0.0, 0.5])

    fig, ax = plot_gap_vs_parameter(
        parameter_values=parameter_values,
        gaps=gaps,
        show=False,
    )

    assert fig is not None
    assert ax is not None
    assert ax.get_xlabel() == "Parameter"
    assert ax.get_ylabel() == "Gap"

    plt.close(fig)
