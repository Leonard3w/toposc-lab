from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_spectrum_vs_parameter(
    parameter_values: np.ndarray,
    spectra: np.ndarray,
    xlabel: str = "Parameter",
    ylabel: str = "Energy",
    title: str = "Spectrum",
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Plot an energy spectrum as a function of one parameter."""
    fig, ax = plt.subplots()

    for band_index in range(spectra.shape[1]):
        ax.plot(
            parameter_values,
            spectra[:, band_index],
            marker=".",
            linestyle="none",
            markersize=2,
        )

    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_gap_vs_parameter(
    parameter_values: np.ndarray,
    gaps: np.ndarray,
    xlabel: str = "Parameter",
    ylabel: str = "Gap",
    title: str = "Gap",
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Plot a spectral gap as a function of one parameter."""
    fig, ax = plt.subplots()

    ax.plot(parameter_values, gaps, marker=".")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax