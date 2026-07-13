from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from toposc_lab.observables.localization import LocalizationProfile
from toposc_lab.visualization.style import paper_style


def plot_eigenvalue_spectrum(
    eigenvalues: np.ndarray,
    *,
    axes: Axes | None = None,
    title: str = "Energy spectrum",
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Plotte das diskrete Spektrum einer einzelnen Simulation.

    Diese Darstellung unterscheidet sich von ``plot_spectrum_vs_parameter``:
    Die x-Achse ist hier der Eigenzustandsindex statt eines Scanparameters.
    """
    values = np.asarray(eigenvalues, dtype=float)

    if values.ndim != 1 or values.size == 0:
        raise ValueError("eigenvalues must be a non-empty one-dimensional array")

    created_axes = axes is None

    with paper_style():
        if axes is None:
            figure, axes = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
        else:
            figure = axes.figure

        axes.scatter(
            np.arange(values.size),
            values,
            s=14,
            color="tab:blue",
            linewidths=0.0,
        )
        axes.axhline(0.0, color="0.35", linestyle="--", linewidth=0.9)
        axes.set_xlabel("Eigenstate index")
        axes.set_ylabel("Energy")
        axes.set_title(title)

    if created_axes and show:
        plt.show()

    return figure, axes


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

""" AB hier Localization zeug test villeicht später eigen Datei """

def plot_localization_1d(
    profile: LocalizationProfile,
    component: int | None = None,
    xlabel: str = "Lattice site",
    ylabel: str = "Probability density",
    title: str = "Localization profile",
    show: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plotte ein 1D-Lokalisierungsprofil.

    Ohne component wird die Gesamtwahrscheinlichkeit pro Gitterplatz gezeigt.
    Mit component wird nur eine interne Komponente dargestellt.
    """
    if profile.probability.ndim != 1:
        raise ValueError("plot_localization_1d requires a one-dimensional profile")

    if component is None:
        values = profile.probability
    else:
        if not 0 <= component < len(profile.component_labels):
            raise ValueError("component is outside the available component range")

        values = profile.component_probabilities[:, component]
        title = f"{title}: {profile.component_labels[component]}"

    fig, ax = plt.subplots()

    sites = np.arange(values.size)
    ax.bar(sites, values)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_localization_2d(
    profile: LocalizationProfile,
    component: int | None = None,
    xlabel: str = "x",
    ylabel: str = "y",
    title: str = "Localization profile",
    cmap: str = "magma",
    show: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plotte ein 2D-Lokalisierungsprofil als Heatmap.

    Ohne component wird die Gesamtwahrscheinlichkeit gezeigt.
    Mit component wird nur eine interne Komponente dargestellt.
    """
    if profile.probability.ndim != 2:
        raise ValueError("plot_localization_2d requires a two-dimensional profile")

    if component is None:
        values = profile.probability
    else:
        if not 0 <= component < len(profile.component_labels):
            raise ValueError("component is outside the available component range")

        values = profile.component_probabilities[:, :, component]
        title = f"{title}: {profile.component_labels[component]}"

    fig, ax = plt.subplots()

    # Transponieren: Die erste Gitterrichtung x wird horizontal dargestellt.
    image = ax.imshow(
        values.T,
        origin="lower",
        aspect="equal",
        cmap=cmap,
    )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Probability density")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax
