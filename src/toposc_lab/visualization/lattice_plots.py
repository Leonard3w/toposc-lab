"""Geometrieabhaengige Darstellung von Lokalisierungsprofilen."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import PowerNorm
from matplotlib.figure import Figure

from toposc_lab.lattices.base import BaseLattice
from toposc_lab.lattices.chain import ChainLattice
from toposc_lab.lattices.honeycomb import HoneycombLattice
from toposc_lab.lattices.ladder import LadderLattice
from toposc_lab.lattices.ribbon import RibbonLattice
from toposc_lab.lattices.square import SquareLattice
from toposc_lab.observables.localization import LocalizationProfile
from toposc_lab.visualization.style import paper_style


def profile_site_probabilities(
    lattice: BaseLattice,
    profile: LocalizationProfile,
) -> np.ndarray:
    """Ordne ein Lokalisierungsprofil den physikalischen Gitterplaetzen zu.

    Bei Ketten und quadratischen Gittern ist die Gesamtwahrscheinlichkeit
    bereits eine Wahrscheinlichkeit pro Platz. Beim Honeycomb-Gitter sind
    A und B als Komponenten einer Einheitszelle gespeichert; dort wird die
    komponentenaufgeloeste Wahrscheinlichkeit auf die realen A/B-Plaetze
    abgeflacht.
    """
    probability = np.asarray(profile.probability)

    if probability.size == lattice.n_sites:
        return probability.reshape(-1)

    component_probabilities = np.asarray(profile.component_probabilities)

    if component_probabilities.size == lattice.n_sites:
        return component_probabilities.reshape(-1)

    raise ValueError(
        "LocalizationProfile cannot be mapped to this lattice: "
        f"profile has {probability.size} spatial values and "
        f"{component_probabilities.size} component values, but lattice has "
        f"{lattice.n_sites} sites"
    )


def _draw_local_bonds(axes: Axes, lattice: BaseLattice) -> None:
    """Zeichne Bonds, aber keine langen Linien ueber periodische Naehte."""
    coordinates = lattice.coordinates
    bond_lengths = np.asarray(
        [
            np.linalg.norm(coordinates[bond.target] - coordinates[bond.source])
            for bond in lattice.bonds
        ],
        dtype=float,
    )

    nonzero_lengths = bond_lengths[bond_lengths > 0.0]
    maximum_length = (
        1.5 * float(np.median(nonzero_lengths))
        if nonzero_lengths.size
        else np.inf
    )

    for bond, length in zip(lattice.bonds, bond_lengths, strict=True):
        if length > maximum_length:
            continue

        source = coordinates[bond.source]
        target = coordinates[bond.target]
        axes.plot(
            [source[0], target[0]],
            [source[1], target[1]],
            color="0.60",
            linewidth=0.75,
            zorder=1,
        )


def _plot_graph_localization(
    axes: Axes,
    lattice: BaseLattice,
    probabilities: np.ndarray,
    *,
    cmap: str,
) -> None:
    """Zeichne die reale Geometrie eines mehrbeinigen oder unregulaeren Gitters."""
    coordinates = lattice.coordinates
    maximum_probability = float(np.max(probabilities))

    _draw_local_bonds(axes, lattice)

    axes.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=24,
        facecolors="white",
        edgecolors="0.25",
        linewidths=0.55,
        zorder=2,
    )

    points = axes.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=probabilities,
        s=26.0 + 180.0 * np.sqrt(probabilities / maximum_probability),
        cmap=cmap,
        norm=PowerNorm(gamma=0.45, vmin=0.0, vmax=maximum_probability),
        edgecolors="0.15",
        linewidths=0.4,
        zorder=3,
    )

    colorbar = axes.figure.colorbar(points, ax=axes)
    colorbar.set_label(r"Site probability $|\psi_i|^2$")
    axes.set_aspect("equal")
    axes.set_xlabel("x")
    axes.set_ylabel("y")


def _plot_square_localization(
    axes: Axes,
    lattice: BaseLattice,
    probabilities: np.ndarray,
    *,
    cmap: str,
) -> None:
    """Zeichne eine rechteckige Geometrie als Ortsdichte-Heatmap."""
    coordinates = lattice.coordinates.astype(int)
    x_coordinates = np.unique(coordinates[:, 0])
    y_coordinates = np.unique(coordinates[:, 1])
    values = np.zeros((x_coordinates.size, y_coordinates.size), dtype=float)

    x_indices = {value: index for index, value in enumerate(x_coordinates)}
    y_indices = {value: index for index, value in enumerate(y_coordinates)}

    for coordinate, probability in zip(coordinates, probabilities, strict=True):
        values[x_indices[coordinate[0]], y_indices[coordinate[1]]] = probability

    image = axes.imshow(
        values.T,
        origin="lower",
        aspect="equal",
        cmap=cmap,
        extent=(
            x_coordinates[0] - 0.5,
            x_coordinates[-1] + 0.5,
            y_coordinates[0] - 0.5,
            y_coordinates[-1] + 0.5,
        ),
    )
    colorbar = axes.figure.colorbar(image, ax=axes)
    colorbar.set_label(r"Site probability $|\psi_i|^2$")
    axes.set_xlabel("x")
    axes.set_ylabel("y")


def _plot_chain_localization(
    axes: Axes,
    lattice: ChainLattice,
    probabilities: np.ndarray,
    *,
    cmap: str,
) -> None:
    """Zeichne eine Kette als gut lesbares 1D-Wahrscheinlichkeitsprofil."""
    positions = lattice.coordinates[:, 0]
    colors = plt.get_cmap(cmap)(probabilities / float(np.max(probabilities)))

    axes.bar(
        positions,
        probabilities,
        color=colors,
        edgecolor="0.2",
        linewidth=0.3,
        width=0.85,
    )
    axes.set_xlabel("Lattice site")
    axes.set_ylabel(r"Probability $|\psi_i|^2$")


def plot_localization_on_lattice(
    lattice: BaseLattice,
    profile: LocalizationProfile,
    *,
    axes: Axes | None = None,
    title: str = "Localization profile",
    cmap: str = "magma",
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Plotte ein Profil automatisch passend zur Geometrie des Gitters.

    * ``ChainLattice``: 1D-Balkenprofil.
    * ``SquareLattice`` und ``RibbonLattice``: Heatmap.
    * ``LadderLattice`` und ``HoneycombLattice``: echte Geometrie mit Bonds,
      Punktgroesse und Farbe als Wahrscheinlichkeit.
    * andere ``BaseLattice``-Unterklassen: allgemeine 2D-Geometrieansicht.
    """
    probabilities = profile_site_probabilities(lattice, profile)
    created_axes = axes is None

    with paper_style():
        if axes is None:
            figure, axes = plt.subplots(figsize=(6.2, 4.5), constrained_layout=True)
        else:
            figure = axes.figure

        if isinstance(lattice, ChainLattice):
            _plot_chain_localization(axes, lattice, probabilities, cmap=cmap)
        elif isinstance(lattice, (SquareLattice, RibbonLattice)):
            _plot_square_localization(axes, lattice, probabilities, cmap=cmap)
        elif isinstance(lattice, (LadderLattice, HoneycombLattice)):
            _plot_graph_localization(axes, lattice, probabilities, cmap=cmap)
            if isinstance(lattice, LadderLattice):
                axes.set_xlabel("Position along ladder")
                axes.set_ylabel("Leg")
                axes.set_yticks(range(lattice.n_legs))
        else:
            _plot_graph_localization(axes, lattice, probabilities, cmap=cmap)

        axes.set_title(title)

    if created_axes and show:
        plt.show()

    return figure, axes
