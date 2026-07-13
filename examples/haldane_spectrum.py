from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

from toposc_lab.models.graphene_model import (
    GrapheneModel,
    GrapheneParameters,
)
from toposc_lab.models.haldane_model import (
    HaldaneModel,
    HaldaneModelParameters,
)
from toposc_lab.observables.localization import (
    localization_profile_from_result,
)
from toposc_lab.solvers.exact_diagonalization import (
    ExactDiagonalizationSolver,
)


def plot_spectrum(
    axis,
    eigenvalues: np.ndarray,
    title: str,
    panel_label: str,
) -> None:
    """Zeichne ein diskretes Spektrum im einheitlichen Paper-Stil."""
    axis.scatter(
        np.arange(eigenvalues.size),
        eigenvalues,
        s=12,
        color="#1f77b4",
        linewidths=0.0,
    )
    axis.axhline(
        0.0,
        color="0.25",
        linewidth=0.8,
        linestyle="--",
    )
    axis.set_xlabel("Eigenstate index")
    axis.set_ylabel("Energy")
    axis.set_title(title, pad=8)
    axis.tick_params(direction="in", top=True, right=True)
    axis.text(
        -0.12,
        1.04,
        panel_label,
        transform=axis.transAxes,
        fontweight="bold",
    )


def plot_honeycomb_localization(
    axis,
    model: HaldaneModel,
    site_probabilities: np.ndarray,
) -> None:
    """Zeichne einen Zustand auf der echten Honeycomb-Geometrie."""
    coordinates = model.lattice.coordinates
    maximum_probability = float(np.max(site_probabilities))

    probability_norm = PowerNorm(
        gamma=0.45,
        vmin=0.0,
        vmax=maximum_probability,
    )

    # Zuerst die geometrischen Bonds.
    for bond in model.lattice.bonds:
        source = coordinates[bond.source]
        target = coordinates[bond.target]

        axis.plot(
            [source[0], target[0]],
            [source[1], target[1]],
            color="0.60",
            linewidth=0.7,
            zorder=1,
        )

    # Neutrale Gitterplätze.
    axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=18,
        facecolors="white",
        edgecolors="0.30",
        linewidths=0.45,
        zorder=2,
    )

    # Physikalische Zustandsdichte.
    points = axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=site_probabilities,
        s=22.0 + 150.0 * np.sqrt(
            site_probabilities / maximum_probability
        ),
        cmap="magma",
        norm=probability_norm,
        edgecolors="0.15",
        linewidths=0.35,
        zorder=3,
    )

    axis.set_aspect("equal")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title("Lowest-energy state, open flake", pad=8)
    axis.tick_params(direction="in", top=True, right=True)
    axis.text(
        -0.14,
        1.04,
        "(c)",
        transform=axis.transAxes,
        fontweight="bold",
    )

    return points


def main() -> None:
    solver = ExactDiagonalizationSolver()

    # 9 × 9 ist durch drei teilbar: Die periodische Graphen-Geometrie
    # enthält dadurch diskrete Impulse direkt an den Dirac-Punkten.
    graphene = GrapheneModel(
        GrapheneParameters(
            n_x=9,
            n_y=9,
            hopping=1.0,
            boundary_x="periodic",
            boundary_y="periodic",
        )
    )

    haldane_bulk = HaldaneModel(
        HaldaneModelParameters(
            n_x=9,
            n_y=9,
            nearest_neighbor_hopping=1.0,
            next_nearest_neighbor_hopping=0.2,
            phase=np.pi / 2.0,
            sublattice_mass=0.0,
            boundary_x="periodic",
            boundary_y="periodic",
        )
    )

    # Dieselben Haldane-Parameter, aber mit echten Rändern.
    haldane_open = HaldaneModel(
        HaldaneModelParameters(
            n_x=9,
            n_y=9,
            nearest_neighbor_hopping=1.0,
            next_nearest_neighbor_hopping=0.2,
            phase=np.pi / 2.0,
            sublattice_mass=0.0,
            boundary_x="open",
            boundary_y="open",
        )
    )

    graphene_result = solver.solve_model(graphene)
    haldane_bulk_result = solver.solve_model(haldane_bulk)
    haldane_open_result = solver.solve_model(haldane_open)

    graphene_gap = np.min(np.abs(graphene_result.eigenvalues))
    haldane_gap = np.min(np.abs(haldane_bulk_result.eigenvalues))

    # Kandidat für einen In-Gap- bzw. Randzustand der offenen Flake.
    state_index = int(
        np.argmin(np.abs(haldane_open_result.eigenvalues))
    )

    profile = localization_profile_from_result(
        haldane_open_result,
        state_index=state_index,
    )

    # Die A/B-Komponenten werden wieder zur Reihenfolge der Honeycomb-Sites
    # abgeflacht: A_00, B_00, A_01, B_01, ...
    site_probabilities = profile.component_probabilities.reshape(-1)

    print(f"Graphene bulk minimum |E|: {graphene_gap:.3e}")
    print(f"Haldane bulk minimum |E|: {haldane_gap:.3e}")
    print(
        "Open Haldane lowest |E|: "
        f"{abs(haldane_open_result.eigenvalues[state_index]):.3e}"
    )
    print(f"Open-state IPR: {profile.inverse_participation_ratio:.4f}")

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(14, 3.8),
        constrained_layout=True,
    )

    plot_spectrum(
        axes[0],
        graphene_result.eigenvalues,
        title="Graphene bulk spectrum",
        panel_label="(a)",
    )

    plot_spectrum(
        axes[1],
        haldane_bulk_result.eigenvalues,
        title="Haldane bulk spectrum",
        panel_label="(b)",
    )

    points = plot_honeycomb_localization(
        axes[2],
        haldane_open,
        site_probabilities,
    )

    colorbar = figure.colorbar(points, ax=axes[2])
    colorbar.set_label(r"Site probability $|\psi_i|^2$")

    plt.show()


if __name__ == "__main__":
    main()