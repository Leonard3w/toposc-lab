from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

from toposc_lab.models.kitaev_ladder import (
    KitaevLadder,
    KitaevLadderParameters,
)
from toposc_lab.observables.localization import (
    left_edge_weight,
    localization_profile_from_result,
    right_edge_weight,
)
from toposc_lab.solvers.exact_diagonalization import (
    ExactDiagonalizationSolver,
)


def main() -> None:
    # Zwei gekoppelte topologische Kitaev-Ketten.
    model = KitaevLadder(
        KitaevLadderParameters(
            n_legs=2,
            length=40,
            hopping=1.0,
            chemical_potential=0.0,
            pairing=0.7,
            rung_hopping=0.25,
            rung_pairing=0.0,
            boundary_length="open",
            boundary_legs="open",
        )
    )

    result = ExactDiagonalizationSolver().solve_model(model)

    # Zustand mit Energie am nächsten bei null:
    # Kandidat für einen hybridisierten Majorana-Randzustand.
    state_index = int(np.argmin(np.abs(result.eigenvalues)))

    profile = localization_profile_from_result(
        result=result,
        state_index=state_index,
        edge_width=2,
    )

    # profile.probability hat die Form (leg, position).
    left_weight = left_edge_weight(
        profile.probability,
        edge_width=2,
        axis=1,
    )
    right_weight = right_edge_weight(
        profile.probability,
        edge_width=2,
        axis=1,
    )

    print(f"Lowest absolute energy: {abs(result.eigenvalues[state_index]):.3e}")
    print(f"IPR: {profile.inverse_participation_ratio:.4f}")
    print(f"Left-end weight: {left_weight:.4f}")
    print(f"Right-end weight: {right_weight:.4f}")

    # Einheitliches, zurückhaltendes Layout für eine Publikationsabbildung.
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 3.6),
        constrained_layout=True,
    )

    # Gesamtes BdG-Energiespektrum.
    axes[0].scatter(
        np.arange(result.eigenvalues.size),
        result.eigenvalues,
        s=13,
        color="#1f77b4",
        linewidths=0.0,
    )
    axes[0].axhline(
        0.0,
        color="0.25",
        linewidth=0.8,
        linestyle="--",
    )
    axes[0].set_xlabel("Eigenstate index")
    axes[0].set_ylabel("Energy")
    axes[0].set_title("Spectrum", pad=8)
    axes[0].text(
        -0.13,
        1.04,
        "(a)",
        transform=axes[0].transAxes,
        fontweight="bold",
    )

    # Die Ladder wird als ihre tatsächliche Geometrie gezeichnet.
    # Dezente Bonds zeigen die Geometrie; die farbigen Punkte tragen die
    # physikalische Information |ψ_i|².
    coordinates = model.lattice.coordinates
    site_probabilities = profile.probability.reshape(-1)
    maximum_probability = float(np.max(site_probabilities))

    probability_norm = PowerNorm(
        gamma=0.45,
        vmin=0.0,
        vmax=maximum_probability,
    )

    for bond in model.lattice.bonds:
        source = coordinates[bond.source]
        target = coordinates[bond.target]
        axes[1].plot(
            [source[0], target[0]],
            [source[1], target[1]],
            color="0.55",
            linewidth=0.8,
            zorder=1,
        )

    # Alle Plätze bleiben als neutrale Gitterpunkte sichtbar.
    axes[1].scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=24,
        facecolors="white",
        edgecolors="0.25",
        linewidths=0.55,
        zorder=2,
    )

    # Die Wurzel-Skalierung verhindert, dass ein einzelner stark
    # lokalisierter Punkt die gesamte Leiterzeichnung verdeckt.
    points = axes[1].scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=site_probabilities,
        s=28.0 + 190.0 * np.sqrt(site_probabilities / maximum_probability),
        cmap="magma",
        norm=probability_norm,
        edgecolors="0.15",
        linewidths=0.4,
        zorder=3,
    )

    axes[1].set_xlim(-1.0, model.params.length)
    axes[1].set_ylim(-0.35, model.params.n_legs - 0.65)
    axes[1].set_yticks(range(model.params.n_legs))
    axes[1].set_xlabel("Position along ladder")
    axes[1].set_ylabel("Leg")
    axes[1].set_title("Lowest-energy state", pad=8)
    axes[1].text(
        -0.13,
        1.04,
        "(b)",
        transform=axes[1].transAxes,
        fontweight="bold",
    )

    colorbar = figure.colorbar(points, ax=axes[1])
    colorbar.set_label(r"Site probability $|\psi_i|^2$")

    for axis in axes:
        axis.tick_params(direction="in", top=True, right=True)

    plt.show()


if __name__ == "__main__":
    main()
