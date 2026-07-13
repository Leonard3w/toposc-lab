from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

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

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Gesamtes BdG-Energiespektrum.
    axes[0].plot(
        result.eigenvalues,
        marker=".",
        linestyle="none",
    )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xlabel("State index")
    axes[0].set_ylabel("Energy")
    axes[0].set_title("Kitaev-ladder spectrum")

    # Räumliche Wahrscheinlichkeit des niederenergetischen Zustands.
    image = axes[1].imshow(
        profile.probability,
        origin="lower",
        aspect="auto",
        cmap="magma",
    )
    axes[1].set_xlabel("Position along ladder")
    axes[1].set_ylabel("Leg")
    axes[1].set_title("Lowest-energy state localization")

    colorbar = figure.colorbar(image, ax=axes[1])
    colorbar.set_label("Probability density")

    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()