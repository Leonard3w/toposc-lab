from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from toposc_lab.models.qwz_model import QWZModel, QWZModelParameters
from toposc_lab.observables.localization import localization_profile
from toposc_lab.solvers.exact_diagonalization import ExactDiagonalizationSolver
from toposc_lab.visualization.plots import plot_localization_2d


def main() -> None:
    n_x = 12
    n_y = 12

    # Offene Ränder: Dadurch können QWZ-Randzustände auftreten.
    # m = -1 liegt in einer topologischen Phase.
    model = QWZModel(
        QWZModelParameters(
            n_x=n_x,
            n_y=n_y,
            mass=-1.0,
            boundary_x="open",
            boundary_y="open",
        )
    )

    result = ExactDiagonalizationSolver().solve(model.hamiltonian())

    # Zustand mit der niedrigsten Energie nahe E = 0 auswählen.
    state_index = int(np.argmin(np.abs(result.eigenvalues)))

    profile = localization_profile(
        eigenvectors=result.eigenvectors,
        state_index=state_index,
        lattice_shape=(n_x, n_y),
        components_per_site=2,
        component_labels=("orbital 1", "orbital 2"),
        edge_width=1,
    )

    print(f"Selected energy: {result.eigenvalues[state_index]:.6f}")
    print(f"Center of mass: {profile.center_of_mass}")
    print(f"IPR: {profile.inverse_participation_ratio:.4f}")
    print(f"Edge weight: {profile.edge_weight:.4f}")

    # Gesamtwahrscheinlichkeit über beide Orbitale.
    plot_localization_2d(
        profile,
        title="QWZ edge-state localization",
        show=False,
    )

    plt.show()


if __name__ == "__main__":
    main()
    