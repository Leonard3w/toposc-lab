from __future__ import annotations

import numpy as np

from toposc_lab.models.qwz_model import QWZModel, QWZModelParameters
from toposc_lab.observables.localization import localization_profile_from_result
from toposc_lab.solvers.exact_diagonalization import ExactDiagonalizationSolver
from toposc_lab.visualization.lattice_plots import plot_localization_on_lattice


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

    result = ExactDiagonalizationSolver().solve_model(model)

    # Zustand mit der niedrigsten Energie nahe E = 0 auswählen.
    state_index = int(np.argmin(np.abs(result.eigenvalues)))

    profile = localization_profile_from_result(
        result=result,
        state_index=state_index,
        edge_width=1,
    )

    print(f"Selected energy: {result.eigenvalues[state_index]:.6f}")
    print(f"Center of mass: {profile.center_of_mass}")
    print(f"IPR: {profile.inverse_participation_ratio:.4f}")
    print(f"Edge weight: {profile.edge_weight:.4f}")

    # Gesamtwahrscheinlichkeit über beide Orbitale.
    plot_localization_on_lattice(
        model.lattice,
        profile,
        title="QWZ edge-state localization",
    )


if __name__ == "__main__":
    main()
    
