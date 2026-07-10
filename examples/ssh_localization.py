from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from toposc_lab.models.ssh_chain import SSHChain, SSHChainParameters
from toposc_lab.observables.localization import localization_profile
from toposc_lab.solvers.exact_diagonalization import ExactDiagonalizationSolver
from toposc_lab.visualization.plots import plot_localization_1d


def main() -> None:
    # v < w: offene SSH-Kette in der topologischen Phase.
    n_cells = 30
    model = SSHChain(
        SSHChainParameters(
            n_cells=n_cells,
            v=0.3,
            w=1.0,
            boundary="open",
        )
    )

    result = ExactDiagonalizationSolver().solve(model.hamiltonian())

    # Der Zustand mit der kleinsten |E|-Energie ist ein Edge State.
    state_index = int(np.argmin(np.abs(result.eigenvalues)))

    profile = localization_profile(
        eigenvectors=result.eigenvectors,
        state_index=state_index,
        # Die SSH-Kette besitzt zwei Plätze pro Einheitszelle.
        lattice_shape=(2 * n_cells,),
        components_per_site=1,
        edge_width=2,
    )

    print(f"Selected energy: {result.eigenvalues[state_index]:.3e}")
    print(f"Center of mass: {profile.center_of_mass[0]:.2f}")
    print(f"IPR: {profile.inverse_participation_ratio:.3f}")
    print(f"Participation ratio: {profile.participation_ratio:.2f}")
    print(f"Edge weight: {profile.edge_weight:.3f}")

    plot_localization_1d(
        profile,
        title="SSH edge-state localization",
        show=False,
    )

    plt.show()


if __name__ == "__main__":
    main()