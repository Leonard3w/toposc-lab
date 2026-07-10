from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from toposc_lab.models.ssh_chain import SSHChain, SSHChainParameters
from toposc_lab.solvers.exact_diagonalization import ExactDiagonalizationSolver


def main() -> None:
    v_values = np.linspace(0.0, 2.0, 101)
    solver = ExactDiagonalizationSolver()
    spectra = []

    for v in v_values:
        model = SSHChain(
            SSHChainParameters(
                n_cells=30,
                v=float(v),
                w=1.0,
                boundary="open",
            )
        )
        spectra.append(solver.solve(model.hamiltonian()).eigenvalues)

    spectra_array = np.asarray(spectra)

    fig, ax = plt.subplots()

    for state in range(spectra_array.shape[1]):
        ax.plot(v_values, spectra_array[:, state], ".", markersize=2)

    ax.axvline(1.0, color="black", linestyle="--", label="v = w")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Intrazell-Hopping v (w = 1)")
    ax.set_ylabel("Energie")
    ax.set_title("SSH-Kette: Spektrum und Phasenübergang")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()