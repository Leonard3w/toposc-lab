from __future__ import annotations

import matplotlib.pyplot as plt

from toposc_lab.models.kitaev_chain import KitaevChain, KitaevChainParameters
from toposc_lab.observables.spectrum import energy_gap
from toposc_lab.solvers.exact_diagonalization import ExactDiagonalizationSolver


def main() -> None:
    params = KitaevChainParameters(
        n_sites=40,
        hopping=1.0,
        chemical_potential=0.0,
        pairing=0.5,
        boundary="open",
    )

    model = KitaevChain(params)
    hamiltonian = model.hamiltonian()

    solver = ExactDiagonalizationSolver()
    result = solver.solve(hamiltonian)

    gap = energy_gap(result.eigenvalues)

    print(f"Hamiltonian shape: {hamiltonian.shape}")
    print(f"Energy gap: {gap:.6f}")

    plt.figure()
    plt.plot(result.eigenvalues, marker="o", linestyle="none")
    plt.xlabel("State index")
    plt.ylabel("Energy")
    plt.title("Kitaev chain spectrum")
    plt.show()


if __name__ == "__main__":
    main()