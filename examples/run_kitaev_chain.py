from toposc_lab.models.kitaev_chain import KitaevChain, KitaevChainParameters
from toposc_lab.observables.spectrum import energy_gap
from toposc_lab.solvers.exact_diagonalization import ExactDiagonalizationSolver
from toposc_lab.visualization.plots import plot_spectrum


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

    plot_spectrum(result.eigenvalues)


if __name__ == "__main__":
    main()

    