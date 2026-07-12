from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from toposc_lab.models.kitaev_chain import KitaevChain, KitaevChainParameters
from toposc_lab.observables.localization import localization_profile
from toposc_lab.solvers.exact_diagonalization import ExactDiagonalizationSolver


def main() -> None:
    n_sites = 60

    # Die Parameter liegen ohne Disorder in der topologischen Kitaev-Phase.
    # W = 1.0 fügt zufällige lokale Abweichungen des chemischen Potentials hinzu.
    model = KitaevChain(
        KitaevChainParameters(
            n_sites=n_sites,
            hopping=1.0,
            chemical_potential=0.0,
            pairing=0.5,
            boundary="open",
            disorder_strength=1.0,
            disorder_seed=42,
        )
    )

    result = ExactDiagonalizationSolver().solve(model.hamiltonian())

    # Der Zustand nahe E = 0 ist der Kandidat für einen Majorana-Edge-State.
    state_index = int(np.argmin(np.abs(result.eigenvalues)))

    # Die Kitaev-BdG-Basis ist blockweise gespeichert: erst alle Elektron-
    # und danach alle Loch-Komponenten. Für die Ortsanalyse ordnen wir sie
    # platzweise als (Elektron_0, Loch_0, Elektron_1, Loch_1, ...) an.
    site_major_eigenvectors = np.empty_like(result.eigenvectors)
    site_major_eigenvectors[0::2, :] = result.eigenvectors[:n_sites, :]
    site_major_eigenvectors[1::2, :] = result.eigenvectors[n_sites:, :]

    profile = localization_profile(
        eigenvectors=site_major_eigenvectors,
        state_index=state_index,
        lattice_shape=(n_sites,),
        # Die BdG-Basis enthält Elektron- und Loch-Komponente pro Gitterplatz.
        components_per_site=2,
        component_labels=("electron", "hole"),
        edge_width=2,
    )

    print(f"Disorder seed: {model.params.disorder_seed}")
    print(f"Lowest absolute energy: {abs(result.eigenvalues[state_index]):.3e}")
    print(f"IPR: {profile.inverse_participation_ratio:.4f}")
    print(f"Edge weight: {profile.edge_weight:.4f}")

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=False)

    # Zeige das verwendete zufällige lokale Potential.
    sites = np.arange(n_sites)
    axes[0].plot(sites, model.disorder_profile, marker=".", linestyle="none")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Lattice site")
    axes[0].set_ylabel("δμ")
    axes[0].set_title("Onsite disorder profile")

    # Gesamtes Energiespektrum der konkreten Disorder-Realisierung.
    axes[1].plot(result.eigenvalues, marker=".", linestyle="none")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("State index")
    axes[1].set_ylabel("Energy")
    axes[1].set_title("Disordered Kitaev-chain spectrum")

    # Räumliche Dichte des Zustands mit kleinster Energie.
    axes[2].bar(sites, profile.probability)
    axes[2].set_xlabel("Lattice site")
    axes[2].set_ylabel("Probability density")
    axes[2].set_title("Lowest-energy state localization")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
