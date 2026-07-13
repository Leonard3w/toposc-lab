from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from toposc_lab.lattices.honeycomb import HoneycombLattice


def main() -> None:
    lattice = HoneycombLattice(
        n_x=6,
        n_y=5,
        boundary_x="open",
        boundary_y="open",
    )

    coordinates = lattice.coordinates

    fig, ax = plt.subplots(figsize=(8, 6))

    # Zuerst die Bonds zeichnen, damit sie hinter den Gitterplätzen liegen.
    for bond in lattice.bonds:
        source = coordinates[bond.source]
        target = coordinates[bond.target]

        ax.plot(
            [source[0], target[0]],
            [source[1], target[1]],
            color="black",
            linewidth=1.0,
            zorder=1,
        )

    # A- und B-Sublattice werden verschieden eingefärbt.
    a_sites = np.array(
        [site for site in range(lattice.n_sites) if lattice.sublattice(site) == "A"]
    )
    b_sites = np.array(
        [site for site in range(lattice.n_sites) if lattice.sublattice(site) == "B"]
    )

    ax.scatter(
        coordinates[a_sites, 0],
        coordinates[a_sites, 1],
        color="tab:blue",
        label="A sublattice",
        s=50,
        zorder=2,
    )
    ax.scatter(
        coordinates[b_sites, 0],
        coordinates[b_sites, 1],
        color="tab:orange",
        label="B sublattice",
        s=50,
        zorder=2,
    )

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Honeycomb lattice")
    ax.legend()
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()