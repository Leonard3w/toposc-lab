from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from toposc_lab.scans.haldane_scan import scan_haldane_mass


def main() -> None:
    next_nearest_hopping = 0.2
    phase = np.pi / 2.0
    mass_values = np.linspace(-2.0, 2.0, 81)

    result = scan_haldane_mass(
        mass_values=mass_values,
        nearest_neighbor_hopping=1.0,
        next_nearest_neighbor_hopping=next_nearest_hopping,
        phase=phase,
        # Durch drei teilbar: Die Dirac-Punkte liegen exakt auf dem Raster.
        n_k=24,
    )

    # Analytische Grenze des Haldane-Übergangs.
    critical_mass = (
        3.0
        * np.sqrt(3.0)
        * next_nearest_hopping
        * np.sin(phase)
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.0, 3.8),
        constrained_layout=True,
        sharex=True,
    )

    axes[0].plot(
        result.mass_values,
        result.bulk_gaps,
        color="#1f77b4",
        marker="o",
        markersize=3,
        linewidth=1.2,
    )
    axes[0].set_xlabel("Sublattice mass M")
    axes[0].set_ylabel("Bulk band gap")
    axes[0].set_title("Bulk-gap closing", pad=8)
    axes[0].text(
        -0.13,
        1.04,
        "(a)",
        transform=axes[0].transAxes,
        fontweight="bold",
    )

    axes[1].plot(
        result.mass_values,
        result.chern_numbers,
        color="#d62728",
        marker="o",
        markersize=3,
        linewidth=1.2,
        drawstyle="steps-mid",
    )
    axes[1].axhline(0.0, color="0.25", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("Sublattice mass M")
    axes[1].set_ylabel("Chern number")
    axes[1].set_yticks((-1, 0, 1))
    axes[1].set_title("Topological phase", pad=8)
    axes[1].text(
        -0.13,
        1.04,
        "(b)",
        transform=axes[1].transAxes,
        fontweight="bold",
    )

    for axis in axes:
        axis.axvline(
            -critical_mass,
            color="0.35",
            linewidth=0.8,
            linestyle="--",
        )
        axis.axvline(
            critical_mass,
            color="0.35",
            linewidth=0.8,
            linestyle="--",
        )
        axis.tick_params(direction="in", top=True, right=True)

    plt.show()


if __name__ == "__main__":
    main()
