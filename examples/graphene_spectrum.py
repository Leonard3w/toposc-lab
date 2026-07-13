from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from toposc_lab.models.graphene_model import (
    GrapheneModel,
    GrapheneParameters,
)
from toposc_lab.solvers.exact_diagonalization import (
    ExactDiagonalizationSolver,
)


def make_graphene_model(boundary: str) -> GrapheneModel:
    """Erzeuge Graphen mit gleichen Parametern, aber anderer Randbedingung."""
    return GrapheneModel(
        GrapheneParameters(
            n_x=9,
            n_y=9,
            hopping=1.0,
            boundary_x=boundary,
            boundary_y=boundary,
        )
    )


def plot_spectrum(
    axis: plt.Axes,
    eigenvalues: np.ndarray,
    title: str,
    panel_label: str,
) -> None:
    """Zeichne ein diskretes Energiespektrum im Paper-Stil."""
    axis.scatter(
        np.arange(eigenvalues.size),
        eigenvalues,
        s=12,
        color="#1f77b4",
        linewidths=0.0,
    )
    axis.axhline(
        0.0,
        color="0.25",
        linewidth=0.8,
        linestyle="--",
    )

    axis.set_xlabel("Eigenstate index")
    axis.set_ylabel("Energy")
    axis.set_title(title, pad=8)
    axis.tick_params(direction="in", top=True, right=True)

    axis.text(
        -0.12,
        1.04,
        panel_label,
        transform=axis.transAxes,
        fontweight="bold",
    )


def main() -> None:
    solver = ExactDiagonalizationSolver()

    # Offene Ränder: endliches Graphen-Flake mit realen Kanten.
    open_model = make_graphene_model("open")
    open_result = solver.solve_model(open_model)

    # Periodische Ränder: Bulk ohne Kanten.
    periodic_model = make_graphene_model("periodic")
    periodic_result = solver.solve_model(periodic_model)

    open_gap = np.min(np.abs(open_result.eigenvalues))
    periodic_gap = np.min(np.abs(periodic_result.eigenvalues))

    print(f"Open-boundary minimum |E|: {open_gap:.3e}")
    print(f"Periodic-boundary minimum |E|: {periodic_gap:.3e}")

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10, 3.8),
        constrained_layout=True,
    )

    plot_spectrum(
        axes[0],
        open_result.eigenvalues,
        title="Graphene with open boundaries",
        panel_label="(a)",
    )

    plot_spectrum(
        axes[1],
        periodic_result.eigenvalues,
        title="Graphene bulk spectrum",
        panel_label="(b)",
    )

    plt.show()


if __name__ == "__main__":
    main()