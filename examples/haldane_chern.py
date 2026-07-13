from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from toposc_lab.models.haldane_model import (
    HaldaneModel,
    HaldaneModelParameters,
)
from toposc_lab.observables.berry import berry_curvature, chern_number


def main() -> None:
    model = HaldaneModel(
        HaldaneModelParameters(
            n_x=3,
            n_y=3,
            nearest_neighbor_hopping=1.0,
            next_nearest_neighbor_hopping=0.2,
            phase=np.pi / 2.0,
            sublattice_mass=0.0,
        )
    )

    result = berry_curvature(model.bloch_hamiltonian, n_k=41)
    chern = chern_number(model.bloch_hamiltonian, n_k=41)

    print(f"Numerical Chern number: {result.chern_number:.8f}")
    print(f"Quantized Chern number: {chern:+d}")

    maximum = float(np.max(np.abs(result.berry_curvature)))

    figure, axis = plt.subplots(
        figsize=(6.0, 4.8),
        constrained_layout=True,
    )

    image = axis.imshow(
        result.berry_curvature.T,
        origin="lower",
        extent=(-np.pi, np.pi, -np.pi, np.pi),
        cmap="coolwarm",
        vmin=-maximum,
        vmax=maximum,
        aspect="equal",
    )

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(r"Berry curvature $\Omega(k_x, k_y)$")

    axis.set_xlabel(r"$k_x$")
    axis.set_ylabel(r"$k_y$")
    axis.set_title(f"Haldane Berry curvature, C = {chern:+d}")
    axis.tick_params(direction="in", top=True, right=True)

    plt.show()


if __name__ == "__main__":
    main()
