import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from toposc_lab.observables.localization import localization_profile
from toposc_lab.visualization.plots import (
    plot_localization_1d,
    plot_localization_2d,
)


def test_plot_localization_1d_returns_figure_and_axes() -> None:
    profile = localization_profile(
        eigenvectors=np.array([[1.0], [0.0], [0.0]]),
        state_index=0,
        lattice_shape=(3,),
    )

    fig, ax = plot_localization_1d(profile, show=False)

    assert fig is not None
    assert ax.get_xlabel() == "Lattice site"

    plt.close(fig)


def test_plot_localization_2d_returns_figure_and_axes() -> None:
    profile = localization_profile(
        eigenvectors=np.array([[1.0], [0.0], [0.0], [0.0]]),
        state_index=0,
        lattice_shape=(2, 2),
    )

    fig, ax = plot_localization_2d(profile, show=False)

    assert fig is not None
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"

    plt.close(fig)