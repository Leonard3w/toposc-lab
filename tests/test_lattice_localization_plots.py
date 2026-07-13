from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from toposc_lab.lattices.chain import ChainLattice
from toposc_lab.lattices.honeycomb import HoneycombLattice
from toposc_lab.lattices.ladder import LadderLattice
from toposc_lab.lattices.square import SquareLattice
from toposc_lab.observables.localization import localization_profile
from toposc_lab.visualization.lattice_plots import (
    plot_localization_on_lattice,
    profile_site_probabilities,
)


def test_chain_profile_uses_one_dimensional_plot() -> None:
    lattice = ChainLattice(length=4)
    profile = localization_profile(
        eigenvectors=np.array([[1.0], [0.0], [0.0], [0.0]]),
        state_index=0,
        lattice_shape=(4,),
    )

    figure, axes = plot_localization_on_lattice(lattice, profile, show=False)

    assert axes.get_xlabel() == "Lattice site"
    assert axes.get_ylabel() == r"Probability $|\psi_i|^2$"
    assert len(axes.patches) == 4

    plt.close(figure)


def test_square_profile_uses_heatmap() -> None:
    lattice = SquareLattice(n_x=2, n_y=2)
    profile = localization_profile(
        eigenvectors=np.array([[1.0], [0.0], [0.0], [0.0]]),
        state_index=0,
        lattice_shape=(2, 2),
    )

    figure, axes = plot_localization_on_lattice(lattice, profile, show=False)

    assert axes.get_xlabel() == "x"
    assert axes.get_ylabel() == "y"
    assert len(axes.images) == 1

    plt.close(figure)


def test_ladder_profile_uses_its_real_geometry() -> None:
    lattice = LadderLattice(n_legs=2, length=3)
    profile = localization_profile(
        eigenvectors=np.array([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]]),
        state_index=0,
        lattice_shape=(2, 3),
    )

    figure, axes = plot_localization_on_lattice(lattice, profile, show=False)

    assert axes.get_xlabel() == "Position along ladder"
    assert axes.get_ylabel() == "Leg"
    assert len(axes.collections) == 2

    plt.close(figure)


def test_honeycomb_components_map_to_physical_a_b_sites() -> None:
    lattice = HoneycombLattice(n_x=2, n_y=2)
    profile = localization_profile(
        eigenvectors=np.array(
            [[1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
        ),
        state_index=0,
        lattice_shape=(2, 2),
        components_per_site=2,
        component_labels=("A", "B"),
    )

    probabilities = profile_site_probabilities(lattice, profile)
    figure, axes = plot_localization_on_lattice(lattice, profile, show=False)

    assert probabilities.shape == (lattice.n_sites,)
    assert np.array_equal(probabilities, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert len(axes.collections) == 2

    plt.close(figure)
