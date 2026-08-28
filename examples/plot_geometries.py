from __future__ import annotations

import matplotlib.pyplot as plt

from toposc_lab.geometry import chain, irregular_cluster, ring, square
from toposc_lab.visualization import plot_geometry


def main() -> None:
    geometries = (
        ("Open chain", chain(7)),
        ("Ring", ring(9)),
        ("Open square lattice", square(4, 5)),
        ("Irregular reference cluster", irregular_cluster()),
    )
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    for axes_item, (title, geometry) in zip(axes.flat, geometries, strict=True):
        plot_geometry(
            geometry,
            axes=axes_item,
            title=title,
            show_site_indices=True,
            show=False,
        )

    figure.suptitle("Model-independent geometries")
    plt.show()


if __name__ == "__main__":
    main()
