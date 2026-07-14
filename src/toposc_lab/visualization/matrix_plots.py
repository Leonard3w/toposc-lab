"""Professional matrix visualizations for Hamiltonian inspection."""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from toposc_lab.visualization.style import paper_style


MatrixRepresentation = Literal["real", "imaginary", "magnitude", "phase"]


def matrix_values(matrix: np.ndarray, representation: MatrixRepresentation) -> np.ndarray:
    """Return one real-valued representation of a square complex matrix."""
    values = np.asarray(matrix, dtype=complex)

    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be a square two-dimensional array")

    if representation == "real":
        return values.real
    if representation == "imaginary":
        return values.imag
    if representation == "magnitude":
        return np.abs(values)
    if representation == "phase":
        return np.angle(values)

    raise ValueError("representation must be real, imaginary, magnitude or phase")


def plot_matrix(
    matrix: np.ndarray,
    *,
    representation: MatrixRepresentation = "magnitude",
    axes: Axes | None = None,
    title: str = "Hamiltonian matrix",
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Plot a matrix using a representation-appropriate publication colormap."""
    values = matrix_values(matrix, representation)
    created_axes = axes is None

    with paper_style():
        if axes is None:
            figure, axes = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
        else:
            figure = axes.figure

        if representation in {"real", "imaginary"}:
            maximum = float(np.max(np.abs(values)))
            maximum = maximum if maximum > 0.0 else 1.0
            image = axes.imshow(
                values,
                origin="upper",
                cmap="RdBu_r",
                vmin=-maximum,
                vmax=maximum,
                interpolation="nearest",
            )
            colorbar_label = f"{representation.capitalize()} H"
        elif representation == "phase":
            image = axes.imshow(
                values,
                origin="upper",
                cmap="twilight",
                vmin=-np.pi,
                vmax=np.pi,
                interpolation="nearest",
            )
            colorbar_label = "Phase arg(H)"
        else:
            image = axes.imshow(
                values,
                origin="upper",
                cmap="magma",
                vmin=0.0,
                interpolation="nearest",
            )
            colorbar_label = "Magnitude |H|"

        colorbar = figure.colorbar(image, ax=axes, pad=0.02)
        colorbar.set_label(colorbar_label)
        axes.set_xlabel("Column basis index j")
        axes.set_ylabel("Row basis index i")
        axes.set_title(title)

    if created_axes and show:
        plt.show()

    return figure, axes
