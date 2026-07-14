from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from toposc_lab.visualization.matrix_plots import matrix_values, plot_matrix


def test_matrix_values_returns_requested_complex_representation() -> None:
    matrix = np.array([[1.0 + 2.0j, -3.0j], [4.0, -1.0 - 1.0j]])

    assert np.array_equal(matrix_values(matrix, "real"), matrix.real)
    assert np.array_equal(matrix_values(matrix, "imaginary"), matrix.imag)
    assert np.array_equal(matrix_values(matrix, "magnitude"), np.abs(matrix))
    assert np.array_equal(matrix_values(matrix, "phase"), np.angle(matrix))


def test_matrix_values_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="square"):
        matrix_values(np.zeros((2, 3)), "magnitude")

    with pytest.raises(ValueError, match="representation"):
        matrix_values(np.eye(2), "invalid")  # type: ignore[arg-type]


def test_plot_matrix_creates_heatmap_and_colorbar() -> None:
    figure, axes = plot_matrix(np.array([[0.0, 1.0j], [-1.0j, 0.0]]), show=False)

    assert axes.get_xlabel() == "Column basis index j"
    assert axes.get_ylabel() == "Row basis index i"
    assert len(figure.axes) == 2  # Heatmap axes and colorbar axes.

    plt.close(figure)
