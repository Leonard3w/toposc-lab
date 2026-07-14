from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from toposc_lab.visualization.export import figure_to_bytes


def test_figure_export_creates_png_and_vector_pdf() -> None:
    figure, axes = plt.subplots()
    axes.plot([0.0, 1.0], [0.0, 1.0])

    png = figure_to_bytes(figure, file_format="png")
    pdf = figure_to_bytes(figure, file_format="pdf")

    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf.startswith(b"%PDF")

    plt.close(figure)


def test_figure_export_validates_arguments() -> None:
    figure, _ = plt.subplots()

    with pytest.raises(ValueError, match="file_format"):
        figure_to_bytes(figure, file_format="svg")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="positive integer"):
        figure_to_bytes(figure, dpi=0)

    plt.close(figure)
