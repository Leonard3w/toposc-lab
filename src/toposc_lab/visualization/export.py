"""High-resolution export helpers for publication-quality figures."""

from __future__ import annotations

from io import BytesIO
from typing import Literal

from matplotlib.figure import Figure


FigureFormat = Literal["png", "pdf"]


def figure_to_bytes(
    figure: Figure,
    *,
    file_format: FigureFormat = "png",
    dpi: int = 300,
) -> bytes:
    """Serialize a Matplotlib figure for a lossless UI download.

    PNG is useful for slides and notebooks; PDF remains vector-based and is
    the appropriate default for papers.  Both exports use a tight bounding
    box so that labels are retained without unnecessary white margins.
    """
    if not isinstance(figure, Figure):
        raise TypeError("figure must be a matplotlib Figure")

    if file_format not in {"png", "pdf"}:
        raise ValueError("file_format must be either 'png' or 'pdf'")

    if not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("dpi must be a positive integer")

    buffer = BytesIO()
    figure.savefig(
        buffer,
        format=file_format,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    return buffer.getvalue()
