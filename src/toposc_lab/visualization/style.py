"""Gemeinsamer, zur Publikation passender Matplotlib-Stil."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import matplotlib.pyplot as plt


PAPER_STYLE: dict[str, object] = {
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "axes.linewidth": 0.9,
    "font.size": 11,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "xtick.direction": "in",
    "xtick.top": True,
    "xtick.labelsize": 10,
    "ytick.direction": "in",
    "ytick.right": True,
    "ytick.labelsize": 10,
}


@contextmanager
def paper_style() -> Iterator[None]:
    """Wende den TopOSC-Lab-Stil nur innerhalb eines ``with``-Blocks an.

    Dadurch veraendert eine Abbildung nicht still den globalen Matplotlib-Stil
    anderer Programme oder eines spaeteren interaktiven Interfaces.
    """
    with plt.rc_context(PAPER_STYLE):
        yield


def add_panel_label(
    axes: plt.Axes,
    label: str,
    *,
    x: float = -0.13,
    y: float = 1.04,
) -> None:
    """Fuege ein Paper-Panel-Label wie ``(a)`` oben links hinzu."""
    axes.text(
        x,
        y,
        label,
        transform=axes.transAxes,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
