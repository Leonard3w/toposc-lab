"""Gemeinsamer, zur Publikation passender Matplotlib-Stil."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import matplotlib.pyplot as plt
from cycler import cycler


PAPER_STYLE: dict[str, object] = {
    "axes.labelpad": 6.0,
    "axes.labelsize": 11,
    "axes.prop_cycle": cycler(
        color=("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00")
    ),
    "axes.titlesize": 13,
    "axes.titlepad": 10.0,
    "axes.linewidth": 0.9,
    "figure.dpi": 120,
    "font.size": 11,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
    "lines.solid_capstyle": "round",
    "mathtext.fontset": "dejavusans",
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
        fontsize=12,
        ha="left",
        va="bottom",
    )
