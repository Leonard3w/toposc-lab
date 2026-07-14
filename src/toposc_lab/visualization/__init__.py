"""Plots fuer Spektren, Observablen und Gittergeometrien."""

from toposc_lab.visualization.study_plots import (
    plot_study_comparison,
    plot_study_observable,
    plot_study_spectrum,
)
from toposc_lab.visualization.lattice_plots import (
    plot_localization_on_lattice,
    profile_site_probabilities,
)
from toposc_lab.visualization.plots import plot_eigenvalue_spectrum
from toposc_lab.visualization.export import figure_to_bytes
from toposc_lab.visualization.style import add_panel_label, paper_style

__all__ = [
    "add_panel_label",
    "figure_to_bytes",
    "paper_style",
    "plot_eigenvalue_spectrum",
    "plot_localization_on_lattice",
    "plot_study_comparison",
    "plot_study_observable",
    "plot_study_spectrum",
    "profile_site_probabilities",
]
