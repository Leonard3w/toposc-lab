"""Plots fuer Spektren, Observablen und Gittergeometrien."""

from toposc_lab.visualization.candidate_plots import (
    TopCandidateVisualization,
    plot_top_ranked_candidates,
)
from toposc_lab.visualization.study_plots import (
    plot_study_comparison,
    plot_study_observable,
    plot_study_spectrum,
)
from toposc_lab.visualization.lattice_plots import (
    plot_localization_on_lattice,
    profile_site_probabilities,
)
from toposc_lab.visualization.geometry_plots import plot_geometry
from toposc_lab.visualization.plots import plot_eigenvalue_spectrum
from toposc_lab.visualization.export import figure_to_bytes
from toposc_lab.visualization.matrix_plots import matrix_values, plot_matrix
from toposc_lab.visualization.style import add_panel_label, paper_style
from toposc_lab.visualization.quantum_gas_plots import (
    bose_condensation_figure,
    quantum_statistics_schematic,
    quantum_gas_state_figure,
)
from toposc_lab.visualization.ensemble_plots import (
    bose_grand_canonical_figure,
    bose_microcanonical_figure,
    classical_ensemble_motion_figure,
)
from toposc_lab.visualization.landau_level_plots import (
    cyclotron_drift_animation,
    degeneracy_figure,
    landau_gauge_state_figure,
    landau_spectrum_figure,
    symmetric_gauge_state_figure,
)
from toposc_lab.visualization.integer_quantum_hall_plots import (
    integer_quantum_hall_dashboard,
    integer_quantum_hall_edge_figure,
)

__all__ = [
    "add_panel_label",
    "bose_condensation_figure",
    "bose_grand_canonical_figure",
    "bose_microcanonical_figure",
    "classical_ensemble_motion_figure",
    "cyclotron_drift_animation",
    "degeneracy_figure",
    "figure_to_bytes",
    "integer_quantum_hall_dashboard",
    "integer_quantum_hall_edge_figure",
    "matrix_values",
    "paper_style",
    "plot_eigenvalue_spectrum",
    "plot_geometry",
    "plot_localization_on_lattice",
    "plot_matrix",
    "plot_study_comparison",
    "plot_study_observable",
    "plot_study_spectrum",
    "profile_site_probabilities",
    "landau_gauge_state_figure",
    "landau_spectrum_figure",
    "quantum_gas_state_figure",
    "quantum_statistics_schematic",
    "symmetric_gauge_state_figure",
    "TopCandidateVisualization",
    "plot_top_ranked_candidates",
]
