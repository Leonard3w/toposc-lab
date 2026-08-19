"""Visual learning material for Bose--Einstein statistics.

The purpose of this figure is deliberately not to animate free particles in
real space.  A homogeneous *ideal* gas is uniform in real space, so such an
animation is visually attractive but teaches very little.  The physics that
changes is the distribution over discrete momentum states.  This module makes
that distribution, and its contrast with a classical gas, explicit.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from toposc_lab.bosons.ideal_bose_gas import (
    BOLTZMANN_CONSTANT,
    BoseClassicalComparison,
    IdealBoseEinsteinCondensationParameters,
    IdealBoseEinsteinCondensationResult,
    IdealBoseGasParameters,
    analyze_ideal_bose_einstein_condensation,
    compare_bose_to_classical,
    ideal_bec_condensate_fraction,
)


# These scales deliberately never depend on T, mu or any other user input.
# A saturated colour means that the value exceeds the displayed comparison
# range; it does not silently rescale the old picture away.
STATE_FRACTION_LOG_RANGE = (-8.0, 0.0)
BOSONIC_EXCESS_PERCENT_RANGE = (-25.0, 25.0)


def _energy_shells(
    comparison: BoseClassicalComparison,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Group degenerate momentum states into energy shells.

    A shell contains all states with equal kinetic energy.  Dividing its
    occupation by its degeneracy answers the useful question: how attractive
    is each *individual* state at that energy?
    """
    energies = comparison.bose.energies.reshape(-1)
    bose_occupations = comparison.bose.occupations.reshape(-1)
    classical_occupations = comparison.classical_occupations.reshape(-1)
    shell_energies, shell_indices = np.unique(energies, return_inverse=True)
    degeneracies = np.bincount(shell_indices)
    bose_per_shell = np.bincount(shell_indices, weights=bose_occupations)
    classical_per_shell = np.bincount(
        shell_indices,
        weights=classical_occupations,
    )
    return (
        shell_energies,
        bose_per_shell / degeneracies,
        classical_per_shell / degeneracies,
        degeneracies,
    )


def bose_statistics_learning_figure(
    parameters: IdealBoseGasParameters,
) -> tuple[BoseClassicalComparison, go.Figure]:
    """Return a direct Bose-versus-classical visual comparison.

    Both distributions contain exactly the same mean number of particles.
    Therefore the bright low-momentum centre and the red/blue difference map
    are caused by the statistical rule, rather than by a different amount of
    gas.
    """
    comparison = compare_bose_to_classical(parameters)
    bose = comparison.bose
    total_population = comparison.expected_particle_number
    bose_fraction = bose.occupations / total_population
    classical_fraction = comparison.classical_occupations / total_population
    excess_fraction = bose_fraction - classical_fraction

    quantum_numbers = bose.quantum_numbers_x[0, :]
    common_custom_data = np.dstack(
        (
            bose.energies / BOLTZMANN_CONSTANT / 1.0e-9,
            bose.occupations,
            comparison.classical_occupations,
            bose_fraction,
            classical_fraction,
        )
    )
    hover_template = (
        "n_x=%{x}, n_y=%{y}<br>"
        "epsilon/k_B=%{customdata[0]:.3g} nK<br>"
        "Bose n_k=%{customdata[1]:.3g}<br>"
        "Classical n_k=%{customdata[2]:.3g}<br>"
        "Bose fraction=%{customdata[3]:.2%}<br>"
        "Classical fraction=%{customdata[4]:.2%}<extra></extra>"
    )

    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "(a) Bose occupation of momentum states",
            "(b) Classical reference, same mean particle number",
            "(c) Bosonic excess in each state",
            "(d) Occupation of one state versus energy",
        ),
        horizontal_spacing=0.13,
        vertical_spacing=0.16,
    )
    figure.add_trace(
        go.Heatmap(
            x=quantum_numbers,
            y=quantum_numbers,
            z=np.log10(bose_fraction),
            customdata=common_custom_data,
            colorscale="Magma",
            zmin=STATE_FRACTION_LOG_RANGE[0],
            zmax=STATE_FRACTION_LOG_RANGE[1],
            showscale=False,
            hovertemplate=hover_template,
            name="Bose state fraction",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Heatmap(
            x=quantum_numbers,
            y=quantum_numbers,
            z=np.log10(classical_fraction),
            customdata=common_custom_data,
            colorscale="Magma",
            zmin=STATE_FRACTION_LOG_RANGE[0],
            zmax=STATE_FRACTION_LOG_RANGE[1],
            colorbar={
                "title": {"text": "log10(state fraction)<br>(fixed scale)"},
                "x": 1.02,
                "y": 0.79,
                "len": 0.38,
            },
            hovertemplate=hover_template,
            name="Classical state fraction",
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Heatmap(
            x=quantum_numbers,
            y=quantum_numbers,
            z=100.0 * excess_fraction,
            customdata=common_custom_data,
            colorscale="RdBu_r",
            zmin=BOSONIC_EXCESS_PERCENT_RANGE[0],
            zmax=BOSONIC_EXCESS_PERCENT_RANGE[1],
            colorbar={
                "title": {"text": "Bose - classical<br>(percentage points, fixed)"},
                "x": 1.02,
                "y": 0.28,
                "len": 0.38,
            },
            hovertemplate=hover_template,
            name="Bosonic excess",
        ),
        row=2,
        col=1,
    )

    shell_energies, bose_per_state, classical_per_state, degeneracies = _energy_shells(
        comparison
    )
    energy_nk = shell_energies / BOLTZMANN_CONSTANT / 1.0e-9
    figure.add_trace(
        go.Scatter(
            x=energy_nk,
            y=bose_per_state,
            mode="lines+markers",
            line={"color": "#CC3311", "width": 3},
            marker={"size": 7},
            customdata=degeneracies,
            name="Bose--Einstein",
            hovertemplate=(
                "epsilon/k_B=%{x:.3g} nK<br>"
                "mean n per state=%{y:.3g}<br>"
                "degeneracy=%{customdata}<extra>Bose</extra>"
            ),
        ),
        row=2,
        col=2,
    )
    figure.add_trace(
        go.Scatter(
            x=energy_nk,
            y=classical_per_state,
            mode="lines+markers",
            line={"color": "#0072B2", "width": 3, "dash": "dash"},
            marker={"size": 7, "symbol": "diamond"},
            customdata=degeneracies,
            name="Maxwell--Boltzmann",
            hovertemplate=(
                "epsilon/k_B=%{x:.3g} nK<br>"
                "mean n per state=%{y:.3g}<br>"
                "degeneracy=%{customdata}<extra>classical</extra>"
            ),
        ),
        row=2,
        col=2,
    )

    figure.update_layout(
        height=830,
        margin={"l": 100, "r": 210, "t": 90, "b": 105},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"size": 14, "color": "#202020"},
        legend={"orientation": "h", "y": -0.16, "x": 0.56},
    )
    for row, column in ((1, 1), (1, 2), (2, 1)):
        figure.update_xaxes(
            title_text="n_x",
            tickmode="linear",
            dtick=max(1, parameters.maximum_mode_index // 4),
            row=row,
            col=column,
        )
        figure.update_yaxes(
            title_text="n_y",
            tickmode="linear",
            dtick=max(1, parameters.maximum_mode_index // 4),
            row=row,
            col=column,
        )
    figure.update_xaxes(title_text="energy epsilon/k_B (nK)", row=2, col=2)
    figure.update_yaxes(
        title_text="mean occupation n per state",
        type="log",
        dtick=1,
        tickformat="~g",
        row=2,
        col=2,
    )
    return comparison, figure


def ideal_bec_learning_figure(
    parameters: IdealBoseEinsteinCondensationParameters,
) -> tuple[IdealBoseEinsteinCondensationResult, go.Figure]:
    """Return a clear fixed-N 3D ideal-gas BEC phase-diagram material.

    The left panel contains the standard analytic law
    ``N_0/N = 1 - (T/T_c)^(3/2)`` below the transition.  The right panel turns
    the current point on that curve into particle numbers, so a user can see
    the difference between a condensate (one k=0 state) and the thermal cloud
    (all excited momentum states).
    """
    result = analyze_ideal_bose_einstein_condensation(parameters)
    critical_temperature_nk = result.critical_temperature / 1.0e-9
    current_temperature_nk = parameters.temperature / 1.0e-9
    maximum_temperature_nk = max(1.5 * critical_temperature_nk, 1.15 * current_temperature_nk)
    temperature_axis_nk = np.linspace(0.0, maximum_temperature_nk, 301)
    condensate_fraction_axis = np.maximum(
        0.0,
        1.0 - (temperature_axis_nk / critical_temperature_nk) ** 1.5,
    )

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "(a) Ideal-gas BEC transition in three dimensions",
            "(b) Population at the selected temperature",
        ),
        horizontal_spacing=0.18,
    )
    figure.add_trace(
        go.Scatter(
            x=temperature_axis_nk,
            y=condensate_fraction_axis,
            mode="lines",
            line={"color": "#CC3311", "width": 4},
            name="N0 / N",
            hovertemplate="T=%{x:.2f} nK<br>N0/N=%{y:.2%}<extra>ideal 3D Bose gas</extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_vline(
        x=critical_temperature_nk,
        line_dash="dash",
        line_color="#555555",
        annotation_text="T_c",
        annotation_position="top right",
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=[current_temperature_nk],
            y=[result.condensate_fraction],
            mode="markers",
            marker={"size": 14, "color": "#0072B2", "line": {"color": "white", "width": 1}},
            name="Current conditions",
            hovertemplate="T=%{x:.2f} nK<br>N0/N=%{y:.2%}<extra>current point</extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=["Selected gas"],
            y=[result.condensate_number],
            name="Condensate: k = 0",
            marker_color="#CC3311",
            text=[f"{result.condensate_number:,.0f}"],
            textposition="inside",
            hovertemplate="N0=%{y:,.0f}<extra>k = 0 condensate</extra>",
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Bar(
            x=["Selected gas"],
            y=[result.thermal_number],
            name="Thermal cloud: k != 0",
            marker_color="#56B4E9",
            text=[f"{result.thermal_number:,.0f}"],
            textposition="inside",
            hovertemplate="Nthermal=%{y:,.0f}<extra>excited states</extra>",
        ),
        row=1,
        col=2,
    )
    figure.update_layout(
        barmode="stack",
        height=500,
        margin={"l": 75, "r": 45, "t": 80, "b": 75},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"size": 14, "color": "#202020"},
        legend={"orientation": "h", "y": -0.2, "x": 0.23},
    )
    figure.update_xaxes(title_text="temperature T (nK)", row=1, col=1)
    figure.update_yaxes(
        title_text="condensate fraction N0 / N",
        range=[0.0, 1.05],
        tickformat=".0%",
        row=1,
        col=1,
    )
    figure.update_yaxes(title_text="particle number", row=1, col=2)
    return result, figure


def ideal_bose_gas_live_material(
    parameters: IdealBoseGasParameters,
) -> tuple[BoseClassicalComparison, go.Figure]:
    """Backward-compatible entry point for the statistics learning material.

    The former moving-particle view was replaced because it hid the central
    physics.  The figure now makes the Bose distribution directly comparable
    to the equal-population classical distribution.
    """
    return bose_statistics_learning_figure(parameters)
