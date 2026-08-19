"""Publication-style, fixed-scale figures for the ideal quantum-gas laboratory."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from toposc_lab.gases.ideal_quantum_gases import (
    BOLTZMANN_CONSTANT,
    QuantumGasResult,
    momentum_state_slice,
)


# Fixed across every gas and every parameter setting.  Colours therefore have
# a stable physical meaning: log10 of the mean occupation of one momentum mode.
MODE_OCCUPATION_LOG_RANGE = (-6.0, 6.0)

_OCCUPATION_COLORSCALE = [
    [0.0, "#0b1020"],
    [0.20, "#243b6b"],
    [0.45, "#7b3f98"],
    [0.70, "#d95f6f"],
    [1.0, "#fff3a1"],
]

_GAS_STYLE = {
    "classical": {"colour": "#666666", "name": "Maxwell--Boltzmann"},
    "boson": {"colour": "#CC3311", "name": "Bose--Einstein"},
}


def quantum_gas_state_figure(
    result: QuantumGasResult,
    *,
    maximum_mode_index: int,
    scale_mode: str = "auto",
) -> go.Figure:
    """Plot a momentum slice with either readable or comparison-locked contrast."""
    n_x, n_y, reduced_energies, occupations = momentum_state_slice(
        result,
        maximum_mode_index=maximum_mode_index,
    )
    flat_energies = reduced_energies.reshape(-1)
    flat_occupations = occupations.reshape(-1)
    order = np.argsort(flat_energies)
    style = _GAS_STYLE[result.statistic]
    logarithmic_occupations = np.log10(np.clip(occupations, 1.0e-12, 1.0e12))
    if scale_mode == "fixed":
        color_min, color_max = MODE_OCCUPATION_LOG_RANGE
        scale_label = "locked absolute scale"
    else:
        flat_logarithms = logarithmic_occupations.reshape(-1)
        color_min = float(np.floor(np.percentile(flat_logarithms, 2.0)))
        color_max = float(np.ceil(np.percentile(flat_logarithms, 98.0)))
        if color_max - color_min < 2.0:
            middle = 0.5 * (color_min + color_max)
            color_min = float(np.floor(middle - 1.0))
            color_max = float(np.ceil(middle + 1.0))
        scale_label = "automatic contrast"
    labels = n_x[0, :]
    custom_data = np.dstack((reduced_energies, occupations))

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "(a) Momentum-state occupation: n_z = 0 slice",
            "(b) One-state occupation law",
        ),
        horizontal_spacing=0.22,
    )
    figure.add_trace(
        go.Heatmap(
            x=labels,
            y=labels,
            z=logarithmic_occupations,
            customdata=custom_data,
            colorscale=_OCCUPATION_COLORSCALE,
            zmin=color_min,
            zmax=color_max,
            colorbar={
                "title": {"text": f"log10(mean occupation)<br>({scale_label})"},
                "x": 1.02,
                "y": 0.5,
                "len": 0.82,
            },
            hovertemplate=(
                "n_x=%{x}, n_y=%{y}, n_z=0<br>"
                "epsilon/(k_B T)=%{customdata[0]:.4g}<br>"
                "mean occupation=%{customdata[1]:.4g}<extra></extra>"
            ),
            name="Momentum states",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=flat_energies[order],
            y=flat_occupations[order],
            mode="markers",
            marker={"size": 6, "color": style["colour"], "opacity": 0.75},
            name=style["name"],
            hovertemplate=(
                "epsilon/(k_B T)=%{x:.4g}<br>"
                "mean occupation=%{y:.4g}<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )
    if result.statistic == "boson" and result.condensate_fraction > 0.0:
        figure.add_annotation(
            x=0.02,
            y=0.96,
            xref="x2 domain",
            yref="y2 domain",
            text=(
                f"Condensate at k=0: {result.condensate_fraction:.1%} of N"
            ),
            showarrow=False,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#CC3311",
            borderwidth=1,
            xanchor="left",
            yanchor="top",
        )

    figure.update_layout(
        height=520,
        margin={"l": 80, "r": 150, "t": 75, "b": 85},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"size": 14, "color": "#202020"},
        legend={"orientation": "h", "y": -0.22, "x": 0.66},
    )
    figure.update_xaxes(
        title_text="momentum quantum number n_x",
        tickmode="linear",
        dtick=max(1, maximum_mode_index // 4),
        row=1,
        col=1,
    )
    figure.update_yaxes(
        title_text="momentum quantum number n_y",
        tickmode="linear",
        dtick=max(1, maximum_mode_index // 4),
        row=1,
        col=1,
    )
    figure.update_xaxes(title_text="reduced energy epsilon/(k_B T)", row=1, col=2)
    figure.update_yaxes(
        title_text="mean occupation of one state",
        type="log",
        range=[color_min, color_max],
        dtick=2,
        tickformat="~g",
        row=1,
        col=2,
    )
    return figure


def bose_condensation_figure(result: QuantumGasResult) -> go.Figure:
    """Plot the 3D ideal-gas BEC fraction and current population split."""
    if result.statistic != "boson" or result.critical_temperature is None:
        raise ValueError("bose_condensation_figure requires a Bose-gas result")

    reduced_temperature = result.parameters.temperature / result.critical_temperature
    reduced_axis = np.linspace(0.0, max(1.5, 1.15 * reduced_temperature), 301)
    condensate_fraction = np.maximum(0.0, 1.0 - reduced_axis**1.5)
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "(c) Condensate fraction versus reduced temperature",
            "(d) Current population accounting",
        ),
        horizontal_spacing=0.2,
    )
    figure.add_trace(
        go.Scatter(
            x=reduced_axis,
            y=condensate_fraction,
            mode="lines",
            line={"color": "#CC3311", "width": 4},
            name="N0 / N",
            hovertemplate="T/Tc=%{x:.3f}<br>N0/N=%{y:.2%}<extra>ideal 3D Bose gas</extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_vline(
        x=1.0,
        line_dash="dash",
        line_color="#555555",
        annotation_text="T = Tc",
        annotation_position="top right",
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=[reduced_temperature],
            y=[result.condensate_fraction],
            mode="markers",
            marker={"size": 14, "color": "#0072B2", "line": {"color": "white", "width": 1}},
            name="Current gas",
            hovertemplate="T/Tc=%{x:.3f}<br>N0/N=%{y:.2%}<extra>current conditions</extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=["Current gas"],
            y=[result.condensate_particle_number],
            marker_color="#CC3311",
            name="Condensate k = 0",
            text=[f"{result.condensate_particle_number:,.0f}"],
            textposition="inside",
            hovertemplate="N0=%{y:,.0f}<extra>condensate</extra>",
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Bar(
            x=["Current gas"],
            y=[result.thermal_particle_number],
            marker_color="#56B4E9",
            name="Thermal cloud k != 0",
            text=[f"{result.thermal_particle_number:,.0f}"],
            textposition="inside",
            hovertemplate="Nthermal=%{y:,.0f}<extra>excited states</extra>",
        ),
        row=1,
        col=2,
    )
    figure.update_layout(
        barmode="stack",
        height=455,
        margin={"l": 80, "r": 45, "t": 70, "b": 85},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"size": 14, "color": "#202020"},
        legend={"orientation": "h", "y": -0.22, "x": 0.18},
    )
    figure.update_xaxes(title_text="reduced temperature T / Tc", row=1, col=1)
    figure.update_yaxes(
        title_text="condensate fraction N0 / N",
        range=[0.0, 1.05],
        tickformat=".0%",
        row=1,
        col=1,
    )
    figure.update_yaxes(title_text="particle number", row=1, col=2)
    return figure


def quantum_statistics_schematic() -> go.Figure:
    """Draw a labelled schematic of the classical and Bose counting rules.

    This is intentionally not a numerical result.  It is a compact visual key
    for reading the quantitatively calculated maps in the three calculators.
    """
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Classical: broad thermal sharing",
            "Bosons: shared low-energy state",
        ),
        horizontal_spacing=0.16,
    )
    energy_levels = np.arange(4)
    colour = {"classical": "#666666", "boson": "#CC3311"}
    for column in range(1, 3):
        for energy in energy_levels:
            figure.add_shape(
                type="line",
                x0=-0.42,
                x1=0.42,
                y0=float(energy),
                y1=float(energy),
                line={"color": "#999999", "width": 1},
                row=1,
                col=column,
            )

    # Arbitrary dots illustrate only the allowed counting rule, never a
    # calculated particle configuration.
    for energy, count in enumerate((3, 2, 2, 1)):
        figure.add_trace(
            go.Scatter(
                x=np.linspace(-0.18, 0.18, count),
                y=np.full(count, energy),
                mode="markers",
                marker={"size": 13, "color": colour["classical"]},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
    for energy, count in enumerate((7, 2, 1, 0)):
        if count:
            figure.add_trace(
                go.Scatter(
                    x=np.linspace(-0.24, 0.24, count),
                    y=np.full(count, energy),
                    mode="markers",
                    marker={"size": 13, "color": colour["boson"]},
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=2,
            )
    figure.update_layout(
        height=330,
        margin={"l": 35, "r": 30, "t": 60, "b": 30},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"size": 13, "color": "#202020"},
        annotations=[
            *figure.layout.annotations,
            {
                "text": "energy",
                "xref": "paper",
                "yref": "paper",
                "x": -0.02,
                "y": 0.5,
                "textangle": -90,
                "showarrow": False,
            },
        ],
    )
    for column in range(1, 3):
        figure.update_xaxes(visible=False, range=[-0.5, 0.5], row=1, col=column)
        figure.update_yaxes(visible=False, range=[-0.35, 3.55], row=1, col=column)
    return figure
