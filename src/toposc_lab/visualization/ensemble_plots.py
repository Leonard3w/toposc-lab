"""Ensemble-specific visualisations with explicit physical interpretation."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from toposc_lab.gases.ensembles import (
    BoseGrandCanonicalResult,
    BoseMicrocanonicalResult,
    ClassicalEnsembleResult,
    ClassicalParticleSample,
    bose_grand_canonical_momentum_slice,
)
from toposc_lab.gases.ideal_quantum_gases import BOLTZMANN_CONSTANT
from toposc_lab.visualization.quantum_gas_plots import MODE_OCCUPATION_LOG_RANGE


_OCCUPATION_COLORSCALE = [
    [0.0, "#0b1020"],
    [0.20, "#243b6b"],
    [0.45, "#7b3f98"],
    [0.70, "#d95f6f"],
    [1.0, "#fff3a1"],
]


def _occupation_colour_limits(
    occupations: np.ndarray,
    *,
    scale_mode: str,
) -> tuple[float, float, str]:
    """Return an explicit, readable colour normalization for state occupations."""
    if scale_mode == "fixed":
        return (*MODE_OCCUPATION_LOG_RANGE, "locked absolute scale")
    logarithms = np.log10(np.clip(occupations.reshape(-1), 1.0e-12, 1.0e12))
    lower = float(np.floor(np.percentile(logarithms, 2.0)))
    upper = float(np.ceil(np.percentile(logarithms, 98.0)))
    if upper - lower < 2.0:
        midpoint = 0.5 * (lower + upper)
        lower = np.floor(midpoint - 1.0)
        upper = np.ceil(midpoint + 1.0)
    return lower, upper, "automatic contrast"


def classical_ensemble_motion_figure(
    result: ClassicalEnsembleResult,
    sample: ClassicalParticleSample,
    *,
    duration_ms: float = 5.0,
    n_frames: int = 31,
) -> go.Figure:
    """Animate a physically valid ballistic classical ideal-gas sample.

    The points are a two-dimensional projection of three-dimensional particles
    in a periodic cubic box.  There are no collisions because this is an ideal
    gas; each trajectory is exactly a straight line, wrapped at the boundary.
    """
    if duration_ms <= 0.0:
        raise ValueError("duration_ms must be positive")
    if n_frames < 2:
        raise ValueError("n_frames must be at least two")
    times_ms = np.linspace(0.0, duration_ms, n_frames)
    positions = sample.positions
    velocities = sample.velocities
    box_length_um = result.box_length * 1.0e6
    frame_positions = [
        (positions + velocities * time_ms * 1.0e-3) % result.box_length
        for time_ms in times_ms
    ]
    speed_mm_per_s = np.linalg.norm(velocities, axis=1) * 1.0e3
    thermal_speed_mm_per_s = np.sqrt(
        BOLTZMANN_CONSTANT * result.temperature / result.particle_mass_kg
    ) * 1.0e3
    speed_axis = np.linspace(0.0, max(4.5 * thermal_speed_mm_per_s, np.max(speed_mm_per_s) * 1.1), 300)
    maxwell_density = (
        np.sqrt(2.0 / np.pi)
        * speed_axis**2
        / thermal_speed_mm_per_s**3
        * np.exp(-speed_axis**2 / (2.0 * thermal_speed_mm_per_s**2))
    )

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "(a) Real-space motion: x-y projection",
            "(b) Speed distribution of visible sample",
        ),
        horizontal_spacing=0.18,
    )
    figure.add_trace(
        go.Scatter(
            x=frame_positions[0][:, 0] * 1.0e6,
            y=frame_positions[0][:, 1] * 1.0e6,
            mode="markers",
            marker={"size": 7, "color": "#0072B2", "opacity": 0.76},
            name="Representative particles",
            hovertemplate="x=%{x:.3f} um<br>y=%{y:.3f} um<extra>particle</extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Histogram(
            x=speed_mm_per_s,
            histnorm="probability density",
            nbinsx=25,
            marker_color="#56B4E9",
            opacity=0.75,
            name="Visible sample",
            hovertemplate="speed=%{x:.3g} mm/s<br>density=%{y:.3g}<extra>sample</extra>",
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Scatter(
            x=speed_axis,
            y=maxwell_density,
            mode="lines",
            line={"color": "#CC3311", "width": 3},
            name="Maxwell speed law",
            hovertemplate="speed=%{x:.3g} mm/s<br>density=%{y:.3g}<extra>theory</extra>",
        ),
        row=1,
        col=2,
    )
    figure.frames = [
        go.Frame(
            name=str(index),
            data=[
                go.Scatter(
                    x=frame_position[:, 0] * 1.0e6,
                    y=frame_position[:, 1] * 1.0e6,
                    mode="markers",
                    marker={"size": 7, "color": "#0072B2", "opacity": 0.76},
                )
            ],
            traces=[0],
        )
        for index, frame_position in enumerate(frame_positions)
    ]
    figure.update_layout(
        height=505,
        margin={"l": 75, "r": 40, "t": 75, "b": 105},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"size": 14, "color": "#202020"},
        legend={"orientation": "h", "y": -0.22, "x": 0.2},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": -0.34,
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play motion",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Time = ", "suffix": " ms"},
                "pad": {"t": 30},
                "steps": [
                    {
                        "args": [[str(index)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        "label": f"{time_ms:.1f}",
                        "method": "animate",
                    }
                    for index, time_ms in enumerate(times_ms)
                ],
            }
        ],
    )
    figure.update_xaxes(title_text="x (um)", range=[0.0, box_length_um], row=1, col=1)
    figure.update_yaxes(
        title_text="y (um)",
        range=[0.0, box_length_um],
        scaleanchor="x",
        scaleratio=1,
        row=1,
        col=1,
    )
    figure.update_xaxes(title_text="speed (mm/s)", row=1, col=2)
    figure.update_yaxes(title_text="probability density", row=1, col=2)
    return figure


def bose_grand_canonical_figure(
    result: BoseGrandCanonicalResult,
    *,
    maximum_mode_index: int,
    scale_mode: str = "auto",
) -> go.Figure:
    """Plot a normal-phase grand-canonical Bose gas with explicit scaling.

    ``auto`` makes one calculation readable.  ``fixed`` locks the absolute
    log-occupation range for a parameter-to-parameter comparison.
    """
    n_x, n_y, reduced_energies, occupations = bose_grand_canonical_momentum_slice(
        result,
        maximum_mode_index=maximum_mode_index,
    )
    labels = n_x[0, :]
    flat_energy = reduced_energies.reshape(-1)
    flat_occupation = occupations.reshape(-1)
    order = np.argsort(flat_energy)
    color_min, color_max, scale_label = _occupation_colour_limits(
        occupations,
        scale_mode=scale_mode,
    )
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "(a) Grand-canonical Bose occupation: n_z = 0 slice",
            "(b) Bose occupation versus reduced energy",
        ),
        horizontal_spacing=0.22,
    )
    figure.add_trace(
        go.Heatmap(
            x=labels,
            y=labels,
            z=np.log10(np.clip(occupations, 1.0e-12, 1.0e12)),
            customdata=np.dstack((reduced_energies, occupations)),
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
            name="Bose state occupation",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=flat_energy[order],
            y=flat_occupation[order],
            mode="markers",
            marker={"size": 6, "color": "#CC3311", "opacity": 0.76},
            name="n_k",
            hovertemplate="epsilon/(k_B T)=%{x:.4g}<br>n_k=%{y:.4g}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    figure.update_layout(
        height=500,
        margin={"l": 75, "r": 150, "t": 75, "b": 80},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"size": 14, "color": "#202020"},
    )
    figure.update_xaxes(title_text="momentum quantum number n_x", row=1, col=1)
    figure.update_yaxes(title_text="momentum quantum number n_y", row=1, col=1)
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


def bose_microcanonical_figure(result: BoseMicrocanonicalResult) -> go.Figure:
    """Show exact Fock-state-averaged occupations at fixed N and energy."""
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "(a) Exact mean occupation of each finite-box mode",
            "(b) Occupation versus mode energy",
        ),
        horizontal_spacing=0.18,
    )
    figure.add_trace(
        go.Bar(
            x=result.mode_quantum_numbers,
            y=result.mean_occupations,
            marker={"color": result.mode_energy_quanta, "colorscale": "Viridis", "showscale": False},
            name="Mean occupation",
            hovertemplate="mode n=%{x}<br>mean n_n=%{y:.4g}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=result.mode_energy_quanta,
            y=result.mean_occupations,
            mode="markers",
            marker={"size": 11, "color": "#CC3311"},
            name="Exact microcanonical average",
            hovertemplate="energy/epsilon_1=%{x}<br>mean n_n=%{y:.4g}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    figure.update_layout(
        height=430,
        margin={"l": 75, "r": 40, "t": 70, "b": 75},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"size": 14, "color": "#202020"},
    )
    figure.update_xaxes(title_text="momentum quantum number n", row=1, col=1)
    figure.update_yaxes(title_text="mean occupation <n_n>", row=1, col=1)
    figure.update_xaxes(title_text="mode energy / epsilon_1", row=1, col=2)
    figure.update_yaxes(title_text="mean occupation <n_n>", row=1, col=2)
    return figure
