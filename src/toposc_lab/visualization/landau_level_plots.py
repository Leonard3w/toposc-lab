"""Publication-style and animated plots for the Landau-level learning lab."""

from __future__ import annotations

from dataclasses import replace
from math import ceil, sqrt

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from toposc_lab.quantum_hall.landau_levels import (
    BOHR_MAGNETON,
    ELEMENTARY_CHARGE,
    HBAR,
    LandauLevelParameters,
    cyclotron_trajectory,
    electric_field_energies_joule,
    landau_gauge_state,
    landau_scales,
    symmetric_gauge_lll_state,
)
from toposc_lab.visualization.style import add_panel_label, paper_style


JOULE_TO_MEV = 1.0e3 / ELEMENTARY_CHARGE


def _energy_mev(values: np.ndarray | float) -> np.ndarray:
    return np.asarray(values, dtype=float) * JOULE_TO_MEV


def landau_spectrum_figure(
    parameters: LandauLevelParameters,
) -> tuple[plt.Figure, np.ndarray]:
    """Show field-dependent levels and their E-field dispersion in ``k l_B``."""
    levels = np.arange(parameters.maximum_level + 1)
    magnetic_field = parameters.magnetic_field_tesla
    field_values = np.linspace(
        max(0.01, 0.05 * magnetic_field),
        max(1.8 * magnetic_field, magnetic_field + 0.5),
        240,
    )
    k_l_b = np.linspace(-4.0, 4.0, 241)
    mass = parameters.effective_mass_kg

    with paper_style():
        figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), constrained_layout=True)

        for level in levels:
            base = HBAR * ELEMENTARY_CHARGE * field_values / mass * (level + 0.5)
            (line,) = axes[0].plot(
                field_values,
                _energy_mev(base),
                label=f"n={level}",
            )
            if parameters.include_zeeman:
                zeeman = 0.5 * parameters.g_factor * BOHR_MAGNETON * field_values
                axes[0].plot(
                    field_values,
                    _energy_mev(base + zeeman),
                    color=line.get_color(),
                    linestyle="--",
                    linewidth=0.9,
                    alpha=0.75,
                )
                axes[0].plot(
                    field_values,
                    _energy_mev(base - zeeman),
                    color=line.get_color(),
                    linestyle=":",
                    linewidth=0.9,
                    alpha=0.75,
                )

        axes[0].axvline(magnetic_field, color="#172554", linestyle="--", linewidth=1.0)
        axes[0].set(
            xlabel="Magnetfeld B (T)",
            ylabel="Energie (meV)",
            title="Gleichmäßig quantisierte Energien",
        )
        axes[0].legend(ncols=2, loc="upper left")

        spectra = electric_field_energies_joule(parameters, k_l_b, levels)
        for index, level in enumerate(levels):
            axes[1].plot(k_l_b, _energy_mev(spectra[index]), label=f"n={level}")
            if parameters.include_zeeman:
                for spin, linestyle in ((1, "--"), (-1, ":")):
                    spin_spectrum = electric_field_energies_joule(
                        parameters,
                        k_l_b,
                        np.asarray([level]),
                        spin_projection=spin,
                    )[0]
                    axes[1].plot(
                        k_l_b,
                        _energy_mev(spin_spectrum),
                        color=axes[1].lines[-1].get_color(),
                        linestyle=linestyle,
                        linewidth=0.8,
                        alpha=0.7,
                    )

        axes[1].axvline(
            parameters.wave_number_l_b,
            color="#172554",
            linestyle="--",
            linewidth=1.0,
        )
        axes[1].set(
            xlabel=r"dimensionslose Wellenzahl $k\,l_B$",
            ylabel="Energie (meV)",
            title=rf"Dispersion bei $E_x={parameters.electric_field_v_per_m:g}$ V/m",
        )
        axes[1].legend(ncols=2, loc="upper left")
        add_panel_label(axes[0], "(a)")
        add_panel_label(axes[1], "(b)")

    return figure, spectra


def landau_gauge_state_figure(
    parameters: LandauLevelParameters,
    *,
    include_electric_shift: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """Visualize a strip-like ``psi_nk`` state and its transverse density."""
    state = landau_gauge_state(
        parameters,
        include_electric_shift=include_electric_shift,
    )
    extent = (
        state.x_over_l_b[0],
        state.x_over_l_b[-1],
        state.y_over_l_b[0],
        state.y_over_l_b[-1],
    )

    with paper_style():
        figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.35), constrained_layout=True)
        image = axes[0].imshow(
            state.probability_map,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="magma",
            interpolation="bilinear",
        )
        axes[0].axvline(state.center_over_l_b, color="white", linestyle="--", linewidth=1.1)
        axes[0].set(
            xlabel=r"$x/l_B$",
            ylabel=r"$y/l_B$",
            title=rf"Landau-Gauge: $n={parameters.selected_level}$, $k\,l_B={parameters.wave_number_l_b:g}$",
        )
        figure.colorbar(
            image,
            ax=axes[0],
            label=r"$l_B |f_n(x)|^2$",
            fraction=0.048,
            pad=0.04,
        )

        axes[1].plot(
            state.x_over_l_b,
            state.transverse_probability_density,
            color="#0072B2",
        )
        axes[1].fill_between(
            state.x_over_l_b,
            state.transverse_probability_density,
            color="#56B4E9",
            alpha=0.3,
        )
        axes[1].axvline(
            state.center_over_l_b,
            color="#D55E00",
            linestyle="--",
            label=rf"Zentrum $x_c/l_B={state.center_over_l_b:.3f}$",
        )
        axes[1].set(
            xlabel=r"$x/l_B$",
            ylabel=r"$l_B |f_n(x)|^2$",
            title="Transversale Wahrscheinlichkeitsdichte",
        )
        axes[1].legend()
        add_panel_label(axes[0], "(a)")
        add_panel_label(axes[1], "(b)")

    return figure, state.probability_map


def symmetric_gauge_state_figure(
    parameters: LandauLevelParameters,
) -> tuple[plt.Figure, np.ndarray]:
    """Show the selected symmetric-gauge LLL ring and its radial density."""
    state = symmetric_gauge_lll_state(parameters)
    extent = (
        state.x_over_l_b[0],
        state.x_over_l_b[-1],
        state.y_over_l_b[0],
        state.y_over_l_b[-1],
    )

    with paper_style():
        figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.35), constrained_layout=True)
        image = axes[0].imshow(
            state.probability_density_l_b2,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="viridis",
            interpolation="bilinear",
        )
        if state.peak_radius_over_l_b > 0.0:
            peak_circle = plt.Circle(
                (0.0, 0.0),
                state.peak_radius_over_l_b,
                fill=False,
                color="white",
                linestyle="--",
                linewidth=1.1,
            )
            axes[0].add_patch(peak_circle)
        axes[0].set(
            xlabel=r"$x/l_B$",
            ylabel=r"$y/l_B$",
            title=rf"LLL-Zustand $|0,m={parameters.angular_momentum}\rangle$",
        )
        figure.colorbar(
            image,
            ax=axes[0],
            label=r"$l_B^2 |\psi_{0,m}|^2$",
            fraction=0.048,
            pad=0.04,
        )

        comparison_m = sorted(
            {
                0,
                max(0, parameters.angular_momentum - 1),
                parameters.angular_momentum,
                parameters.angular_momentum + 1,
            }
        )
        for angular_momentum in comparison_m:
            comparison = symmetric_gauge_lll_state(
                replace(parameters, angular_momentum=angular_momentum)
            )
            selected = angular_momentum == parameters.angular_momentum
            axes[1].plot(
                comparison.radius_over_l_b,
                comparison.radial_density_l_b2,
                linewidth=2.2 if selected else 1.1,
                alpha=1.0 if selected else 0.6,
                label=f"m={angular_momentum}",
            )
        axes[1].axvline(
            state.peak_radius_over_l_b,
            color="#172554",
            linestyle="--",
            linewidth=1.0,
            label=rf"$\sqrt{{2m}}={state.peak_radius_over_l_b:.2f}$",
        )
        axes[1].set(
            xlabel=r"Radius $r/l_B$",
            ylabel=r"$l_B^2 |\psi_{0,m}(r)|^2$",
            title="Ringradius wächst mit dem Drehimpuls",
        )
        axes[1].legend()
        add_panel_label(axes[0], "(a)")
        add_panel_label(axes[1], "(b)")

    return figure, state.probability_density_l_b2


def degeneracy_figure(
    parameters: LandauLevelParameters,
) -> tuple[plt.Figure, np.ndarray]:
    """Show representative orbit centres and the flux-controlled degeneracy."""
    scales = landau_scales(parameters)
    width_um = parameters.sample_width_m * 1.0e6
    height_um = parameters.sample_height_m * 1.0e6
    requested_points = max(1, int(round(scales.flux_quanta)))
    displayed_points = min(requested_points, 500)
    aspect_ratio = max(width_um / height_um, 1.0e-6)
    columns = max(1, int(sqrt(displayed_points * aspect_ratio)))
    rows = max(1, int(ceil(displayed_points / columns)))
    x_values = (np.arange(columns) + 0.5) * width_um / columns
    y_values = (np.arange(rows) + 0.5) * height_um / rows
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    centers = np.column_stack((x_grid.ravel(), y_grid.ravel()))[:displayed_points]

    field_values = np.linspace(
        0.0,
        max(1.8 * parameters.magnetic_field_tesla, parameters.magnetic_field_tesla + 0.5),
        241,
    )
    degeneracy_values = field_values * parameters.sample_area_m2 / scales.flux_quantum_weber

    with paper_style():
        figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.25), constrained_layout=True)
        axes[0].scatter(
            centers[:, 0],
            centers[:, 1],
            s=14,
            color="#0072B2",
            alpha=0.75,
            edgecolors="none",
        )
        axes[0].set(
            xlim=(0.0, width_um),
            ylim=(0.0, height_um),
            xlabel=r"$x$ ($\mu$m)",
            ylabel=r"$y$ ($\mu$m)",
            title=f"Repräsentative Leitzentren ({displayed_points:,} gezeigt)",
        )
        axes[0].set_aspect("equal", adjustable="box")

        axes[1].plot(field_values, degeneracy_values, color="#009E73")
        axes[1].scatter(
            [parameters.magnetic_field_tesla],
            [scales.flux_quanta],
            color="#D55E00",
            zorder=3,
            label=rf"aktuell: $N_\Phi={scales.flux_quanta:.3g}$",
        )
        axes[1].set(
            xlabel="Magnetfeld B (T)",
            ylabel=r"Flussquanten $N_\Phi=BA/\Phi_0$",
            title="Ein Zustand pro Flussquant",
        )
        axes[1].legend()
        add_panel_label(axes[0], "(a)")
        add_panel_label(axes[1], "(b)")

    return figure, centers


def cyclotron_drift_animation(
    parameters: LandauLevelParameters,
    *,
    samples: int = 121,
) -> go.Figure:
    """Build a playable E-cross-B cyclotron trajectory."""
    trajectory = cyclotron_trajectory(parameters, samples=samples)
    x_values = trajectory.x_over_l_b
    y_values = trajectory.y_over_l_b
    center_x = trajectory.guiding_center_x_over_l_b
    center_y = trajectory.guiding_center_y_over_l_b
    padding = max(0.6, 0.12 * max(np.ptp(x_values), np.ptp(y_values), 1.0))

    figure = go.Figure(
        data=(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                line={"color": "#9fb3c8", "width": 2},
                name="vollständige Bahn",
                hoverinfo="skip",
            ),
            go.Scatter(
                x=[x_values[0]],
                y=[y_values[0]],
                mode="markers",
                marker={"size": 13, "color": "#D55E00"},
                name="Teilchen",
            ),
            go.Scatter(
                x=[center_x[0]],
                y=[center_y[0]],
                mode="markers",
                marker={"size": 10, "color": "#0072B2", "symbol": "x"},
                name="Leitzentrum",
            ),
        ),
        frames=tuple(
            go.Frame(
                name=str(index),
                data=(
                    go.Scatter(x=[x_values[index]], y=[y_values[index]]),
                    go.Scatter(x=[center_x[index]], y=[center_y[index]]),
                ),
                traces=(1, 2),
            )
            for index in range(samples)
        ),
    )
    figure.update_layout(
        template="plotly_white",
        title=(
            "Zyklotronbewegung und E×B-Drift "
            f"(v_y = {landau_scales(parameters).drift_velocity_m_s:.3g} m/s)"
        ),
        xaxis={
            "title": "x / l_B",
            "range": [float(np.min(x_values) - padding), float(np.max(x_values) + padding)],
            "scaleanchor": "y",
            "scaleratio": 1,
        },
        yaxis={
            "title": "y / l_B",
            "range": [float(np.min(y_values) - padding), float(np.max(y_values) + padding)],
        },
        legend={"orientation": "h", "y": 1.02, "x": 0.0},
        margin={"l": 55, "r": 25, "t": 85, "b": 55},
        updatemenus=(
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.0,
                "y": -0.12,
                "buttons": (
                    {
                        "label": "▶ Start",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 45, "redraw": False},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "❚❚ Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                            },
                        ],
                    },
                ),
            },
        ),
        sliders=(
            {
                "active": 0,
                "x": 0.28,
                "len": 0.68,
                "y": -0.08,
                "currentvalue": {"prefix": "Phase: "},
                "steps": tuple(
                    {
                        "label": f"{phase / (2.0 * np.pi):.1f} T",
                        "method": "animate",
                        "args": [
                            [str(index)],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                    for index, phase in enumerate(trajectory.phase_radians)
                ),
            },
        ),
    )
    return figure


__all__ = [
    "JOULE_TO_MEV",
    "cyclotron_drift_animation",
    "degeneracy_figure",
    "landau_gauge_state_figure",
    "landau_spectrum_figure",
    "symmetric_gauge_state_figure",
]
