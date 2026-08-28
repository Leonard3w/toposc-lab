"""Model-independent visualization of discrete geometries."""

from __future__ import annotations

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from toposc_lab.geometry import Geometry, GeometryEdge
from toposc_lab.visualization.style import paper_style


def plot_geometry(
    geometry: Geometry,
    *,
    axes: Axes | None = None,
    title: str | None = None,
    show_site_indices: bool = False,
    show_boundary_sites: bool = True,
    show_periodic_edges: bool = False,
    site_size: float = 48.0,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Plot sites and edges of a model-independent geometry.

    One-dimensional coordinates are drawn on a horizontal line. Coordinates
    with two or more components use their first two components. If coordinates
    are absent, a deterministic circular layout is used. Periodic edges whose
    endpoint coordinates span a unit-cell seam are hidden by default because a
    straight line between those endpoints would be geometrically misleading.
    """
    if not np.isfinite(site_size) or site_size <= 0.0:
        raise ValueError("site_size must be finite and positive")

    positions, layout_kind = _plot_positions(geometry)
    created_axes = axes is None

    with paper_style():
        if axes is None:
            figure, axes = plt.subplots(figsize=(6.2, 5.0), constrained_layout=True)
        else:
            figure = cast(Figure, axes.figure)

        _draw_edges(
            axes,
            geometry,
            positions,
            show_periodic_edges=show_periodic_edges,
        )
        _draw_sites(axes, geometry, positions, site_size=site_size)

        if show_boundary_sites and geometry.boundary_sites:
            boundary_indices = np.asarray(sorted(geometry.boundary_sites), dtype=int)
            axes.scatter(
                positions[boundary_indices, 0],
                positions[boundary_indices, 1],
                s=1.65 * site_size,
                facecolors="none",
                edgecolors="tab:orange",
                linewidths=1.35,
                label="Boundary site",
                zorder=4,
            )

        if show_site_indices:
            for site, position in enumerate(positions):
                axes.annotate(
                    str(site),
                    position,
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    zorder=5,
                )

        axes.set_title(_plot_title(geometry) if title is None else title)
        _configure_axes(axes, layout_kind=layout_kind)

        handles, labels = axes.get_legend_handles_labels()
        if labels:
            axes.legend(handles, labels, loc="best")

    if created_axes and show:
        plt.show()

    return figure, axes


def _plot_positions(geometry: Geometry) -> tuple[np.ndarray, str]:
    if geometry.coordinates is None:
        angles = 2.0 * np.pi * np.arange(geometry.n_sites, dtype=float) / geometry.n_sites
        return np.column_stack((np.cos(angles), np.sin(angles))), "abstract"

    if geometry.dimension == 1:
        return (
            np.column_stack(
                (
                    geometry.coordinates[:, 0],
                    np.zeros(geometry.n_sites, dtype=float),
                )
            ),
            "one_dimensional",
        )

    return geometry.coordinates[:, :2].copy(), "embedded"


def _draw_edges(
    axes: Axes,
    geometry: Geometry,
    positions: np.ndarray,
    *,
    show_periodic_edges: bool,
) -> None:
    typed_edges = sorted(
        {edge.edge_type for edge in geometry.edges if edge.edge_type is not None}
    )
    color_by_type = len(typed_edges) > 1
    color_map = plt.get_cmap("tab10")
    type_colors = {
        edge_type: color_map(index % color_map.N)
        for index, edge_type in enumerate(typed_edges)
    }
    labeled_types: set[str] = set()

    for edge in geometry.edges:
        wrapped = _is_wrapped_periodic_edge(geometry, edge)
        if wrapped and not show_periodic_edges:
            continue

        source = positions[edge.source]
        target = positions[edge.target]
        label = None
        if color_by_type and edge.edge_type is not None and edge.edge_type not in labeled_types:
            label = f"Edge type: {edge.edge_type}"
            labeled_types.add(edge.edge_type)

        axes.plot(
            [source[0], target[0]],
            [source[1], target[1]],
            color=(
                type_colors[edge.edge_type]
                if color_by_type and edge.edge_type is not None
                else "0.60"
            ),
            linestyle="--" if edge.boundary_crossing else "-",
            linewidth=1.1,
            label=label,
            zorder=1,
        )


def _draw_sites(
    axes: Axes,
    geometry: Geometry,
    positions: np.ndarray,
    *,
    site_size: float,
) -> None:
    if geometry.site_types is None or len(set(geometry.site_types)) == 1:
        axes.scatter(
            positions[:, 0],
            positions[:, 1],
            s=site_size,
            color="tab:blue",
            edgecolors="white",
            linewidths=0.65,
            zorder=3,
        )
        return

    site_types = sorted(set(geometry.site_types), key=lambda value: "" if value is None else value)
    color_map = plt.get_cmap("tab10")
    for index, site_type in enumerate(site_types):
        site_indices = np.asarray(
            [site for site, value in enumerate(geometry.site_types) if value == site_type],
            dtype=int,
        )
        axes.scatter(
            positions[site_indices, 0],
            positions[site_indices, 1],
            s=site_size,
            color=color_map(index % color_map.N),
            edgecolors="white",
            linewidths=0.65,
            label=f"Site type: {site_type if site_type is not None else 'untyped'}",
            zorder=3,
        )


def _is_wrapped_periodic_edge(geometry: Geometry, edge: GeometryEdge) -> bool:
    if (
        geometry.coordinates is None
        or not edge.boundary_crossing
        or edge.displacement is None
    ):
        return False

    coordinate_displacement = geometry.coordinates[edge.target] - geometry.coordinates[edge.source]
    return not np.allclose(coordinate_displacement, edge.displacement)


def _configure_axes(axes: Axes, *, layout_kind: str) -> None:
    if layout_kind == "one_dimensional":
        axes.set_xlabel("x")
        axes.set_ylabel("")
        axes.set_yticks([])
        axes.margins(x=0.08, y=0.35)
        return

    axes.set_aspect("equal")
    axes.set_xlabel("layout x" if layout_kind == "abstract" else "x")
    axes.set_ylabel("layout y" if layout_kind == "abstract" else "y")
    axes.margins(0.12)


def _plot_title(geometry: Geometry) -> str:
    raw_title = geometry.metadata.get("name", geometry.metadata.get("generator", "geometry"))
    return str(raw_title).replace("_", " ").title()
