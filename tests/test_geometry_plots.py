from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from toposc_lab.geometry import Geometry, GeometryEdge, chain, cubic, irregular_cluster, square
from toposc_lab.visualization import plot_geometry


def test_plot_geometry_draws_one_dimensional_chain() -> None:
    geometry = chain(5)
    figure, axes = plot_geometry(geometry, show_site_indices=True, show=False)

    site_positions = axes.collections[0].get_offsets()
    assert len(axes.lines) == 4
    assert np.array_equal(site_positions[:, 1], np.zeros(5))
    assert axes.get_xlabel() == "x"
    assert axes.get_ylabel() == ""
    assert len(axes.texts) == 5

    plt.close(figure)


def test_plot_geometry_colors_types_and_marks_boundary_sites() -> None:
    geometry = irregular_cluster()
    figure, axes = plot_geometry(geometry, show=False)

    assert len(axes.lines) == geometry.n_edges
    assert len(axes.collections) == 4
    assert axes.get_aspect() in (1.0, "equal")
    assert axes.get_title() == "Seven Site Reference"
    assert "Edge type: long_range" in axes.get_legend_handles_labels()[1]
    assert "Boundary site" in axes.get_legend_handles_labels()[1]

    plt.close(figure)


def test_plot_geometry_uses_circular_layout_without_coordinates() -> None:
    geometry = Geometry(
        n_sites=4,
        edges=tuple(GeometryEdge(site, (site + 1) % 4) for site in range(4)),
    )
    figure, axes = plot_geometry(geometry, show=False)
    positions = np.asarray(axes.collections[0].get_offsets(), dtype=float)

    assert np.allclose(np.linalg.norm(positions, axis=1), 1.0)
    assert axes.get_xlabel() == "layout x"
    assert axes.get_ylabel() == "layout y"

    plt.close(figure)


def test_plot_geometry_hides_wrapped_periodic_edges_by_default() -> None:
    geometry = square(3, 3, boundary_x="periodic", boundary_y="periodic")

    hidden_figure, hidden_axes = plot_geometry(geometry, show=False)
    shown_figure, shown_axes = plot_geometry(
        geometry,
        show_periodic_edges=True,
        show=False,
    )

    assert len(hidden_axes.lines) == 12
    assert len(shown_axes.lines) == 18

    plt.close(hidden_figure)
    plt.close(shown_figure)


def test_plot_geometry_reuses_provided_axes() -> None:
    figure, axes = plt.subplots()
    returned_figure, returned_axes = plot_geometry(
        chain(3),
        axes=axes,
        title="Custom title",
        show=False,
    )

    assert returned_figure is figure
    assert returned_axes is axes
    assert axes.get_title() == "Custom title"

    plt.close(figure)


def test_plot_geometry_projects_three_dimensional_coordinates_to_xy() -> None:
    geometry = cubic(2, 2, 2)
    figure, axes = plot_geometry(geometry, show=False)
    positions = np.asarray(axes.collections[0].get_offsets(), dtype=float)

    assert geometry.coordinates is not None
    assert np.array_equal(positions, geometry.coordinates[:, :2])
    assert axes.get_xlabel() == "x"
    assert axes.get_ylabel() == "y"

    plt.close(figure)


def test_plot_geometry_rejects_invalid_site_size() -> None:
    for site_size in (0.0, -1.0, np.inf, np.nan):
        try:
            plot_geometry(chain(3), site_size=site_size, show=False)
        except ValueError as error:
            assert "finite and positive" in str(error)
        else:
            raise AssertionError("invalid site_size was accepted")
