"""Repeatable end-to-end validation smoke test for the completed Phase 6."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from toposc_lab.geometry import (
    BUILTIN_GEOMETRY_GENERATORS,
    Geometry,
    ammann_beenker_patch,
    canonical_graph_hash,
    chain,
    cubic,
    geometry_from_bytes,
    geometry_to_bytes,
    random_graph,
    sierpinski_gasket,
    square,
    validate_geometry,
)
from toposc_lab.hamiltonians import (
    NambuBasis,
    build_bdg_hamiltonian,
    build_spinless_p_wave_pairing,
    build_tight_binding_hamiltonian,
)
from toposc_lab.solvers.exact_diagonalization import ExactDiagonalizationSolver
from toposc_lab.visualization import plot_geometry


@dataclass(frozen=True, slots=True)
class GeneratorCase:
    parameters: dict[str, Any]
    seed: int | None = None


GENERATOR_CASES: dict[str, GeneratorCase] = {
    "ammann_beenker_patch": GeneratorCase({"radius": 3.0}),
    "artificial_rule_graph": GeneratorCase(
        {
            "iterations": 2,
            "displacement_rules": [[-1, 0], [0, -1], [0, 1], [1, 0]],
        }
    ),
    "body_centered_cubic": GeneratorCase({"n_x": 2, "n_y": 2, "n_z": 2}),
    "cayley_tree": GeneratorCase({"coordination": 3, "shells": 2}),
    "chain": GeneratorCase({"n_sites": 7}),
    "coordinate_cutoff_graph": GeneratorCase(
        {
            "coordinates": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            "cutoff": 1.01,
        }
    ),
    "cubic": GeneratorCase({"n_x": 2, "n_y": 2, "n_z": 2}),
    "fibonacci_chain": GeneratorCase({"order": 5}),
    "honeycomb": GeneratorCase({"n_x": 2, "n_y": 3}),
    "irregular_cluster": GeneratorCase({}),
    "k_nearest_neighbor_graph": GeneratorCase(
        {
            "coordinates": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            "k": 2,
        }
    ),
    "kagome": GeneratorCase({"n_x": 2, "n_y": 3}),
    "menger_sponge": GeneratorCase({"order": 1}),
    "random_graph": GeneratorCase(
        {"n_sites": 10, "edge_probability": 0.4},
        seed=17,
    ),
    "random_regular_graph": GeneratorCase(
        {"n_sites": 10, "degree": 2},
        seed=17,
    ),
    "ring": GeneratorCase({"n_sites": 8}),
    "scale_free_graph": GeneratorCase(
        {"n_sites": 10, "attachments_per_site": 2},
        seed=17,
    ),
    "sierpinski_carpet": GeneratorCase({"order": 2}),
    "sierpinski_gasket": GeneratorCase({"order": 3}),
    "silver_mean_chain": GeneratorCase({"order": 4}),
    "small_world_network": GeneratorCase(
        {
            "n_sites": 10,
            "neighbor_degree": 4,
            "rewiring_probability": 0.3,
        },
        seed=17,
    ),
    "square": GeneratorCase({"n_x": 3, "n_y": 4}),
    "tree": GeneratorCase({"parents": [0, 0, 1, 1, 2, 2]}),
    "triangular": GeneratorCase({"n_x": 3, "n_y": 4}),
}


def _assert_round_trip(actual: Geometry, expected: Geometry) -> None:
    assert actual.n_sites == expected.n_sites
    assert actual.edges == expected.edges
    assert actual.embedding_dimension == expected.embedding_dimension
    assert actual.boundary_sites == expected.boundary_sites
    assert actual.boundary_components == expected.boundary_components
    assert actual.site_types == expected.site_types
    assert actual.dimension_records == expected.dimension_records
    assert actual.faces == expected.faces
    assert actual.rooted_tree == expected.rooted_tree
    assert actual.metadata == expected.metadata
    if expected.coordinates is None:
        assert actual.coordinates is None
    else:
        assert actual.coordinates is not None
        assert actual.coordinates.dtype == expected.coordinates.dtype
        assert actual.coordinates.shape == expected.coordinates.shape
        assert actual.coordinates.tobytes() == expected.coordinates.tobytes()


def audit_all_generators() -> tuple[tuple[str, int, int, int], ...]:
    """Generate and cross-check every public built-in geometry generator."""
    registered = {generator.key for generator in BUILTIN_GEOMETRY_GENERATORS.generators()}
    assert set(GENERATOR_CASES) == registered

    rows: list[tuple[str, int, int, int]] = []
    for key, case in GENERATOR_CASES.items():
        first = BUILTIN_GEOMETRY_GENERATORS.generate(
            key,
            parameters=case.parameters,
            seed=case.seed,
        )
        second = BUILTIN_GEOMETRY_GENERATORS.generate(
            key,
            parameters=case.parameters,
            seed=case.seed,
        )
        report = validate_geometry(first)
        assert report.is_valid, (key, report.errors)
        assert first.metadata["generator"] == key
        generation = first.metadata["generation"]
        assert generation["generator_key"] == key
        assert generation["generator_version"] == 1
        assert generation["seed"] == case.seed

        assert first.edges == second.edges
        assert first.metadata == second.metadata
        if first.coordinates is None:
            assert second.coordinates is None
        else:
            assert second.coordinates is not None
            assert np.array_equal(first.coordinates, second.coordinates)
            assert first.embedding_dimension is not None
            assert first.coordinates.shape == (first.n_sites, first.embedding_dimension)

        if first.embedding_dimension is not None:
            assert all(
                edge.displacement is None
                or len(edge.displacement) == first.embedding_dimension
                for edge in first.edges
            )

        loaded = geometry_from_bytes(geometry_to_bytes(first))
        _assert_round_trip(loaded, first)
        assert canonical_graph_hash(loaded) == canonical_graph_hash(first)
        rows.append((key, first.n_sites, first.n_edges, len(report.connected_components)))
    return tuple(rows)


def representative_geometries() -> tuple[tuple[str, Geometry], ...]:
    """Return the six geometry families used in the visual and solver smoke test."""
    return (
        ("1D: open chain", chain(9)),
        ("2D: square lattice", square(4, 4)),
        ("3D: cubic lattice", cubic(3, 3, 3)),
        ("Fractal: Sierpinski gasket", sierpinski_gasket(3)),
        ("Random: G(n,p) (circular layout)", random_graph(12, 0.3, seed=23)),
        ("Quasiperiodic: Ammann-Beenker", ammann_beenker_patch(3.5)),
    )


def validate_hamiltonian_pipeline(
    geometries: tuple[tuple[str, Geometry], ...],
) -> tuple[tuple[str, int, float, float], ...]:
    """Build and solve normal and algebraic graph-BdG Hamiltonians."""
    solver = ExactDiagonalizationSolver()
    rows: list[tuple[str, int, float, float]] = []
    for label, geometry in geometries:
        report = validate_geometry(geometry)
        assert report.is_valid, (label, report.errors)

        normal = build_tight_binding_hamiltonian(
            geometry,
            onsite=0.25,
            hopping=-1.0,
        )
        normal_result = solver.solve(normal)
        assert normal.shape == (geometry.n_sites, geometry.n_sites)
        assert np.all(np.isfinite(normal_result.eigenvalues))
        assert np.allclose(normal, normal.conj().T)

        pairing = build_spinless_p_wave_pairing(geometry, pairing=0.2)
        bdg = build_bdg_hamiltonian(
            normal,
            pairing,
            basis=NambuBasis(n_sites=geometry.n_sites),
        )
        bdg_result = solver.solve(bdg)
        assert bdg.shape == (2 * geometry.n_sites, 2 * geometry.n_sites)
        assert np.all(np.isfinite(bdg_result.eigenvalues))
        assert np.allclose(bdg, bdg.conj().T)
        assert np.allclose(
            bdg_result.eigenvalues,
            -bdg_result.eigenvalues[::-1],
            rtol=0.0,
            atol=1e-10,
        )
        rows.append(
            (
                label,
                geometry.n_sites,
                float(np.min(normal_result.eigenvalues)),
                float(np.min(np.abs(bdg_result.eigenvalues))),
            )
        )
    return tuple(rows)


def save_geometry_overview(
    geometries: tuple[tuple[str, Geometry], ...],
    output_path: Path,
) -> Path:
    """Render every representative geometry, using fallback layout if needed."""
    figure = plt.figure(figsize=(12, 14), constrained_layout=True)
    for panel, (title, geometry) in enumerate(geometries, start=1):
        if geometry.embedding_dimension == 3:
            axes_item = figure.add_subplot(3, 2, panel, projection="3d")
            _plot_three_dimensional_geometry(geometry, axes_item, title=title)
            continue

        axes_item = figure.add_subplot(3, 2, panel)
        plot_geometry(geometry, axes=axes_item, title=title, show=False, site_size=32.0)
        legend = axes_item.get_legend()
        if title.startswith("Quasiperiodic") and legend is not None:
            legend.remove()
    figure.suptitle("Phase 6 geometry validation smoke test", fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    assert output_path.is_file() and output_path.stat().st_size > 0
    return output_path


def _plot_three_dimensional_geometry(
    geometry: Geometry,
    axes: Any,
    *,
    title: str,
) -> None:
    assert geometry.coordinates is not None
    assert geometry.coordinates.shape[1] == 3
    coordinates = geometry.coordinates
    for edge in geometry.edges:
        edge_coordinates = coordinates[[edge.source, edge.target]]
        axes.plot(
            edge_coordinates[:, 0],
            edge_coordinates[:, 1],
            edge_coordinates[:, 2],
            color="0.65",
            linewidth=0.9,
        )
    axes.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        coordinates[:, 2],
        s=24.0,
        color="tab:blue",
        edgecolors="white",
        linewidths=0.5,
    )
    if geometry.boundary_sites:
        boundary = np.asarray(sorted(geometry.boundary_sites), dtype=int)
        axes.scatter(
            coordinates[boundary, 0],
            coordinates[boundary, 1],
            coordinates[boundary, 2],
            s=42.0,
            facecolors="none",
            edgecolors="tab:orange",
            linewidths=1.0,
        )
    axes.set_title(title)
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z")
    axes.set_box_aspect((1.0, 1.0, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase_6_validation/geometries.png"),
        help="output path for the geometry overview PNG",
    )
    arguments = parser.parse_args()

    generator_rows = audit_all_generators()
    geometries = representative_geometries()
    pipeline_rows = validate_hamiltonian_pipeline(geometries)
    output_path = save_geometry_overview(geometries, arguments.output)

    print(f"Validated {len(generator_rows)} registered geometry generators.")
    for key, n_sites, n_edges, components in generator_rows:
        print(
            f"  {key:32} sites={n_sites:4d} edges={n_edges:4d} "
            f"components={components}"
        )
    print("Hamiltonian/solver smoke results:")
    for label, n_sites, minimum_energy, bdg_gap in pipeline_rows:
        print(
            f"  {label:42} sites={n_sites:3d} "
            f"E_min={minimum_energy: .6f} BdG|min(E)|={bdg_gap:.6f}"
        )
    print(f"Saved visualization to {output_path.resolve()}")


if __name__ == "__main__":
    main()
