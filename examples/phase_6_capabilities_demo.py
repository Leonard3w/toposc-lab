"""Concise, verified capability demonstration for Toposc-Lab after Phase 6."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from phase_6_validation_smoke import (
    representative_geometries,
    save_geometry_overview,
)
from toposc_lab.models.geometry_kitaev_chain import GeometryKitaevChain
from toposc_lab.models.kitaev_chain import KitaevChainParameters
from toposc_lab.models.qwz_model import QWZModel, QWZModelParameters
from toposc_lab.observables.localization import (
    boundary_weight_from_geometry,
    inverse_participation_ratio,
)
from toposc_lab.observables.majorana import (
    finite_size_splitting_diagnostics,
    majorana_diagnostics_from_result,
)
from toposc_lab.observables.spectrum import lowest_abs_energy, spectral_gap
from toposc_lab.observables.symmetries import check_bdg_particle_hole_symmetry
from toposc_lab.solvers.exact_diagonalization import ExactDiagonalizationSolver
from toposc_lab.topology import (
    bott_index,
    local_chern_marker,
    one_dimensional_pfaffian_invariant,
    spectral_localizer,
)
from toposc_lab.topology.symmetry import SymmetryClassification


@dataclass(frozen=True, slots=True)
class KitaevDemoResult:
    energies: np.ndarray
    site_probability: np.ndarray
    polarization_magnitude: np.ndarray
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class QWZDemoResult:
    energies: np.ndarray
    marker_positions: np.ndarray
    local_marker: np.ndarray
    bulk_mask: np.ndarray
    summary: dict[str, Any]


def _class_bdi() -> SymmetryClassification:
    return SymmetryClassification.from_signature(
        time_reversal_square=1,
        particle_hole_square=1,
        chiral_symmetry=True,
    )


def _class_a() -> SymmetryClassification:
    return SymmetryClassification.from_signature(
        time_reversal_square=None,
        particle_hole_square=None,
        chiral_symmetry=False,
    )


def _kitaev_endpoint_hamiltonians(
    chemical_potential: float,
    *,
    hopping: float,
) -> tuple[np.ndarray, np.ndarray]:
    energy_zero = -chemical_potential - 2.0 * hopping
    energy_pi = -chemical_potential + 2.0 * hopping
    return (
        np.diag([energy_zero, -energy_zero]),
        np.diag([energy_pi, -energy_pi]),
    )


def run_kitaev_demo() -> KitaevDemoResult:
    """Solve an open Kitaev chain and evaluate existing 1D diagnostics."""
    parameters = KitaevChainParameters(
        n_sites=8,
        hopping=1.0,
        chemical_potential=0.5,
        pairing=0.8,
        boundary="open",
    )
    model = GeometryKitaevChain(parameters)
    result = ExactDiagonalizationSolver().solve_model(model)
    state_index = int(np.argmin(np.abs(result.eigenvalues)))
    majorana = majorana_diagnostics_from_result(
        result,
        state_index,
        model.nambu_basis,
    )
    splitting = finite_size_splitting_diagnostics(
        result.eigenvalues,
        splitting_tolerance=1.0e-3,
    )
    particle_hole = check_bdg_particle_hole_symmetry(
        model.hamiltonian(),
        model.nambu_basis,
    )
    endpoint_zero, endpoint_pi = _kitaev_endpoint_hamiltonians(
        parameters.chemical_potential,
        hopping=parameters.hopping,
    )
    pfaffian = one_dimensional_pfaffian_invariant(
        endpoint_zero,
        endpoint_pi,
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        _class_bdi(),
    )

    boundary_weight = boundary_weight_from_geometry(
        majorana.site_probability,
        model.geometry,
    )
    summary: dict[str, Any] = {
        "model": model.model_name,
        "n_sites": model.geometry.n_sites,
        "hamiltonian_dimension": model.hamiltonian().shape[0],
        "closest_abs_energy": lowest_abs_energy(result.eigenvalues),
        "spectral_gap_at_zero": spectral_gap(result.eigenvalues),
        "finite_size_classification": splitting.classification,
        "next_excitation_energy": splitting.next_excitation_energy,
        "isolation_ratio": splitting.isolation_ratio,
        "selected_state_index": state_index,
        "boundary_weight": boundary_weight,
        "inverse_participation_ratio": inverse_participation_ratio(
            majorana.site_probability
        ),
        "majorana_self_conjugacy": majorana.self_conjugacy,
        "majorana_polarization_norm": majorana.polarization_norm,
        "particle_hole_residual": particle_hole.residual,
        "pfaffian_invariant": pfaffian.invariant,
        "pfaffian_is_topological": pfaffian.is_topological,
    }

    assert splitting.classification == "split_pair_candidate"
    assert boundary_weight > 0.9
    assert particle_hole.satisfied
    assert pfaffian.invariant == -1
    return KitaevDemoResult(
        energies=result.eigenvalues,
        site_probability=majorana.site_probability,
        polarization_magnitude=majorana.polarization_magnitude,
        summary=summary,
    )


def run_qwz_demo() -> QWZDemoResult:
    """Solve a finite QWZ model and cross-check existing 2D diagnostics."""
    size = 8
    model = QWZModel(
        QWZModelParameters(
            n_x=size,
            n_y=size,
            mass=1.0,
            boundary_x="open",
            boundary_y="open",
        )
    )
    hamiltonian = model.hamiltonian()
    eigensystem = ExactDiagonalizationSolver().solve(hamiltonian)
    site_coordinates = model.lattice.coordinates.astype(float)
    basis_coordinates = np.repeat(site_coordinates, 2, axis=0)
    bulk_mask = np.all(
        (site_coordinates >= 3.0) & (site_coordinates < 5.0),
        axis=1,
    )
    classification = _class_a()
    bott = bott_index(
        hamiltonian,
        basis_coordinates,
        np.array([size, size], dtype=float),
        classification,
    )
    marker = local_chern_marker(
        hamiltonian,
        basis_coordinates,
        1.0,
        bulk_mask,
        classification,
    )
    localizer = spectral_localizer(
        hamiltonian,
        basis_coordinates,
        np.array([(size - 1.0) / 2.0, (size - 1.0) / 2.0]),
        classification,
        kappa=0.2,
    )
    summary: dict[str, Any] = {
        "model": model.model_name,
        "shape": [size, size],
        "hamiltonian_dimension": hamiltonian.shape[0],
        "minimum_fermi_distance": float(np.min(np.abs(eigensystem.eigenvalues))),
        "bott_estimate": bott.bott_estimate,
        "bott_index": bott.bott_index,
        "bott_quantization_error": bott.quantization_error,
        "bulk_chern_estimate": marker.bulk_chern_estimate,
        "local_chern_number": marker.chern_number,
        "local_chern_quantization_error": marker.quantization_error,
        "spectral_localizer_index": localizer.local_chern_number,
        "spectral_localizer_gap": localizer.localizer_gap,
    }

    assert bott.bott_index == 1
    assert marker.chern_number == 1
    assert localizer.local_chern_number == 1
    return QWZDemoResult(
        energies=eigensystem.eigenvalues,
        marker_positions=marker.positions,
        local_marker=marker.local_marker,
        bulk_mask=marker.bulk_mask,
        summary=summary,
    )


def save_physics_figure(
    kitaev: KitaevDemoResult,
    qwz: QWZDemoResult,
    output_path: Path,
) -> Path:
    """Save spectra, localization, and a local topology diagnostic."""
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    kitaev_indices = np.arange(kitaev.energies.size)
    axes[0, 0].scatter(kitaev_indices, kitaev.energies, s=20.0)
    axes[0, 0].axhline(0.0, color="0.45", linewidth=0.8)
    axes[0, 0].set_title("Open Kitaev-chain BdG spectrum")
    axes[0, 0].set_xlabel("sorted eigenvalue index")
    axes[0, 0].set_ylabel("energy")

    sites = np.arange(kitaev.site_probability.size)
    axes[0, 1].plot(sites, kitaev.site_probability, "o-", label="site probability")
    axes[0, 1].plot(
        sites,
        kitaev.polarization_magnitude,
        "s--",
        label="Majorana polarization magnitude",
    )
    axes[0, 1].set_title("Nearest-zero Kitaev state")
    axes[0, 1].set_xlabel("site")
    axes[0, 1].set_ylabel("weight")
    axes[0, 1].legend(loc="best")

    qwz_indices = np.arange(qwz.energies.size)
    axes[1, 0].scatter(qwz_indices, qwz.energies, s=10.0)
    axes[1, 0].axhline(0.0, color="0.45", linewidth=0.8)
    axes[1, 0].set_title("Finite QWZ spectrum, mass = 1")
    axes[1, 0].set_xlabel("sorted eigenvalue index")
    axes[1, 0].set_ylabel("energy")

    marker_limit = float(np.max(np.abs(qwz.local_marker)))
    marker_plot = axes[1, 1].scatter(
        qwz.marker_positions[:, 0],
        qwz.marker_positions[:, 1],
        c=qwz.local_marker,
        cmap="coolwarm",
        vmin=-marker_limit,
        vmax=marker_limit,
        marker="s",
        s=150.0,
    )
    bulk_positions = qwz.marker_positions[qwz.bulk_mask]
    axes[1, 1].scatter(
        bulk_positions[:, 0],
        bulk_positions[:, 1],
        facecolors="none",
        edgecolors="black",
        marker="s",
        s=210.0,
        linewidths=1.3,
        label="bulk average region",
    )
    axes[1, 1].set_aspect("equal")
    axes[1, 1].set_title("QWZ local Chern marker")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("y")
    axes[1, 1].legend(loc="upper right")
    figure.colorbar(marker_plot, ax=axes[1, 1], label="local marker")

    figure.suptitle("Toposc-Lab after Phase 6: verified physics capabilities")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase_6_capabilities"),
    )
    arguments = parser.parse_args()
    output_directory: Path = arguments.output_dir
    output_directory.mkdir(parents=True, exist_ok=True)

    kitaev = run_kitaev_demo()
    qwz = run_qwz_demo()
    geometry_path = save_geometry_overview(
        representative_geometries(),
        output_directory / "geometry_families.png",
    )
    physics_path = save_physics_figure(
        kitaev,
        qwz,
        output_directory / "physics_diagnostics.png",
    )
    summary = {
        "scope": "verified capabilities available after Phase 6",
        "kitaev_chain": kitaev.summary,
        "qwz_chern_insulator": qwz.summary,
        "notes": [
            "finite-size gaps are not thermodynamic bulk-gap proofs",
            "Majorana diagnostics do not replace an independent topology invariant",
            "no Phase 7 automated geometry evaluation is used",
        ],
    }
    summary_path = output_directory / "numerical_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Geometry overview: {geometry_path.resolve()}")
    print(f"Physics diagnostics: {physics_path.resolve()}")
    print(f"Numerical summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
