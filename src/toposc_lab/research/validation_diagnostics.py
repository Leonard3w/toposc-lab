"""Additional finite-system diagnostics; the Phase-17 objective is unchanged."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from toposc_lab.geometry import Geometry
from toposc_lab.models.chiral_p_wave import ChiralPWaveModel, ChiralPWaveParameters
from toposc_lab.research.physics import FiniteSystemEvaluator, PhysicsProtocol
from toposc_lab.robustness.disorder import exact_hamiltonian_id
from toposc_lab.robustness.onsite import apply_uniform_onsite_disorder
from toposc_lab.topology import SymmetryClassification, local_chern_marker, spectral_localizer

DIAGNOSTIC_VERSION = "phase18.spatial-boundary.v1"


def boundary_distance(coordinates: np.ndarray) -> np.ndarray:
    """Euclidean distance to the nearest side of the fixed open square."""
    return np.min(
        np.column_stack(
            (coordinates - coordinates.min(axis=0), coordinates.max(axis=0) - coordinates)
        ),
        axis=1,
    )


def probe_positions(side: int) -> list[list[float]]:
    center = (side - 1) / 2
    inner = (2.0, center, float(side - 3)) if side >= 6 else (center,)
    cut = sorted({0.0, 1.0, *inner, float(side - 2), float(side - 1)})
    points = {(x, y) for x in inner for y in inner}
    points.update((x, center) for x in cut)
    points.update((center, y) for y in cut)
    return [list(p) for p in sorted(points)]


def boundary_diagnostics(
    energies: np.ndarray,
    vectors: np.ndarray,
    coordinates: np.ndarray,
    *,
    energy_cutoff: float = 0.5,
    group_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Whole-window and near-degenerate projector densities, not mode certificates.

    Signed energy groups are contiguous at tolerance. Include an entire group
    when any member meets the fixed window, avoiding arbitrary degenerate cuts.
    Both BdG signs contribute; the count is not an independent-particle count.
    """
    n = len(coordinates)
    if energy_cutoff <= 0 or group_tolerance <= 0:
        raise ValueError("positive energy cutoff and group tolerance required")
    if vectors.shape != (2 * n, len(energies)) or np.any(np.diff(energies) < 0):
        raise ValueError("sorted energies and component-major BdG eigenvectors required")
    if not np.isfinite(energies).all() or not np.isfinite(vectors).all():
        raise ValueError("finite eigensystem required")
    distance = boundary_distance(coordinates)
    groups = np.split(
        np.arange(len(energies)), np.flatnonzero(np.diff(energies) > group_tolerance) + 1
    )
    groups = [g for g in groups if np.any(np.abs(energies[g]) <= energy_cutoff)]
    selected = np.concatenate(groups) if groups else np.array([], dtype=int)
    probability = np.abs(vectors[:n]) ** 2 + np.abs(vectors[n:]) ** 2

    def profile(weights: np.ndarray) -> dict[str, Any]:
        return {
            "site_probability": weights.tolist(),
            "strip_weights": {str(w): float(weights[distance < w].sum()) for w in (1, 2, 3)},
            "shell_weights": [
                float(weights[(distance >= d) & (distance < d + 1)].sum())
                for d in range(int(distance.max()) + 1)
            ],
            "mean_boundary_distance": float(weights @ distance),
        }

    result: dict[str, Any] = {
        "status": "available" if len(selected) else "empty_window",
        "energy_cutoff": energy_cutoff,
        "group_tolerance": group_tolerance,
        "state_count": len(selected),
        "state_indices": selected.tolist(),
        "energies": energies[selected].tolist(),
        "distance_to_boundary": distance.tolist(),
        "states": [
            {"index": int(i), "energy": float(energies[i]), **profile(probability[:, i])}
            for i in selected
        ],
        "groups": [
            {
                "indices": g.tolist(),
                "energies": energies[g].tolist(),
                **profile(probability[:, g].mean(axis=1)),
            }
            for g in groups
        ],
        "window": profile(probability[:, selected].mean(axis=1)) if len(selected) else None,
        "claim": "Finite low-energy localization only; no chirality/transport certificate",
    }
    return result


class ValidationEvaluator:
    """Reuse the exact adapter, then augment its result with separately timed diagnostics."""

    def __init__(self, protocol: PhysicsProtocol, provenance: Any = None) -> None:
        self.base = FiniteSystemEvaluator(protocol, provenance)
        self.protocol = protocol

    def plan(self) -> list[dict[str, Any]]:
        return self.base.plan()

    def evaluate(self, geometry: Geometry, stage: dict[str, Any]) -> dict[str, Any]:
        start = time.perf_counter()
        result = self.base.evaluate(geometry, stage)
        base_seconds = time.perf_counter() - start
        model = ChiralPWaveModel(
            geometry, ChiralPWaveParameters(hopping=1, chemical_potential=2, pairing=1, chirality=1)
        )
        matrix = model.hamiltonian()
        if stage["kind"] == "disorder":
            matrix = np.asarray(
                apply_uniform_onsite_disorder(
                    geometry,
                    matrix,
                    width=stage["width"],
                    seed=stage["seed"],
                    nambu_basis=model.nambu_basis,
                ).state
            )
        if exact_hamiltonian_id(matrix) != result["hamiltonian_id"]:
            raise ValueError("Diagnostic matrix differs from primary adapter matrix")
        energies, vectors = np.linalg.eigh(matrix)
        tol = self.protocol.tolerance
        if not np.allclose(energies, result["spectrum"], atol=tol, rtol=0):
            raise ValueError("Diagnostic spectrum differs from primary spectrum")
        if np.max(np.abs(matrix @ vectors - vectors * energies)) > tol:
            raise ValueError("Diagnostic eigensystem residual exceeds tolerance")
        assert geometry.coordinates is not None
        coordinates = geometry.coordinates
        side = round(geometry.n_sites**0.5)
        classification = SymmetryClassification.from_signature(
            time_reversal_square=None, particle_hole_square=1, chiral_symmetry=False
        )
        basis_coordinates = np.tile(coordinates, (2, 1))
        spatial = []
        for point in probe_positions(side):
            for k, kappa in enumerate(self.protocol.kappas):
                if point == result["probe"]:
                    index, gap = result["indices"][k], result["localizer_gaps"][k]
                    spatial.append(
                        {
                            "point": point,
                            "kappa": kappa,
                            "index": index,
                            "gap": gap,
                            "valid": index is not None and gap > tol,
                            "status": "available" if gap > tol else "singular",
                        }
                    )
                    continue
                try:
                    local = spectral_localizer(
                        matrix,
                        basis_coordinates,
                        np.asarray(point),
                        classification,
                        kappa=kappa,
                        tolerance=tol,
                    )
                    if not np.isfinite(local.localizer_gap):
                        raise ValueError("Nonfinite localizer gap")
                    spatial.append(
                        {
                            "point": point,
                            "kappa": kappa,
                            "index": local.local_chern_number,
                            "gap": local.localizer_gap,
                            "valid": local.is_invertible,
                            "status": "available" if local.is_invertible else "singular",
                        }
                    )
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                    spatial.append(
                        {
                            "point": point,
                            "kappa": kappa,
                            "valid": False,
                            "index": None,
                            "gap": None,
                            "status": "numerical_error",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
        unique = np.unique(basis_coordinates, axis=0)
        mask = boundary_distance(unique) >= 2
        marker: dict[str, Any]
        if not mask.any():
            marker = {"status": "unavailable", "reason": "No sites in declared bulk strip"}
        elif float(np.min(np.abs(energies))) <= tol:
            marker = {"status": "unavailable", "reason": "Fermi-level eigenstate"}
        else:
            try:
                c = local_chern_marker(
                    matrix, basis_coordinates, 1.0, mask, classification, tolerance=tol
                )
                marker = {
                    "status": "available",
                    "positions": c.positions.tolist(),
                    "values": c.local_marker.tolist(),
                    "bulk_mask": c.bulk_mask.tolist(),
                    "bulk_mean": c.bulk_chern_estimate,
                    "trace_residual": c.finite_sample_trace_residual,
                    "projector_residual": c.maximum_projector_residual,
                    "minimum_fermi_distance": c.minimum_fermi_distance,
                    "convention": "4pi Im diag(P X Q Y P); Nambu sum; area=1",
                    "scope": "Descriptive finite bulk mean; no mobility-gap assumption verified",
                }
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                marker = {"status": "numerical_error", "reason": str(error)}
        center_valid = all(i is not None for i in result["indices"]) and all(
            gap > tol for gap in result["localizer_gaps"]
        )
        symmetry_valid = (
            max(
                result["metrics"]["operator_phs_residual"],
                result["metrics"]["spectral_phs_residual"],
            )
            <= tol
        )
        result["primary_valid"] = bool(center_valid and symmetry_valid)
        result["primary_invalid_reason"] = None if result["primary_valid"] else "singular_or_phs"
        result["diagnostic_version"] = DIAGNOSTIC_VERSION
        result["spatial"] = spatial
        result["chern_marker"] = marker
        result["boundary_window"] = boundary_diagnostics(energies, vectors, coordinates)
        result["timing"] = {
            "primary_seconds": base_seconds,
            "additional_seconds": time.perf_counter() - start - base_seconds,
        }
        result["operations"] = {
            "localizer_diagonalizations": len(spatial),
            # Each existing localizer also re-evaluates the H spectrum.
            "hamiltonian_diagonalizations": len(spatial)
            + 2
            + int(mask.any() and float(np.min(np.abs(energies))) > tol),
            "hamiltonian_dimension": len(energies),
            "localizer_dimension": 2 * len(energies),
        }
        return result
