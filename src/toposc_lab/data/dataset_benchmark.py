"""Small fixed-seed scientific acceptance benchmark for Dataset Gate 11."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from toposc_lab.data.dataset_codec import create_dataset_record
from toposc_lab.data.dataset_schema import (
    DatasetRecord,
    GeometryRecord,
    ModelParametersRecord,
    ObservableResultRecord,
    ReproducibilityMetadata,
    SpectrumRecord,
    TopologyResultRecord,
    TopologyValidity,
)
from toposc_lab.data.dataset_splitting import DatasetSplitConfig, split_dataset
from toposc_lab.data.dataset_storage import (
    ExactPhysicsDataset,
    dataset_to_bytes,
    load_dataset,
    save_dataset,
)
from toposc_lab.data.dataset_validation import validate_dataset
from toposc_lab.models.geometry_kitaev_chain import GeometryKitaevChain
from toposc_lab.models.kitaev_chain import KitaevChainParameters
from toposc_lab.solvers.exact_diagonalization import ExactDiagonalizationSolver
from toposc_lab.topology.pfaffian import one_dimensional_pfaffian_invariant
from toposc_lab.topology.symmetry import SymmetryClassification

DATASET_GATE_BENCHMARK_SEED = 11_011


@dataclass(frozen=True, slots=True)
class DatasetFoundationBenchmarkResult:
    """Compact reproducibility evidence emitted by the Gate-11 benchmark."""

    seed: int
    record_count: int
    dataset_sha256: str
    topological_classifications: tuple[bool, ...]
    split_record_counts: tuple[int, int, int]
    validation_passed: bool
    round_trip_identical: bool


def run_dataset_foundation_benchmark(
    output_path: str | Path,
    *,
    seed: int = DATASET_GATE_BENCHMARK_SEED,
) -> DatasetFoundationBenchmarkResult:
    """Run exact positive/negative Kitaev references through the full data layer."""
    cases = ((6, 0.0), (7, 3.0), (8, -1.0))
    records = tuple(
        _exact_kitaev_record(n_sites=n_sites, chemical_potential=mu, seed=seed + index)
        for index, (n_sites, mu) in enumerate(cases)
    )
    dataset = ExactPhysicsDataset(records)
    report = validate_dataset(dataset)
    report.raise_for_errors()
    path = save_dataset(output_path, dataset)
    loaded = load_dataset(path)
    before = dataset_to_bytes(dataset)
    after = dataset_to_bytes(loaded)
    split = split_dataset(
        loaded,
        config=DatasetSplitConfig(
            train_fraction=0.6,
            validation_fraction=0.2,
            test_fraction=0.2,
            seed=seed,
        ),
    )
    return DatasetFoundationBenchmarkResult(
        seed=seed,
        record_count=len(records),
        dataset_sha256=hashlib.sha256(after).hexdigest(),
        topological_classifications=tuple(
            bool(record.topology[0].is_topological) for record in loaded.records
        ),
        split_record_counts=(
            len(split.train_ids),
            len(split.validation_ids),
            len(split.test_ids),
        ),
        validation_passed=report.is_valid and not split.diagnostics.leakage_groups,
        round_trip_identical=before == after,
    )


def _exact_kitaev_record(
    *,
    n_sites: int,
    chemical_potential: float,
    seed: int,
) -> DatasetRecord:
    parameters = KitaevChainParameters(
        n_sites=n_sites,
        hopping=1.0,
        chemical_potential=chemical_potential,
        pairing=0.5,
        boundary="open",
    )
    model = GeometryKitaevChain(parameters)
    simulation = ExactDiagonalizationSolver().solve_model(model)
    endpoint_zero, endpoint_pi = _endpoint_hamiltonians(
        chemical_potential=chemical_potential,
        hopping=parameters.hopping,
    )
    invariant = one_dimensional_pfaffian_invariant(
        endpoint_zero,
        endpoint_pi,
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        SymmetryClassification.from_signature(
            time_reversal_square=1,
            particle_hole_square=1,
            chiral_symmetry=True,
        ),
        tolerance=1e-10,
    )
    gap = float(np.min(np.abs(simulation.eigenvalues)))
    return create_dataset_record(
        geometry=GeometryRecord.from_geometry(
            model.geometry,
            family_label=f"open_kitaev_chain_n{n_sites}",
        ),
        model=ModelParametersRecord(
            model_name=model.model_name,
            model_version="1",
            parameters=model.parameters,
        ),
        spectrum=SpectrumRecord(
            eigenvalues=tuple(float(value) for value in simulation.eigenvalues),
            energy_unit="hopping",
            selection="all",
            basis_size=simulation.basis_layout.dimension,
            is_complete=True,
        ),
        observables=(
            ObservableResultRecord(
                kind="minimum_absolute_energy",
                version="1",
                values={"gap": gap},
                units={"gap": "hopping"},
                conventions={"finite_system": True},
            ),
        ),
        topology=(
            TopologyResultRecord(
                method="pfaffian_1d",
                version="1",
                validity=TopologyValidity.VALID,
                invariant_value=invariant.invariant,
                is_topological=invariant.is_topological,
                parameters={
                    "bulk_reference": "translation_invariant_kitaev_endpoints",
                    "chemical_potential": chemical_potential,
                },
                tolerances={"particle_hole": 1e-10, "antisymmetry": 1e-10},
                warnings=(
                    "The bulk endpoint invariant is stored separately from the open finite-chain spectrum.",
                ),
            ),
        ),
        robustness=(),
        provenance=ReproducibilityMetadata(
            seed=seed,
            git_commit="dadd0da",
            git_dirty=True,
            package_version="0.1.0",
            solver_name="exact_diagonalization",
            solver_version="numpy.linalg.eigh",
            solver_settings={"eigenvectors": True, "basis_ordering": "component_major"},
            tolerances={"hermiticity": 1e-10, "particle_hole": 1e-10},
            timestamp_utc="2026-09-11T00:00:00Z",
            runtime={"benchmark": "dataset_foundation_v1"},
        ),
    )


def _endpoint_hamiltonians(
    *,
    chemical_potential: float,
    hopping: float,
) -> tuple[np.ndarray, np.ndarray]:
    energy_zero = -chemical_potential - 2.0 * hopping
    energy_pi = -chemical_potential + 2.0 * hopping
    return (
        np.diag([energy_zero, -energy_zero]),
        np.diag([energy_pi, -energy_pi]),
    )
