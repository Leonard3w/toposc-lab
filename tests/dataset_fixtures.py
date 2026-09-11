from __future__ import annotations

from toposc_lab.data.dataset_codec import create_dataset_record
from toposc_lab.data.dataset_schema import (
    DatasetRecord,
    GeometryRecord,
    ModelParametersRecord,
    ObservableResultRecord,
    ReproducibilityMetadata,
    RobustnessFailureRecord,
    RobustnessResultRecord,
    SpectrumRecord,
    TopologyResultRecord,
    TopologyValidity,
)
from toposc_lab.geometry import Geometry, chain


def representative_dataset_record(
    *,
    geometry: Geometry | None = None,
    seed: int = 17,
    family_label: str | None = "chain_family",
) -> DatasetRecord:
    selected_geometry = chain(4) if geometry is None else geometry
    return create_dataset_record(
        geometry=GeometryRecord.from_geometry(selected_geometry, family_label=family_label),
        model=ModelParametersRecord(
            model_name="geometry_kitaev_chain",
            model_version="1",
            parameters={"mu": 0.25, "hopping": -1.0, "pairing": 0.7 + 0.1j},
        ),
        spectrum=SpectrumRecord(
            eigenvalues=(-1.2, -0.2, 0.2, 1.2),
            energy_unit="hopping",
            selection="closest_to_zero",
            basis_size=2 * selected_geometry.n_sites,
            is_complete=False,
        ),
        observables=(
            ObservableResultRecord(
                kind="spectral_gap",
                version="1",
                values={"gap": 0.2},
                units={"gap": "hopping"},
                conventions={"definition": "smallest_positive_energy"},
            ),
        ),
        topology=(
            TopologyResultRecord(
                method="pfaffian_1d",
                version="1",
                validity=TopologyValidity.VALID,
                invariant_value=-1,
                is_topological=True,
                parameters={"boundary": "periodic_reference"},
                tolerances={"phs": 1e-10},
            ),
        ),
        robustness=(
            RobustnessResultRecord(
                protocol="onsite_uniform",
                version="1",
                seeds=(101, 102, 103),
                parameters={"width": 0.1},
                statistics={"success_fraction": 2 / 3, "sample_count": 3},
                uncertainty={"standard_error": 0.2721655269759087},
                failures=(
                    RobustnessFailureRecord(
                        seed=103,
                        stage="evaluation",
                        error_type="LinAlgError",
                        message="reference fixture failure",
                    ),
                ),
            ),
        ),
        provenance=ReproducibilityMetadata(
            seed=seed,
            git_commit="dadd0da",
            git_dirty=True,
            package_version="0.1.0",
            solver_name="exact_diagonalization",
            solver_version="numpy-eigh",
            solver_settings={"driver": "eigh", "vectors": True},
            tolerances={"zero_energy": 1e-9, "phs": 1e-10},
            timestamp_utc="2026-09-11T12:00:00Z",
            runtime={"python": "3.11", "numpy": "2"},
        ),
    )
