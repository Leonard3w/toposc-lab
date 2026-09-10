"""Frozen inputs for TOPOSC-P10-LOCAL-ENUM-001."""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from toposc_lab.evaluation.reproducibility import exact_geometry_id
from toposc_lab.search.phase_10_local_catalog import (
    LocalCatalogCandidate,
    build_local_rewire_catalog,
    validate_local_catalog_candidate,
)
from toposc_lab.search.phase_10_measurement_validation import (
    FULL_SIZES,
    MODEL_ROLES,
    PREFLIGHT_SIZE,
    ModelRole,
    build_measurement_cell,
    measurement_reference,
    model_parameters,
)

LOCAL_ENUMERATION_PROTOCOL_ID = "TOPOSC-P10-LOCAL-ENUM-001"
LOCAL_ENUMERATION_PROTOCOL_COMMIT = "c20e413d4fbcd441fd0142ff1ea930d29141d2c5"
LOCAL_ENUMERATION_PROTOCOL_PATH = (
    "docs/decisions/pre_phase_10_local_enumeration_protocol_v1.md"
)
CATALOG_COMMIT = "32e4f1ef33b63be0b9b617a9d43880d8b8be24f5"
CATALOG_IDS_SHA256 = "13c9e2c750af250f20fd3d245d48b6abc9b2936176e5187dd76ba245ff1c6be0"
PRIMARY_KAPPAS = (0.1, 0.2, 0.3)
EXTRA_KAPPAS = (0.09, 0.11, 0.18, 0.22, 0.27, 0.33)
ALL_KAPPAS = (*PRIMARY_KAPPAS, *EXTRA_KAPPAS)
PREFLIGHT_BUDGET = 8
FULL_BUDGET = 268


def build_local_enumeration_plan(*, preflight: bool) -> tuple[dict[str, Any], ...]:
    """Build the exact frozen eight- or 268-slot plan without evaluating physics."""
    if preflight:
        cells = [
            build_measurement_cell(PREFLIGHT_SIZE, role, block="control_start")
            for role in MODEL_ROLES
        ]
        cells.extend(
            build_measurement_cell(
                PREFLIGHT_SIZE,
                "topological",
                block="intervention",
                offset=0,
                arm=arm,
            )
            for arm in ("boundary", "interior")
        )
        cells.extend(
            build_measurement_cell(PREFLIGHT_SIZE, role, block="control_end")
            for role in MODEL_ROLES
        )
        prepared = tuple(_decorate_preflight(cell) for cell in cells)
        _audit_budget(prepared, PREFLIGHT_BUDGET)
        return prepared

    catalog = build_local_rewire_catalog()
    _audit_catalog(catalog.candidates)
    cells = []
    for n in FULL_SIZES:
        cells.extend(_control_cell(n, role, "control_start") for role in MODEL_ROLES)
        cells.extend(
            _candidate_cell(candidate)
            for candidate in catalog.candidates
            if candidate.n == n
        )
        cells.extend(_control_cell(n, role, "control_end") for role in MODEL_ROLES)
    prepared = tuple(cells)
    _audit_budget(prepared, FULL_BUDGET)
    return prepared


def paired_candidate_ids() -> tuple[tuple[str, str], ...]:
    """Return the verified, exhaustive 16-to-20 operation continuation."""
    candidates = build_local_rewire_catalog().candidates
    by_size = {n: [candidate for candidate in candidates if candidate.n == n] for n in FULL_SIZES}
    target = {
        (candidate.removed_edges, candidate.added_edges): candidate
        for candidate in by_size[20]
    }
    pairs = []
    used: set[str] = set()
    for candidate in by_size[16]:
        arm = candidate.origins[0].arm
        if any(origin.arm != arm for origin in candidate.origins):
            raise AssertionError("one candidate unexpectedly spans both catalog arms")
        dx = 0 if arm == "boundary" else 1
        operation = (
            _continue_edges(candidate.removed_edges, dx),
            _continue_edges(candidate.added_edges, dx),
        )
        other = target.get(operation)
        if other is None or other.geometry_id in used:
            raise AssertionError("catalog continuation is not a bijection")
        first_origins = {(origin.offset, origin.arm) for origin in candidate.origins}
        second_origins = {(origin.offset, origin.arm) for origin in other.origins}
        if first_origins != second_origins:
            raise AssertionError("catalog continuation changed its patch origins")
        pairs.append((candidate.geometry_id, other.geometry_id))
        used.add(other.geometry_id)
    if len(pairs) != 128 or len(used) != 128:
        raise AssertionError("catalog continuation must contain 128 unique pairs")
    return tuple(pairs)


def _decorate_preflight(cell: dict[str, Any]) -> dict[str, Any]:
    intervention = cell["block"] == "intervention"
    origin = () if not intervention else (
        {
            "offset": cell["offset"],
            "arm": cell["arm"],
            "measurement_sites": cell["measurement_sites"],
        },
    )
    return {
        **cell,
        "candidate_id": cell["geometry_id"] if intervention else None,
        "origins": origin,
        "known_measurement_variant": intervention,
        "rotation_signature": None,
        "dihedral_signature": None,
    }


def _control_cell(
    n: int, role: ModelRole, block: Literal["control_start", "control_end"]
) -> dict[str, Any]:
    return _decorate_preflight(build_measurement_cell(n, role, block=block))


def _candidate_cell(candidate: LocalCatalogCandidate) -> dict[str, Any]:
    geometry = candidate.genome.to_geometry()
    reference = measurement_reference(candidate.n)
    issues = validate_local_catalog_candidate(
        reference,
        geometry,
        candidate.removed_edges,
        candidate.added_edges,
        candidate.n,
    )
    if issues or exact_geometry_id(geometry) != candidate.geometry_id:
        raise AssertionError("catalog candidate no longer satisfies its frozen contracts")
    validation = {"is_valid": True, "issues": (), "source": "local_catalog_delta_contract"}
    return {
        "n": candidate.n,
        "block": "candidate",
        "model_role": "topological",
        "model_parameters": model_parameters("topological").model_dump(mode="json"),
        "offset": None,
        "arm": None,
        "affected_sites": candidate.affected_sites,
        "removed_edges": candidate.removed_edges,
        "added_edges": candidate.added_edges,
        "measurement_sites": (),
        "intervention_depth": min(
            min(site // candidate.n, site % candidate.n,
                candidate.n - 1 - site // candidate.n,
                candidate.n - 1 - site % candidate.n)
            for site in candidate.affected_sites
        ),
        "genome": candidate.genome,
        "geometry_id": candidate.geometry_id,
        "candidate_id": candidate.geometry_id,
        "validation": validation,
        "degree_sequence": tuple(
            len(geometry.neighbors(site)) for site in range(geometry.n_sites)
        ),
        "edge_distance": 2,
        "evaluation_seed": None,
        "origins": tuple(
            {
                "offset": origin.offset,
                "arm": origin.arm,
                "measurement_sites": origin.measurement_sites,
            }
            for origin in candidate.origins
        ),
        "known_measurement_variant": candidate.known_measurement_variant,
        "rotation_signature": candidate.rotation_signature,
        "dihedral_signature": candidate.dihedral_signature,
    }


def _audit_catalog(candidates: tuple[LocalCatalogCandidate, ...]) -> None:
    if len(candidates) != 256:
        raise AssertionError("local catalog must contain exactly 256 candidates")
    digest = hashlib.sha256(
        "\n".join(candidate.geometry_id for candidate in candidates).encode("ascii")
    ).hexdigest()
    if digest != CATALOG_IDS_SHA256:
        raise AssertionError("local catalog identity digest differs from the frozen protocol")
    paired_candidate_ids()


def _audit_budget(cells: tuple[dict[str, Any], ...], expected: int) -> None:
    if len(cells) != expected:
        raise AssertionError("local enumeration plan differs from its frozen budget")
    identities = [(cell["n"], cell["block"], cell["model_role"], cell["geometry_id"])
                  for cell in cells]
    if len(set(identities)) != expected:
        raise AssertionError("local enumeration plan contains duplicate slot identities")
    if any(cell["evaluation_seed"] is not None for cell in cells):
        raise AssertionError("local enumeration plan must not contain random seeds")


def _continue_edges(
    edges: tuple[tuple[int, int], tuple[int, int]], dx: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    transformed = []
    for edge in edges:
        endpoints = []
        for site in edge:
            x_value, y_value = divmod(site, 16)
            endpoints.append((x_value + dx) * 20 + y_value + 2)
        transformed.append((min(endpoints), max(endpoints)))
    return tuple(sorted(transformed))  # type: ignore[return-value]
