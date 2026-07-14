from __future__ import annotations

from toposc_lab.app.model_guides import model_guide, observable_guides
from toposc_lab.app.registry import MODEL_REGISTRY


def test_every_registered_model_has_a_curated_hamiltonian_guide() -> None:
    for specification in MODEL_REGISTRY.specifications():
        guide = model_guide(specification.key)

        assert guide.hamiltonian_latex
        assert guide.basis_description
        assert guide.construction_steps
        assert guide.parameters


def test_observable_guide_covers_core_workspace_quantities() -> None:
    names = {observable.name for observable in observable_guides()}

    assert {
        "eigenvalues",
        "minimum_abs_energy",
        "bulk_gap",
        "ipr",
        "edge_weight",
        "chern_number",
    } <= names
