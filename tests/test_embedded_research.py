"""Old/new adapter equivalence is a prerequisite to new-family evaluation."""

from dataclasses import replace

import numpy as np
import pytest

from toposc_lab.geometry import Geometry, GeometryEdge, square
from toposc_lab.geometry.serialization import geometry_from_bytes, geometry_to_bytes
from toposc_lab.models.chiral_p_wave import ChiralPWaveModel, ChiralPWaveParameters
from toposc_lab.research.embedded import (
    EmbeddedDomain,
    EmbeddedPhysicsProtocol,
    EmbeddedValidationEvaluator,
)
from toposc_lab.research.physics import PhysicsProtocol
from toposc_lab.research.validation_diagnostics import ValidationEvaluator


@pytest.mark.parametrize("disorder", [False, True])
def test_existing_square_and_direct_graph_equivalent(disorder):
    original = square(6, 6)
    graph = Geometry(
        n_sites=original.n_sites,
        coordinates=original.coordinates,
        edges=tuple(GeometryEdge(e.source, e.target) for e in original.edges),
        boundary_sites=original.boundary_sites,
    )
    graph = geometry_from_bytes(geometry_to_bytes(graph))
    parameters = ChiralPWaveParameters(hopping=1, chemical_potential=2, pairing=1)
    a = ChiralPWaveModel(original, parameters)
    b = ChiralPWaveModel(graph, parameters)
    np.testing.assert_array_equal(a.hamiltonian(), b.hamiltonian())
    domain = EmbeddedDomain(bounds=(0.0, 5.0, 0.0, 5.0))
    settings = {
        "disorder_widths": (3.0,),
        "disorder_seeds": (191001, 191002),
        "confirmation": False,
    }
    legacy = ValidationEvaluator(PhysicsProtocol(**settings))
    generic = EmbeddedValidationEvaluator(EmbeddedPhysicsProtocol(domain=domain, **settings))
    stage = legacy.plan()[1 if disorder else 0]
    old, new = legacy.evaluate(original, stage), generic.evaluate(graph, stage)
    for field in ("spectrum", "onsite_offsets", "localizer_gaps", "indices"):
        np.testing.assert_allclose(old[field], new[field], rtol=0, atol=1e-10)
    assert old["metrics"] == new["metrics"]
    assert old["majorana"] == new["majorana"]
    assert old["spatial"] == new["spatial"]
    assert old["boundary_window"] == new["boundary_window"]
    assert old["chern_marker"]["bulk_mean"] == pytest.approx(
        new["chern_marker"]["bulk_mean"], abs=1e-10
    )
    assert new["adapter_id"] != old["adapter_id"]


def test_free_coordinates_direction_symmetry_and_reversal():
    coords = np.array([[0.0, 0.0], [3.0, 0.0], [5.0, 2.0], [2.0, 5.0], [0.0, 4.0]])
    edges = tuple(GeometryEdge(i, (i + 1) % 5) for i in range(5))
    g = Geometry(n_sites=5, coordinates=coords, edges=edges, boundary_sites=frozenset(range(5)))
    p = ChiralPWaveParameters(hopping=1, chemical_potential=2, pairing=1)
    m = ChiralPWaveModel(g, p)
    h, delta = m.hamiltonian(), m.pairing_matrix()
    np.testing.assert_allclose(h, h.conj().T, atol=1e-14)
    np.testing.assert_allclose(delta, -delta.T, atol=1e-14)
    exchange = m.nambu_basis.particle_hole_operator
    np.testing.assert_allclose(exchange @ h.conj() @ exchange, -h, atol=1e-14)
    reversed_graph = replace(g, edges=tuple(GeometryEdge(e.target, e.source) for e in edges))
    np.testing.assert_array_equal(h, ChiralPWaveModel(reversed_graph, p).hamiltonian())
    np.testing.assert_allclose(g.distance(1, 2), np.sqrt(8))
    assert np.arctan2(*g.displacement_between(1, 2)[::-1]) == pytest.approx(np.pi / 4)


def test_domain_rejects_wrong_boundary_and_protocol_changes():
    d = EmbeddedDomain()
    d.validate(square(8, 8))
    with pytest.raises(ValueError, match="Boundary"):
        d.validate(replace(square(8, 8), boundary_sites=frozenset()))
    with pytest.raises(ValueError, match="frozen"):
        EmbeddedPhysicsProtocol(pairing=2)
    with pytest.raises(ValueError, match="interior"):
        EmbeddedDomain(bounds=(0.0, 2.0, 0.0, 2.0))


@pytest.mark.parametrize("family", ["rewired_square", "amorphous_planar", "constrained_embedded"])
def test_family_seed_reproducibility_and_common_constraints(family):
    from toposc_lab.geometry.generators.protocol import GeometryGenerationRequest
    from toposc_lab.research.embedded_cohort import RECIPES, canonical_sites, comparison_policy
    from toposc_lab.search.mutation_validity import validate_geometry_constraints

    request = GeometryGenerationRequest(seed=190001)
    a, b = (RECIPES[family].generate(request) for _ in range(2))
    assert geometry_to_bytes(a) == geometry_to_bytes(b)
    graph = canonical_sites(a)
    validate_geometry_constraints(graph, policy=comparison_policy()).raise_for_errors()
    EmbeddedDomain().validate(graph)
    with pytest.raises(ValueError):
        validate_geometry_constraints(
            replace(graph, edges=()), policy=comparison_policy()
        ).raise_for_errors()
    assert np.sum(EmbeddedDomain().areas(graph.coordinates)) == pytest.approx(64)


def test_canonical_order_does_not_discard_modified_displacement():
    from toposc_lab.research.embedded_cohort import canonical_sites

    graph = square(8, 8)
    edge = replace(graph.edges[0], displacement=(42.0, 0.0))
    with pytest.raises(ValueError, match="displacement"):
        canonical_sites(replace(graph, edges=(edge, *graph.edges[1:])))


def test_contrast_uses_valid_matching_seeds_per_metric():
    from toposc_lab.research.embedded_reporting import METRICS, paired_contrast

    rows = [
        dict(candidate=c, kind="disorder", width=6.0, seed=s, valid=v, **dict.fromkeys(METRICS, q))
        for c, s, v, q in [
            ("regular", 1, True, 0.2),
            ("regular", 2, False, 100),
            ("other", 1, True, 0.5),
            ("other", 2, True, 0.7),
        ]
    ]
    difference = paired_contrast(rows, "other", 6.0)
    assert difference["quality"] == pytest.approx(0.3)
    assert difference["quality_paired_seeds"] == [1]


def test_embedded_study_resume_and_complete_raw_export(tmp_path, monkeypatch):
    import json

    from toposc_lab.research.embedded_cohort import prepare_cohort, validate_cohort
    from toposc_lab.research.embedded_study import EmbeddedStudy, settings_for
    from toposc_lab.research.storage import ResearchStore
    from toposc_lab.research.validation import THREAD_VARIABLES

    for name in THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")
    cohort = prepare_cohort(per_family=1)
    validate_cohort(json.loads(json.dumps(cohort)))
    settings = settings_for(cohort)
    settings.update(widths=[6.0], seeds=[191001, 191002], exact_budget=12)
    directory = EmbeddedStudy.create_study(tmp_path / "study", settings, cohort)
    assert EmbeddedStudy(directory).run(max_stages=2)["status"] == "PAUSED"
    store = ResearchStore(directory)
    before = store.get("exact_result", "regular:clean")
    assert EmbeddedStudy(directory).run()["status"] == "COMPLETED"
    assert len(store.attempts()) == store.count("exact_result") == 12
    assert store.get("exact_result", "regular:clean") == before
    assert EmbeddedStudy(directory).run()["status"] == "COMPLETED"
    assert len(store.attempts()) == 12
    summary = json.loads((directory / "reports/summary.json").read_text())
    assert summary["records"] == 12 and summary["invalid"] == 0
    assert summary["state"]["status"] == "COMPLETED"
    raw = (directory / "reports/realizations.jsonl").read_text().splitlines()
    assert len(raw) == 12
