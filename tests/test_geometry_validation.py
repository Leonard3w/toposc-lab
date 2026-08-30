from __future__ import annotations

from fractions import Fraction
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from toposc_lab.geometry import (
    BUILTIN_GEOMETRY_GENERATORS,
    Geometry,
    GeometryEdge,
    GeometryFace,
    GeometryValidationError,
    GeometryValidationIssue,
    chain,
    validate_geometry,
)


_BUILTIN_VALIDATION_REQUESTS: dict[str, tuple[dict[str, Any], int | None]] = {
    "ammann_beenker_patch": ({"radius": 2.0}, None),
    "artificial_rule_graph": (
        {"iterations": 1, "displacement_rules": [[-1], [1]]},
        None,
    ),
    "body_centered_cubic": ({"n_x": 2, "n_y": 2, "n_z": 2}, None),
    "cayley_tree": ({"coordination": 3, "shells": 2}, None),
    "chain": ({"n_sites": 3}, None),
    "coordinate_cutoff_graph": (
        {"coordinates": [[0.0], [1.0], [2.0]], "cutoff": 1.1},
        None,
    ),
    "cubic": ({"n_x": 2, "n_y": 2, "n_z": 2}, None),
    "fibonacci_chain": ({"order": 2}, None),
    "honeycomb": ({"n_x": 2, "n_y": 2}, None),
    "irregular_cluster": ({}, None),
    "k_nearest_neighbor_graph": (
        {"coordinates": [[0.0], [1.0], [2.0]], "k": 1},
        None,
    ),
    "kagome": ({"n_x": 2, "n_y": 2}, None),
    "menger_sponge": ({"order": 1}, None),
    "random_graph": ({"n_sites": 5, "edge_probability": 1.0}, 7),
    "random_regular_graph": ({"n_sites": 6, "degree": 2}, 7),
    "ring": ({"n_sites": 4}, None),
    "scale_free_graph": ({"n_sites": 6, "attachments_per_site": 2}, 7),
    "sierpinski_carpet": ({"order": 1}, None),
    "sierpinski_gasket": ({"order": 1}, None),
    "silver_mean_chain": ({"order": 2}, None),
    "small_world_network": (
        {
            "n_sites": 8,
            "neighbor_degree": 2,
            "rewiring_probability": 0.25,
        },
        7,
    ),
    "square": ({"n_x": 2, "n_y": 2}, None),
    "tree": ({"parents": [0, 0, 1]}, None),
    "triangular": ({"n_x": 2, "n_y": 2}, None),
}


def _issue_codes(geometry: Geometry, **options: Any) -> tuple[str, ...]:
    return tuple(issue.code for issue in validate_geometry(geometry, **options).issues)


def test_connected_geometry_has_clean_structured_report() -> None:
    geometry = chain(5)
    report = validate_geometry(geometry)

    assert report.is_valid
    assert report.is_connected
    assert report.issues == ()
    assert report.errors == ()
    assert report.warnings == ()
    assert report.connected_components == (frozenset(range(5)),)
    assert report.require_connected is False
    report.raise_for_errors()


def test_every_builtin_generator_produces_metadata_valid_geometry() -> None:
    generators = BUILTIN_GEOMETRY_GENERATORS.generators()
    assert set(_BUILTIN_VALIDATION_REQUESTS) == {
        generator.key for generator in generators
    }

    for generator in generators:
        parameters, seed = _BUILTIN_VALIDATION_REQUESTS[generator.key]
        geometry = BUILTIN_GEOMETRY_GENERATORS.generate(
            generator.key,
            parameters=parameters,
            seed=seed,
        )
        report = validate_geometry(geometry)
        assert report.is_valid, (generator.key, report.errors)


def test_disconnected_components_are_canonical_and_nonfatal_by_default() -> None:
    geometry = Geometry(
        n_sites=6,
        edges=(GeometryEdge(4, 2), GeometryEdge(1, 0)),
    )
    report = validate_geometry(geometry)

    assert report.is_valid
    assert not report.is_connected
    assert report.connected_components == (
        frozenset({0, 1}),
        frozenset({2, 4}),
        frozenset({3}),
        frozenset({5}),
    )
    assert tuple((issue.code, issue.severity) for issue in report.issues) == (
        ("disconnected_components", "warning"),
    )


def test_connectivity_is_an_error_only_when_explicitly_required() -> None:
    geometry = Geometry(n_sites=2)
    report = validate_geometry(geometry, require_connected=True)

    assert not report.is_valid
    assert not report.is_connected
    assert report.warnings == ()
    assert tuple(issue.code for issue in report.errors) == (
        "disconnected_components",
    )
    with pytest.raises(GeometryValidationError, match="disconnected_components") as caught:
        report.raise_for_errors()
    assert caught.value.report is report


def test_singleton_geometry_is_connected_without_edges() -> None:
    report = validate_geometry(Geometry(n_sites=1), require_connected=True)

    assert report.is_valid
    assert report.is_connected
    assert report.connected_components == (frozenset({0}),)


def test_pipeline_defensively_reports_invalid_edge_indices() -> None:
    geometry = Geometry(n_sites=2, edges=(GeometryEdge(0, 1),))
    object.__setattr__(geometry.edges[0], "target", 2)
    report = validate_geometry(geometry)

    assert "invalid_site_index" in _issue_codes(geometry)
    issue = next(issue for issue in report.issues if issue.code == "invalid_site_index")
    assert issue.severity == "error"
    assert issue.path == "edges[0].target"
    assert not report.is_valid


def test_pipeline_defensively_reports_duplicate_undirected_edges() -> None:
    geometry = Geometry(
        n_sites=3,
        edges=(GeometryEdge(0, 1), GeometryEdge(1, 2)),
    )
    object.__setattr__(geometry.edges[1], "target", 0)

    assert "duplicate_edge" in _issue_codes(geometry)


def test_pipeline_defensively_reports_self_loops() -> None:
    geometry = Geometry(n_sites=2, edges=(GeometryEdge(0, 1),))
    object.__setattr__(geometry.edges[0], "target", 0)

    assert "self_loop" in _issue_codes(geometry)


def test_geometry_constructor_still_rejects_obvious_bad_edges() -> None:
    with pytest.raises(ValueError, match="self-edges"):
        GeometryEdge(0, 0)
    with pytest.raises(ValueError, match="duplicate edge"):
        Geometry(
            n_sites=2,
            edges=(GeometryEdge(0, 1), GeometryEdge(1, 0)),
        )
    with pytest.raises(ValueError, match="outside the geometry"):
        Geometry(n_sites=2, edges=(GeometryEdge(0, 2),))


def test_boundary_and_face_indices_are_checked_defensively() -> None:
    geometry = Geometry(
        n_sites=3,
        edges=(GeometryEdge(0, 1), GeometryEdge(1, 2), GeometryEdge(2, 0)),
        boundary_sites=frozenset({0}),
        faces=(GeometryFace((0, 1, 2)),),
    )
    object.__setattr__(geometry, "boundary_sites", frozenset({3}))
    object.__setattr__(geometry.faces[0], "sites", (0, 1, 4))

    issues = validate_geometry(geometry).issues
    paths = {issue.path for issue in issues if issue.code == "invalid_site_index"}
    assert paths == {"boundary_sites[3]", "faces[0].sites[2]"}


def test_dimensionless_abstract_graph_is_valid() -> None:
    geometry = Geometry(n_sites=2, edges=(GeometryEdge(0, 1),))

    report = validate_geometry(geometry)
    assert report.is_valid
    assert geometry.embedding_dimension is None
    assert geometry.coordinates is None


def test_coordinate_dimension_mismatch_is_reported_defensively() -> None:
    geometry = Geometry(
        n_sites=2,
        edges=(GeometryEdge(0, 1, displacement=(1.0, 0.0)),),
        coordinates=np.asarray(((0.0, 0.0), (1.0, 0.0))),
    )
    object.__setattr__(geometry, "embedding_dimension", 3)

    codes = _issue_codes(geometry)
    assert "inconsistent_embedding_dimension" in codes


def test_mixed_edge_displacement_dimensions_are_reported_defensively() -> None:
    geometry = Geometry(
        n_sites=3,
        edges=(
            GeometryEdge(0, 1, displacement=(1.0, 0.0)),
            GeometryEdge(1, 2, displacement=(1.0, 0.0)),
        ),
        embedding_dimension=2,
    )
    object.__setattr__(geometry.edges[1], "displacement", (1.0, 0.0, 0.0))

    assert "inconsistent_displacement_dimensions" in _issue_codes(geometry)


@pytest.mark.parametrize(
    "coordinates",
    (
        np.asarray((0.0, 1.0)),
        np.zeros((2, 0)),
        np.asarray(((0.0,), (np.nan,))),
        np.asarray(((0.0,), (np.inf,))),
        np.asarray(((0.0,), (1.0,), (2.0,))),
        np.asarray(((0.0 + 0.0j,), (1.0 + 1.0j,))),
    ),
)
def test_bad_coordinates_are_reported_defensively(coordinates: np.ndarray) -> None:
    geometry = Geometry(n_sites=2)
    object.__setattr__(geometry, "coordinates", coordinates)

    assert "bad_coordinates" in _issue_codes(geometry)


def test_ragged_coordinates_are_reported_without_validator_failure() -> None:
    geometry = Geometry(n_sites=2)
    object.__setattr__(geometry, "coordinates", [[0.0], [1.0, 2.0]])

    assert "bad_coordinates" in _issue_codes(geometry)


def test_ragged_edge_displacement_is_reported_without_validator_failure() -> None:
    geometry = Geometry(n_sites=2, edges=(GeometryEdge(0, 1),))
    object.__setattr__(geometry.edges[0], "displacement", [[1.0], [2.0, 3.0]])

    assert "invalid_edge_displacement" in _issue_codes(geometry)


def test_constructor_explicitly_rejects_complex_coordinates() -> None:
    with pytest.raises(ValueError, match="complex coordinates"):
        Geometry(
            n_sites=2,
            coordinates=np.asarray(((0.0 + 0.0j,), (1.0 + 1.0j,))),
        )


def test_finite_nested_metadata_and_typed_arrays_are_valid() -> None:
    geometry = Geometry(
        n_sites=2,
        edges=(
            GeometryEdge(
                0,
                1,
                metadata={"phase": 1.0 + 2.0j, "labels": ["a", "b"]},
            ),
        ),
        metadata={
            "none": None,
            "flag": True,
            "count": 3,
            "value": 0.5,
            "payload": b"exact",
            "nested": {"array": np.asarray((1.0, 2.0))},
        },
    )

    assert validate_geometry(geometry).is_valid


@pytest.mark.parametrize("bad_value", (np.nan, np.inf, -np.inf, complex(np.inf, 0.0)))
def test_nonfinite_metadata_is_invalid_at_every_nested_path(bad_value: object) -> None:
    geometry = Geometry(n_sites=1, metadata={"nested": {"bad": bad_value}})
    report = validate_geometry(geometry)

    assert not report.is_valid
    issue = next(issue for issue in report.issues if issue.code == "nonfinite_metadata")
    assert issue.path == "metadata['nested']['bad']"


def test_nonfinite_metadata_array_is_invalid() -> None:
    geometry = Geometry(n_sites=1, metadata={"values": np.asarray((1.0, np.nan))})

    assert "nonfinite_metadata" in _issue_codes(geometry)


def test_metadata_mapping_keys_must_be_strings() -> None:
    geometry = Geometry(n_sites=1, metadata={1: "bad"})  # type: ignore[dict-item]
    report = validate_geometry(geometry)

    issue = next(issue for issue in report.issues if issue.code == "invalid_metadata_key")
    assert issue.path == "metadata[1]"


def test_metadata_root_must_remain_a_mapping() -> None:
    geometry = Geometry(n_sites=1)
    object.__setattr__(geometry, "metadata", ("not", "a", "mapping"))

    assert _issue_codes(geometry) == ("invalid_metadata_container",)


@pytest.mark.parametrize(
    "bad_value",
    (
        object(),
        frozenset({1, 2}),
        np.asarray((object(),), dtype=object),
        np.asarray(("2026-01-01",), dtype="datetime64[D]"),
    ),
)
def test_unsupported_metadata_values_are_invalid(bad_value: object) -> None:
    geometry = Geometry(n_sites=1, metadata={"bad": bad_value})

    codes = _issue_codes(geometry)
    assert {"invalid_metadata_type", "invalid_metadata_array"} & set(codes)


@pytest.mark.parametrize(
    "bad_value",
    (
        Fraction(1, 3),
        range(3),
        np.datetime64("2026-08-30"),
    ),
)
def test_metadata_rejects_values_without_exact_archive_semantics(
    bad_value: object,
) -> None:
    geometry = Geometry(n_sites=1, metadata={"bad": bad_value})

    assert {
        "invalid_metadata_type",
        "invalid_metadata_scalar",
    } & set(_issue_codes(geometry))


def test_geometry_edge_and_face_metadata_are_both_validated() -> None:
    geometry = Geometry(
        n_sites=3,
        edges=(
            GeometryEdge(0, 1, metadata={"bad": object()}),
            GeometryEdge(1, 2),
            GeometryEdge(2, 0),
        ),
        faces=(GeometryFace((0, 1, 2), metadata={"bad": np.nan}),),
    )
    issues = validate_geometry(geometry).issues

    assert any(
        issue.code == "invalid_metadata_type"
        and issue.path == "edges[0].metadata['bad']"
        for issue in issues
    )
    assert any(
        issue.code == "nonfinite_metadata"
        and issue.path == "faces[0].metadata['bad']"
        for issue in issues
    )


def test_cyclic_metadata_is_reported_without_recursing_forever() -> None:
    geometry = Geometry(n_sites=1)
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    object.__setattr__(geometry, "metadata", MappingProxyType(cyclic))

    assert "cyclic_metadata" in _issue_codes(geometry)


def test_issue_and_report_data_are_immutable() -> None:
    report = validate_geometry(Geometry(n_sites=2))

    assert isinstance(report.issues, tuple)
    assert all(isinstance(issue, GeometryValidationIssue) for issue in report.issues)
    with pytest.raises(AttributeError):
        report.connected_components[0].add(1)  # type: ignore[attr-defined]


@pytest.mark.parametrize("geometry", (None, object(), "geometry"))
def test_validator_requires_geometry_instance(geometry: object) -> None:
    with pytest.raises(TypeError, match="Geometry instance"):
        validate_geometry(geometry)  # type: ignore[arg-type]


@pytest.mark.parametrize("require_connected", (0, 1, None, "yes"))
def test_validator_requires_boolean_connectivity_policy(
    require_connected: object,
) -> None:
    with pytest.raises(TypeError, match="must be a boolean"):
        validate_geometry(
            Geometry(n_sites=1),
            require_connected=require_connected,  # type: ignore[arg-type]
        )
