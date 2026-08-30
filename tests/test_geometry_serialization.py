from __future__ import annotations

import json
import zipfile
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import toposc_lab.geometry.serialization as geometry_serialization

from toposc_lab.geometry import (
    GEOMETRY_ARCHIVE_FORMAT,
    GEOMETRY_ARCHIVE_SCHEMA_VERSION,
    Geometry,
    GeometryBoundaryComponent,
    GeometryDimension,
    GeometryEdge,
    GeometryFace,
    GeometrySerializationError,
    GeometryValidationError,
    geometry_from_bytes,
    geometry_to_bytes,
    load_geometry,
    save_geometry,
    tree,
)


_MANIFEST_KEY = "__toposc_lab_geometry_manifest__"


def _rich_geometry() -> Geometry:
    coordinates = np.asarray(
        (
            (-0.0, np.nextafter(0.0, 1.0)),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        ),
        dtype=np.float64,
    )
    return Geometry(
        n_sites=4,
        edges=(
            GeometryEdge(
                0,
                1,
                edge_type="bottom",
                boundary_crossing=True,
                displacement=(-0.0, 0.5),
                metadata={
                    "phase": complex(-0.0, 0.25),
                    "weights": np.asarray((1, 2), dtype=np.int16),
                },
            ),
            GeometryEdge(1, 2, displacement=(0.0, 1.0)),
            GeometryEdge(2, 3, displacement=(-1.0, 0.0)),
            GeometryEdge(3, 0, displacement=(0.0, -1.0)),
        ),
        coordinates=coordinates,
        boundary_sites=frozenset({0, 1, 2, 3}),
        boundary_components=(
            GeometryBoundaryComponent("outer", 0, frozenset({0, 1, 2, 3})),
            GeometryBoundaryComponent("hole", 0, frozenset({1, 2})),
        ),
        site_types=("A", "B", None, "edge"),
        dimension_records=(
            GeometryDimension(
                "lattice",
                2.0,
                "finite_geometry",
                "test construction",
                exact=True,
            ),
        ),
        metadata={
            "none": None,
            "boolean": True,
            "unicode": "Quasikristall Ω",
            "bytes": b"\x00\xffexact",
            "large_integer": 2**200,
            "negative_zero": -0.0,
            "complex": complex(0.125, -0.0),
            "tuple": ("nested", {"value": 3}),
            "int_array": np.asarray([[1, 2], [3, 4]], dtype=np.int32),
            "float_array": np.asarray([-0.0, 1.25], dtype=np.float32),
            "complex_array": np.asarray([1.0 + 2.0j], dtype=np.complex128),
            "unicode_array": np.asarray(["α", "β"], dtype="U1"),
            "bytes_array": np.asarray([b"a", b"bc"], dtype="S2"),
            "big_endian_array": np.asarray([1, 2], dtype=">i2"),
            "numpy_integer": np.int16(-7),
            "numpy_real": np.float32(-0.0),
            "numpy_complex": np.complex64(1.0 - 2.0j),
            "numpy_boolean": np.bool_(True),
            "numpy_string": np.str_("typed"),
        },
        faces=(
            GeometryFace(
                (0, 1, 2, 3),
                face_type="square",
                metadata={"normal": np.asarray((0.0, 0.0, 1.0))},
            ),
        ),
    )


def _assert_metadata_exact(actual: Any, expected: Any) -> None:
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.tobytes(order="C") == expected.tobytes(order="C")
        return
    if isinstance(expected, np.generic):
        assert isinstance(actual, np.generic)
        assert actual.dtype == expected.dtype
        assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
        return
    if isinstance(expected, Mapping):
        assert isinstance(actual, Mapping)
        assert tuple(actual) == tuple(expected)
        for key in expected:
            _assert_metadata_exact(actual[key], expected[key])
        return
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_metadata_exact(actual_item, expected_item)
        return
    assert type(actual) is type(expected)
    if isinstance(expected, float):
        assert actual.hex() == expected.hex()
    elif isinstance(expected, complex):
        assert actual.real.hex() == expected.real.hex()
        assert actual.imag.hex() == expected.imag.hex()
    else:
        assert actual == expected


def _assert_geometry_exact(actual: Geometry, expected: Geometry) -> None:
    assert actual.n_sites == expected.n_sites
    assert actual.embedding_dimension == expected.embedding_dimension
    assert actual.boundary_sites == expected.boundary_sites
    assert actual.boundary_components == expected.boundary_components
    assert actual.site_types == expected.site_types
    assert actual.dimension_records == expected.dimension_records
    assert actual.rooted_tree == expected.rooted_tree
    if expected.coordinates is None:
        assert actual.coordinates is None
    else:
        assert actual.coordinates is not None
        assert actual.coordinates.dtype == expected.coordinates.dtype
        assert actual.coordinates.shape == expected.coordinates.shape
        assert actual.coordinates.tobytes() == expected.coordinates.tobytes()
    assert len(actual.edges) == len(expected.edges)
    for actual_edge, expected_edge in zip(actual.edges, expected.edges, strict=True):
        assert actual_edge.source == expected_edge.source
        assert actual_edge.target == expected_edge.target
        assert actual_edge.edge_type == expected_edge.edge_type
        assert actual_edge.boundary_crossing is expected_edge.boundary_crossing
        if expected_edge.displacement is None:
            assert actual_edge.displacement is None
        else:
            assert actual_edge.displacement is not None
            assert tuple(value.hex() for value in actual_edge.displacement) == tuple(
                value.hex() for value in expected_edge.displacement
            )
        _assert_metadata_exact(actual_edge.metadata, expected_edge.metadata)
    assert len(actual.faces) == len(expected.faces)
    for actual_face, expected_face in zip(actual.faces, expected.faces, strict=True):
        assert actual_face.sites == expected_face.sites
        assert actual_face.face_type == expected_face.face_type
        _assert_metadata_exact(actual_face.metadata, expected_face.metadata)
    _assert_metadata_exact(actual.metadata, expected.metadata)


def _unpack_archive(data: bytes) -> dict[str, np.ndarray]:
    with np.load(BytesIO(data), allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _pack_archive(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = BytesIO()
    np.savez_compressed(buffer, **arrays)  # type: ignore[arg-type]
    return buffer.getvalue()


def _manifest(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(arrays[_MANIFEST_KEY].tobytes().decode("utf-8")),
    )


def _replace_manifest(
    arrays: dict[str, np.ndarray],
    manifest: Mapping[str, Any],
) -> bytes:
    encoded = json.dumps(
        manifest,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    arrays[_MANIFEST_KEY] = np.frombuffer(encoded, dtype=np.uint8)
    return _pack_archive(arrays)


def test_bytes_round_trip_preserves_every_geometry_field_exactly() -> None:
    geometry = _rich_geometry()
    loaded = geometry_from_bytes(geometry_to_bytes(geometry))

    _assert_geometry_exact(loaded, geometry)
    assert loaded.coordinates is not None
    assert not loaded.coordinates.flags.writeable


def test_rooted_tree_round_trip_preserves_hierarchy() -> None:
    geometry = tree((0, 0, 1, 1, 3))
    loaded = geometry_from_bytes(geometry_to_bytes(geometry))

    _assert_geometry_exact(loaded, geometry)
    assert loaded.rooted_tree is not None
    assert loaded.rooted_tree.parents == (None, 0, 0, 1, 1, 3)


def test_disconnected_geometry_round_trip_remains_allowed() -> None:
    geometry = Geometry(n_sites=4, edges=(GeometryEdge(0, 1),))

    _assert_geometry_exact(geometry_from_bytes(geometry_to_bytes(geometry)), geometry)


def test_file_round_trip_appends_suffix_and_creates_parent_directory(
    tmp_path: Path,
) -> None:
    geometry = _rich_geometry()
    path = save_geometry(tmp_path / "nested" / "reference_geometry", geometry)
    loaded = load_geometry(path)

    assert path == tmp_path / "nested" / "reference_geometry.npz"
    assert path.is_file()
    _assert_geometry_exact(loaded, geometry)


def test_save_atomically_replaces_existing_geometry(tmp_path: Path) -> None:
    destination = tmp_path / "geometry.npz"
    save_geometry(destination, Geometry(n_sites=1))
    replacement = Geometry(n_sites=2, edges=(GeometryEdge(0, 1),))

    assert save_geometry(destination, replacement) == destination
    _assert_geometry_exact(load_geometry(destination), replacement)
    assert tuple(tmp_path.glob(".geometry.npz.*.tmp")) == ()


def test_failed_atomic_save_preserves_existing_file_and_removes_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = save_geometry(tmp_path / "geometry", Geometry(n_sites=1))
    original_bytes = destination.read_bytes()

    def fail_write(handle: Any, payload: Any) -> None:
        del handle, payload
        raise OSError("simulated write failure")

    monkeypatch.setattr(geometry_serialization, "_write_archive", fail_write)
    with pytest.raises(OSError, match="simulated write failure"):
        save_geometry(destination, Geometry(n_sites=2, edges=(GeometryEdge(0, 1),)))

    assert destination.read_bytes() == original_bytes
    assert tuple(tmp_path.glob(".geometry.npz.*.tmp")) == ()


def test_archive_manifest_has_stable_identity_and_version() -> None:
    arrays = _unpack_archive(geometry_to_bytes(Geometry(n_sites=1)))
    manifest = _manifest(arrays)

    assert manifest["format"] == GEOMETRY_ARCHIVE_FORMAT
    assert manifest["schema_version"] == GEOMETRY_ARCHIVE_SCHEMA_VERSION
    assert set(manifest) == {"format", "schema_version", "geometry"}
    assert arrays[_MANIFEST_KEY].dtype == np.uint8


def test_invalid_geometry_metadata_is_rejected_before_writing(tmp_path: Path) -> None:
    geometry = Geometry(n_sites=1, metadata={"bad": object()})

    with pytest.raises(GeometryValidationError, match="invalid_metadata_type"):
        geometry_to_bytes(geometry)
    with pytest.raises(GeometryValidationError, match="invalid_metadata_type"):
        save_geometry(tmp_path / "invalid", geometry)
    assert not (tmp_path / "invalid.npz").exists()


def test_boundary_crossing_must_be_a_boolean() -> None:
    with pytest.raises(TypeError, match="boundary_crossing must be a boolean"):
        GeometryEdge(0, 1, boundary_crossing=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("function", (save_geometry, load_geometry))
def test_file_apis_reject_non_npz_suffix(function: Any, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"\.npz suffix"):
        if function is save_geometry:
            function(tmp_path / "geometry.json", Geometry(n_sites=1))
        else:
            function(tmp_path / "geometry.json")


def test_load_reports_missing_file_after_suffix_normalization(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"missing\.npz"):
        load_geometry(tmp_path / "missing")


def test_bytes_loader_requires_bytes() -> None:
    with pytest.raises(TypeError, match="data must be bytes"):
        geometry_from_bytes(bytearray(geometry_to_bytes(Geometry(n_sites=1))))  # type: ignore[arg-type]


@pytest.mark.parametrize("data", (b"", b"not a zip archive", b"PK\x03\x04truncated"))
def test_loader_rejects_non_archives_and_truncated_archives(data: bytes) -> None:
    with pytest.raises(GeometrySerializationError):
        geometry_from_bytes(data)


def test_loader_rejects_archive_without_manifest() -> None:
    data = _pack_archive({"coordinates": np.zeros((1, 1))})

    with pytest.raises(GeometrySerializationError, match="no manifest"):
        geometry_from_bytes(data)


def test_loader_rejects_invalid_manifest_dtype_and_json() -> None:
    arrays = _unpack_archive(geometry_to_bytes(Geometry(n_sites=1)))
    arrays[_MANIFEST_KEY] = np.asarray("not uint8")
    with pytest.raises(GeometrySerializationError, match="uint8 vector"):
        geometry_from_bytes(_pack_archive(arrays))

    arrays[_MANIFEST_KEY] = np.frombuffer(b"{", dtype=np.uint8)
    with pytest.raises(GeometrySerializationError, match="valid UTF-8 JSON"):
        geometry_from_bytes(_pack_archive(arrays))


def test_loader_rejects_duplicate_json_keys_and_nonfinite_constants() -> None:
    arrays = _unpack_archive(geometry_to_bytes(Geometry(n_sites=1)))
    manifest = _manifest(arrays)
    geometry_json = json.dumps(manifest["geometry"], separators=(",", ":"))
    duplicate_json = (
        '{"format":"toposc_lab_geometry","format":"duplicate",'
        f'"schema_version":1,"geometry":{geometry_json}}}'
    ).encode()
    arrays[_MANIFEST_KEY] = np.frombuffer(duplicate_json, dtype=np.uint8)
    with pytest.raises(GeometrySerializationError, match="duplicate JSON key"):
        geometry_from_bytes(_pack_archive(arrays))

    nonfinite_json = (
        '{"format":"toposc_lab_geometry","schema_version":NaN,'
        f'"geometry":{geometry_json}}}'
    ).encode()
    arrays[_MANIFEST_KEY] = np.frombuffer(nonfinite_json, dtype=np.uint8)
    with pytest.raises(GeometrySerializationError, match="non-finite JSON constant"):
        geometry_from_bytes(_pack_archive(arrays))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("format", "other_geometry", "archive format"),
        ("schema_version", 2, "schema version 2"),
    ),
)
def test_loader_rejects_wrong_format_or_schema_version(
    field: str,
    value: object,
    message: str,
) -> None:
    arrays = _unpack_archive(geometry_to_bytes(Geometry(n_sites=1)))
    manifest = _manifest(arrays)
    manifest[field] = value

    with pytest.raises(GeometrySerializationError, match=message):
        geometry_from_bytes(_replace_manifest(arrays, manifest))


def test_loader_rejects_missing_and_extra_manifest_fields() -> None:
    arrays = _unpack_archive(geometry_to_bytes(Geometry(n_sites=1)))
    manifest = _manifest(arrays)
    del manifest["geometry"]
    with pytest.raises(GeometrySerializationError, match="invalid fields"):
        geometry_from_bytes(_replace_manifest(arrays, manifest))

    arrays = _unpack_archive(geometry_to_bytes(Geometry(n_sites=1)))
    manifest = _manifest(arrays)
    manifest["unexpected"] = True
    with pytest.raises(GeometrySerializationError, match="invalid fields"):
        geometry_from_bytes(_replace_manifest(arrays, manifest))


def test_loader_rejects_unreferenced_and_missing_archive_arrays() -> None:
    arrays = _unpack_archive(geometry_to_bytes(Geometry(n_sites=1)))
    arrays["unexpected"] = np.asarray((1, 2, 3))
    with pytest.raises(GeometrySerializationError, match="unexpected arrays"):
        geometry_from_bytes(_pack_archive(arrays))

    geometry = Geometry(n_sites=1, coordinates=np.zeros((1, 2)))
    arrays = _unpack_archive(geometry_to_bytes(geometry))
    del arrays["coordinates"]
    with pytest.raises(GeometrySerializationError, match="missing archive array"):
        geometry_from_bytes(_pack_archive(arrays))


def test_loader_never_enables_pickle_for_referenced_object_array() -> None:
    geometry = Geometry(n_sites=1, metadata={"array": np.asarray((1, 2))})
    arrays = _unpack_archive(geometry_to_bytes(geometry))
    manifest = _manifest(arrays)
    array_name = manifest["geometry"]["metadata"]["items"][0][1]["array"]
    arrays[array_name] = np.asarray([{"unsafe": True}], dtype=object)

    with pytest.raises(GeometrySerializationError, match="could not load archive array"):
        geometry_from_bytes(_pack_archive(arrays))


def test_loader_rejects_inconsistent_geometry_payload() -> None:
    geometry = Geometry(
        n_sites=2,
        edges=(GeometryEdge(0, 1),),
        coordinates=np.zeros((2, 1)),
    )
    arrays = _unpack_archive(geometry_to_bytes(geometry))
    manifest = _manifest(arrays)
    manifest["geometry"]["n_sites"] = 3

    with pytest.raises(GeometrySerializationError, match="inconsistent geometry data"):
        geometry_from_bytes(_replace_manifest(arrays, manifest))


def test_loader_rejects_path_like_archive_members() -> None:
    valid_data = geometry_to_bytes(Geometry(n_sites=1))
    source = BytesIO(valid_data)
    output = BytesIO()
    with zipfile.ZipFile(source, "r") as original, zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as modified:
        for member in original.infolist():
            modified.writestr(member.filename, original.read(member.filename))
        modified.writestr("../unexpected.npy", b"not an array")

    with pytest.raises(GeometrySerializationError, match="invalid geometry archive member"):
        geometry_from_bytes(output.getvalue())


def test_archive_load_limits_accept_exact_values_and_reject_next_smaller() -> None:
    data = geometry_to_bytes(_rich_geometry())
    with zipfile.ZipFile(BytesIO(data), "r") as archive:
        uncompressed_bytes = sum(member.file_size for member in archive.infolist())
        members = len(archive.infolist())

    _assert_geometry_exact(
        geometry_from_bytes(
            data,
            max_archive_bytes=len(data),
            max_uncompressed_bytes=uncompressed_bytes,
            max_members=members,
        ),
        _rich_geometry(),
    )
    with pytest.raises(GeometrySerializationError, match="max_archive_bytes"):
        geometry_from_bytes(data, max_archive_bytes=len(data) - 1)
    with pytest.raises(GeometrySerializationError, match="max_uncompressed_bytes"):
        geometry_from_bytes(data, max_uncompressed_bytes=uncompressed_bytes - 1)
    with pytest.raises(GeometrySerializationError, match="max_members"):
        geometry_from_bytes(data, max_members=members - 1)


@pytest.mark.parametrize(
    "options",
    (
        {"max_archive_bytes": 0},
        {"max_uncompressed_bytes": -1},
        {"max_members": 0},
    ),
)
def test_loader_rejects_nonpositive_limits(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        geometry_from_bytes(b"unused", **options)


@pytest.mark.parametrize(
    "options",
    (
        {"max_archive_bytes": True},
        {"max_uncompressed_bytes": 1.5},
        {"max_members": "10"},
    ),
)
def test_loader_rejects_noninteger_limits(options: dict[str, Any]) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        geometry_from_bytes(b"unused", **options)
