"""Exact, pickle-free persistence for model-independent geometries."""

from __future__ import annotations

import base64
import json
import os
import re
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from io import BytesIO
from numbers import Integral
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from toposc_lab.geometry.base import (
    Geometry,
    GeometryBoundaryComponent,
    GeometryDimension,
    GeometryEdge,
    GeometryFace,
    RootedTreeStructure,
)
from toposc_lab.geometry.validation import validate_geometry

GEOMETRY_ARCHIVE_FORMAT = "toposc_lab_geometry"
GEOMETRY_ARCHIVE_SCHEMA_VERSION = 1
DEFAULT_GEOMETRY_ARCHIVE_MAX_BYTES = 512 * 1024 * 1024
DEFAULT_GEOMETRY_ARCHIVE_MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_GEOMETRY_ARCHIVE_MAX_MEMBERS = 100_000

_MANIFEST_KEY = "__toposc_lab_geometry_manifest__"
_MANIFEST_MEMBER = f"{_MANIFEST_KEY}.npy"
_COORDINATES_KEY = "coordinates"
_SUPPORTED_ARRAY_KINDS = frozenset("biufcSU")
_ARRAY_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INTEGER_PATTERN = re.compile(r"^(?:0|-?[1-9][0-9]*)$")
_MAX_METADATA_DEPTH = 100


class GeometrySerializationError(ValueError):
    """A geometry archive is malformed, unsupported, or unsafe to load."""


@dataclass(slots=True)
class _ArrayStore:
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    next_metadata_index: int = 0

    def add_metadata_array(self, values: np.ndarray) -> str:
        name = f"metadata_array_{self.next_metadata_index:08d}"
        self.next_metadata_index += 1
        self.arrays[name] = np.asarray(values)
        return name


@dataclass(slots=True)
class _ArchiveReader:
    archive: Any
    available_arrays: frozenset[str]
    used_arrays: set[str] = field(default_factory=set)

    def array(self, name: object, *, path: str) -> np.ndarray:
        if not isinstance(name, str) or _ARRAY_KEY_PATTERN.fullmatch(name) is None:
            raise GeometrySerializationError(f"{path} has an invalid array reference")
        if name == _MANIFEST_KEY or name not in self.available_arrays:
            raise GeometrySerializationError(
                f"{path} references missing archive array {name!r}"
            )
        if name in self.used_arrays:
            raise GeometrySerializationError(
                f"archive array {name!r} is referenced more than once"
            )
        try:
            values = np.asarray(self.archive[name])
        except (OSError, TypeError, ValueError) as error:
            raise GeometrySerializationError(
                f"could not load archive array {name!r}"
            ) from error
        if values.dtype.hasobject:
            raise GeometrySerializationError(
                f"archive array {name!r} must not use object dtype"
            )
        self.used_arrays.add(name)
        return values.copy()


def save_geometry(path: str | Path, geometry: Geometry) -> Path:
    """Atomically save ``geometry`` as a compressed, versioned ``.npz`` archive."""
    destination = _normalize_geometry_path(path)
    payload = _build_archive_payload(geometry)
    destination.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            _write_archive(handle, payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def load_geometry(
    path: str | Path,
    *,
    max_archive_bytes: int = DEFAULT_GEOMETRY_ARCHIVE_MAX_BYTES,
    max_uncompressed_bytes: int = DEFAULT_GEOMETRY_ARCHIVE_MAX_UNCOMPRESSED_BYTES,
    max_members: int = DEFAULT_GEOMETRY_ARCHIVE_MAX_MEMBERS,
) -> Geometry:
    """Load a geometry archive without enabling NumPy pickle support."""
    source = _normalize_geometry_path(path)
    if not source.is_file():
        raise FileNotFoundError(f"geometry file does not exist: {source}")
    limits = _validated_load_limits(
        max_archive_bytes=max_archive_bytes,
        max_uncompressed_bytes=max_uncompressed_bytes,
        max_members=max_members,
    )
    try:
        with source.open("rb") as handle:
            if os.fstat(handle.fileno()).st_size > limits[0]:
                raise GeometrySerializationError(
                    f"geometry archive exceeds max_archive_bytes={limits[0]}"
                )
            _validate_zip_container(
                handle,
                max_uncompressed_bytes=limits[1],
                max_members=limits[2],
            )
            handle.seek(0)
            with np.load(handle, allow_pickle=False) as archive:
                return _geometry_from_archive(archive)
    except GeometrySerializationError:
        raise
    except (OSError, TypeError, ValueError, zipfile.BadZipFile) as error:
        raise GeometrySerializationError(
            f"could not load geometry file: {source}"
        ) from error


def geometry_to_bytes(geometry: Geometry) -> bytes:
    """Serialize one geometry for downloads or non-filesystem storage backends."""
    payload = _build_archive_payload(geometry)
    buffer = BytesIO()
    _write_archive(buffer, payload)
    return buffer.getvalue()


def geometry_from_bytes(
    data: bytes,
    *,
    max_archive_bytes: int = DEFAULT_GEOMETRY_ARCHIVE_MAX_BYTES,
    max_uncompressed_bytes: int = DEFAULT_GEOMETRY_ARCHIVE_MAX_UNCOMPRESSED_BYTES,
    max_members: int = DEFAULT_GEOMETRY_ARCHIVE_MAX_MEMBERS,
) -> Geometry:
    """Load a geometry from in-memory archive bytes with the file safety checks."""
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    limits = _validated_load_limits(
        max_archive_bytes=max_archive_bytes,
        max_uncompressed_bytes=max_uncompressed_bytes,
        max_members=max_members,
    )
    if len(data) > limits[0]:
        raise GeometrySerializationError(
            f"geometry archive exceeds max_archive_bytes={limits[0]}"
        )

    buffer = BytesIO(data)
    try:
        _validate_zip_container(
            buffer,
            max_uncompressed_bytes=limits[1],
            max_members=limits[2],
        )
        buffer.seek(0)
        with np.load(buffer, allow_pickle=False) as archive:
            return _geometry_from_archive(archive)
    except GeometrySerializationError:
        raise
    except (OSError, TypeError, ValueError, zipfile.BadZipFile) as error:
        raise GeometrySerializationError("could not load geometry bytes") from error


def _normalize_geometry_path(path: str | Path) -> Path:
    try:
        destination = Path(path)
    except TypeError as error:
        raise TypeError("path must be a string or pathlib.Path") from error
    if destination.suffix == "":
        return destination.with_suffix(".npz")
    if destination.suffix.lower() != ".npz":
        raise ValueError("geometry files must use the .npz suffix")
    return destination


def _validated_load_limits(
    *,
    max_archive_bytes: int,
    max_uncompressed_bytes: int,
    max_members: int,
) -> tuple[int, int, int]:
    return (
        _positive_integer(max_archive_bytes, name="max_archive_bytes"),
        _positive_integer(
            max_uncompressed_bytes,
            name="max_uncompressed_bytes",
        ),
        _positive_integer(max_members, name="max_members"),
    )


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _validate_zip_container(
    source: BinaryIO,
    *,
    max_uncompressed_bytes: int,
    max_members: int,
) -> None:
    source.seek(0)
    with zipfile.ZipFile(source, mode="r") as archive:
        members = archive.infolist()
        if len(members) > max_members:
            raise GeometrySerializationError(
                f"geometry archive exceeds max_members={max_members}"
            )
        names = [member.filename for member in members]
        if len(set(names)) != len(names):
            raise GeometrySerializationError("geometry archive has duplicate members")
        if _MANIFEST_MEMBER not in names:
            raise GeometrySerializationError("geometry archive has no manifest")
        total_size = 0
        for member in members:
            if (
                member.is_dir()
                or not member.filename.endswith(".npy")
                or "/" in member.filename
                or "\\" in member.filename
                or _ARRAY_KEY_PATTERN.fullmatch(member.filename[:-4]) is None
            ):
                raise GeometrySerializationError(
                    f"invalid geometry archive member {member.filename!r}"
                )
            if member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise GeometrySerializationError(
                    f"unsupported compression for member {member.filename!r}"
                )
            total_size += member.file_size
            if total_size > max_uncompressed_bytes:
                raise GeometrySerializationError(
                    "geometry archive exceeds "
                    f"max_uncompressed_bytes={max_uncompressed_bytes}"
                )


def _write_archive(handle: BinaryIO, payload: Mapping[str, np.ndarray]) -> None:
    np.savez_compressed(handle, **payload)  # type: ignore[arg-type]


def _build_archive_payload(geometry: Geometry) -> dict[str, np.ndarray]:
    if not isinstance(geometry, Geometry):
        raise TypeError("geometry must be a Geometry instance")
    validate_geometry(geometry).raise_for_errors()

    store = _ArrayStore()
    coordinates_reference: str | None = None
    if geometry.coordinates is not None:
        coordinates_reference = _COORDINATES_KEY
        store.arrays[coordinates_reference] = np.asarray(geometry.coordinates)

    geometry_record: dict[str, Any] = {
        "n_sites": geometry.n_sites,
        "edges": [_encode_edge(edge, store=store) for edge in geometry.edges],
        "coordinates": coordinates_reference,
        "embedding_dimension": geometry.embedding_dimension,
        "boundary_sites": sorted(geometry.boundary_sites),
        "boundary_components": [
            {
                "kind": component.kind,
                "component_index": component.component_index,
                "sites": sorted(component.sites),
            }
            for component in geometry.boundary_components
        ],
        "site_types": None if geometry.site_types is None else list(geometry.site_types),
        "dimension_records": [
            {
                "kind": record.kind,
                "value": record.value.hex(),
                "scope": record.scope,
                "method": record.method,
                "exact": record.exact,
            }
            for record in geometry.dimension_records
        ],
        "rooted_tree": _encode_rooted_tree(geometry.rooted_tree),
        "metadata": _encode_metadata_mapping(geometry.metadata, store=store),
        "faces": [_encode_face(face, store=store) for face in geometry.faces],
    }
    manifest = {
        "format": GEOMETRY_ARCHIVE_FORMAT,
        "schema_version": GEOMETRY_ARCHIVE_SCHEMA_VERSION,
        "geometry": geometry_record,
    }
    manifest_bytes = json.dumps(
        manifest,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        _MANIFEST_KEY: np.frombuffer(manifest_bytes, dtype=np.uint8),
        **store.arrays,
    }


def _encode_edge(edge: GeometryEdge, *, store: _ArrayStore) -> dict[str, Any]:
    return {
        "source": edge.source,
        "target": edge.target,
        "edge_type": edge.edge_type,
        "boundary_crossing": edge.boundary_crossing,
        "displacement": (
            None
            if edge.displacement is None
            else [value.hex() for value in edge.displacement]
        ),
        "metadata": _encode_metadata_mapping(edge.metadata, store=store),
    }


def _encode_face(face: GeometryFace, *, store: _ArrayStore) -> dict[str, Any]:
    return {
        "sites": list(face.sites),
        "face_type": face.face_type,
        "metadata": _encode_metadata_mapping(face.metadata, store=store),
    }


def _encode_rooted_tree(structure: RootedTreeStructure | None) -> dict[str, Any] | None:
    if structure is None:
        return None
    return {
        "root_site": structure.root_site,
        "parents": list(structure.parents),
    }


def _encode_metadata_mapping(
    values: Mapping[str, Any],
    *,
    store: _ArrayStore,
) -> dict[str, Any]:
    return {
        "type": "mapping",
        "items": [
            [key, _encode_metadata_value(value, store=store)]
            for key, value in values.items()
        ],
    }


def _encode_metadata_value(value: Any, *, store: _ArrayStore) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "array": store.add_metadata_array(value),
        }
    if isinstance(value, np.generic):
        return {
            "type": "numpy_scalar",
            "array": store.add_metadata_array(np.asarray(value)),
        }
    if value is None:
        return {"type": "none"}
    if type(value) is bool:
        return {"type": "boolean", "value": value}
    if type(value) is str:
        return {"type": "string", "value": value}
    if type(value) is bytes:
        return {
            "type": "bytes",
            "value": base64.b64encode(value).decode("ascii"),
        }
    if type(value) is int:
        return {"type": "integer", "value": str(value)}
    if type(value) is float:
        return {"type": "real", "value": value.hex()}
    if type(value) is complex:
        return {
            "type": "complex",
            "real": float(value.real).hex(),
            "imag": float(value.imag).hex(),
        }
    if isinstance(value, Mapping):
        return _encode_metadata_mapping(value, store=store)
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [_encode_metadata_value(item, store=store) for item in value],
        }
    raise GeometrySerializationError(
        f"unsupported validated metadata type {type(value).__name__}"
    )


def _geometry_from_archive(archive: Any) -> Geometry:
    files = frozenset(archive.files)
    if _MANIFEST_KEY not in files:
        raise GeometrySerializationError("geometry archive has no manifest")
    manifest_array = np.asarray(archive[_MANIFEST_KEY])
    if manifest_array.ndim != 1 or manifest_array.dtype != np.dtype(np.uint8):
        raise GeometrySerializationError("geometry manifest must be a uint8 vector")
    try:
        manifest = json.loads(
            manifest_array.tobytes().decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise GeometrySerializationError("geometry manifest is not valid UTF-8 JSON") from error

    reader = _ArchiveReader(
        archive=archive,
        available_arrays=files - {_MANIFEST_KEY},
    )
    geometry = _decode_manifest(manifest, reader=reader)
    unused_arrays = reader.available_arrays - reader.used_arrays
    if unused_arrays:
        raise GeometrySerializationError(
            f"geometry archive has unexpected arrays: {sorted(unused_arrays)!r}"
        )
    validate_geometry(geometry).raise_for_errors()
    return geometry


def _decode_manifest(manifest: Any, *, reader: _ArchiveReader) -> Geometry:
    root = _mapping(manifest, path="manifest")
    _exact_keys(root, {"format", "schema_version", "geometry"}, path="manifest")
    if root["format"] != GEOMETRY_ARCHIVE_FORMAT:
        raise GeometrySerializationError("unsupported geometry archive format")
    version = _integer(root["schema_version"], path="manifest.schema_version")
    if version != GEOMETRY_ARCHIVE_SCHEMA_VERSION:
        raise GeometrySerializationError(
            f"unsupported geometry schema version {version}"
        )
    record = _mapping(root["geometry"], path="manifest.geometry")
    expected_keys = {
        "n_sites",
        "edges",
        "coordinates",
        "embedding_dimension",
        "boundary_sites",
        "boundary_components",
        "site_types",
        "dimension_records",
        "rooted_tree",
        "metadata",
        "faces",
    }
    _exact_keys(record, expected_keys, path="manifest.geometry")

    n_sites = _integer(record["n_sites"], path="geometry.n_sites")
    coordinates = _decode_coordinates(record["coordinates"], reader=reader)
    embedding_dimension = _optional_integer(
        record["embedding_dimension"],
        path="geometry.embedding_dimension",
    )
    edges = tuple(
        _decode_edge(item, index=index, reader=reader)
        for index, item in enumerate(_list(record["edges"], path="geometry.edges"))
    )
    boundary_sites = frozenset(
        _integer(item, path=f"geometry.boundary_sites[{index}]")
        for index, item in enumerate(
            _list(record["boundary_sites"], path="geometry.boundary_sites")
        )
    )
    boundary_components = tuple(
        _decode_boundary_component(item, index=index)
        for index, item in enumerate(
            _list(record["boundary_components"], path="geometry.boundary_components")
        )
    )
    site_types = _decode_site_types(record["site_types"])
    dimension_records = tuple(
        _decode_dimension_record(item, index=index)
        for index, item in enumerate(
            _list(record["dimension_records"], path="geometry.dimension_records")
        )
    )
    rooted_tree = _decode_rooted_tree(record["rooted_tree"])
    metadata = _decode_metadata_mapping(
        record["metadata"],
        path="geometry.metadata",
        reader=reader,
        depth=0,
    )
    faces = tuple(
        _decode_face(item, index=index, reader=reader)
        for index, item in enumerate(_list(record["faces"], path="geometry.faces"))
    )
    try:
        return Geometry(
            n_sites=n_sites,
            edges=edges,
            coordinates=coordinates,
            embedding_dimension=embedding_dimension,
            boundary_sites=boundary_sites,
            boundary_components=boundary_components,
            site_types=site_types,
            dimension_records=dimension_records,
            rooted_tree=rooted_tree,
            metadata=metadata,
            faces=faces,
        )
    except (TypeError, ValueError) as error:
        raise GeometrySerializationError(
            "geometry manifest contains inconsistent geometry data"
        ) from error


def _decode_coordinates(value: Any, *, reader: _ArchiveReader) -> np.ndarray | None:
    if value is None:
        return None
    coordinates = reader.array(value, path="geometry.coordinates")
    if coordinates.dtype != np.dtype(float):
        raise GeometrySerializationError("coordinate array must use float64 dtype")
    return coordinates


def _decode_edge(value: Any, *, index: int, reader: _ArchiveReader) -> GeometryEdge:
    path = f"geometry.edges[{index}]"
    record = _mapping(value, path=path)
    _exact_keys(
        record,
        {
            "source",
            "target",
            "edge_type",
            "boundary_crossing",
            "displacement",
            "metadata",
        },
        path=path,
    )
    displacement_value = record["displacement"]
    displacement = (
        None
        if displacement_value is None
        else tuple(
            _finite_float(item, path=f"{path}.displacement[{item_index}]")
            for item_index, item in enumerate(
                _list(displacement_value, path=f"{path}.displacement")
            )
        )
    )
    try:
        return GeometryEdge(
            source=_integer(record["source"], path=f"{path}.source"),
            target=_integer(record["target"], path=f"{path}.target"),
            edge_type=_optional_string(record["edge_type"], path=f"{path}.edge_type"),
            boundary_crossing=_boolean(
                record["boundary_crossing"],
                path=f"{path}.boundary_crossing",
            ),
            displacement=displacement,
            metadata=_decode_metadata_mapping(
                record["metadata"],
                path=f"{path}.metadata",
                reader=reader,
                depth=0,
            ),
        )
    except (TypeError, ValueError) as error:
        raise GeometrySerializationError(f"{path} is invalid") from error


def _decode_boundary_component(value: Any, *, index: int) -> GeometryBoundaryComponent:
    path = f"geometry.boundary_components[{index}]"
    record = _mapping(value, path=path)
    _exact_keys(record, {"kind", "component_index", "sites"}, path=path)
    sites = frozenset(
        _integer(item, path=f"{path}.sites[{site_index}]")
        for site_index, item in enumerate(_list(record["sites"], path=f"{path}.sites"))
    )
    try:
        return GeometryBoundaryComponent(
            _string(record["kind"], path=f"{path}.kind"),  # type: ignore[arg-type]
            _integer(record["component_index"], path=f"{path}.component_index"),
            sites,
        )
    except (TypeError, ValueError) as error:
        raise GeometrySerializationError(f"{path} is invalid") from error


def _decode_dimension_record(value: Any, *, index: int) -> GeometryDimension:
    path = f"geometry.dimension_records[{index}]"
    record = _mapping(value, path=path)
    _exact_keys(record, {"kind", "value", "scope", "method", "exact"}, path=path)
    try:
        return GeometryDimension(
            kind=_string(record["kind"], path=f"{path}.kind"),  # type: ignore[arg-type]
            value=_finite_float(record["value"], path=f"{path}.value"),
            scope=_string(record["scope"], path=f"{path}.scope"),  # type: ignore[arg-type]
            method=_string(record["method"], path=f"{path}.method"),
            exact=_boolean(record["exact"], path=f"{path}.exact"),
        )
    except (TypeError, ValueError) as error:
        raise GeometrySerializationError(f"{path} is invalid") from error


def _decode_rooted_tree(value: Any) -> RootedTreeStructure | None:
    if value is None:
        return None
    path = "geometry.rooted_tree"
    record = _mapping(value, path=path)
    _exact_keys(record, {"root_site", "parents"}, path=path)
    parents = tuple(
        _optional_integer(item, path=f"{path}.parents[{index}]")
        for index, item in enumerate(_list(record["parents"], path=f"{path}.parents"))
    )
    try:
        return RootedTreeStructure(
            _integer(record["root_site"], path=f"{path}.root_site"),
            parents,
        )
    except (TypeError, ValueError) as error:
        raise GeometrySerializationError(f"{path} is invalid") from error


def _decode_face(value: Any, *, index: int, reader: _ArchiveReader) -> GeometryFace:
    path = f"geometry.faces[{index}]"
    record = _mapping(value, path=path)
    _exact_keys(record, {"sites", "face_type", "metadata"}, path=path)
    sites = tuple(
        _integer(item, path=f"{path}.sites[{site_index}]")
        for site_index, item in enumerate(_list(record["sites"], path=f"{path}.sites"))
    )
    try:
        return GeometryFace(
            sites,
            face_type=_optional_string(record["face_type"], path=f"{path}.face_type"),
            metadata=_decode_metadata_mapping(
                record["metadata"],
                path=f"{path}.metadata",
                reader=reader,
                depth=0,
            ),
        )
    except (TypeError, ValueError) as error:
        raise GeometrySerializationError(f"{path} is invalid") from error


def _decode_site_types(value: Any) -> tuple[str | None, ...] | None:
    if value is None:
        return None
    return tuple(
        _optional_string(item, path=f"geometry.site_types[{index}]")
        for index, item in enumerate(_list(value, path="geometry.site_types"))
    )


def _decode_metadata_mapping(
    value: Any,
    *,
    path: str,
    reader: _ArchiveReader,
    depth: int,
) -> dict[str, Any]:
    result = _decode_metadata_value(value, path=path, reader=reader, depth=depth)
    if not isinstance(result, dict):
        raise GeometrySerializationError(f"{path} must encode a metadata mapping")
    return result


def _decode_metadata_value(
    value: Any,
    *,
    path: str,
    reader: _ArchiveReader,
    depth: int,
) -> Any:
    if depth > _MAX_METADATA_DEPTH:
        raise GeometrySerializationError(
            f"{path} exceeds the metadata nesting limit"
        )
    node = _mapping(value, path=path)
    node_type = _string(node.get("type"), path=f"{path}.type")
    if node_type == "none":
        _exact_keys(node, {"type"}, path=path)
        return None
    if node_type == "boolean":
        _exact_keys(node, {"type", "value"}, path=path)
        return _boolean(node["value"], path=f"{path}.value")
    if node_type == "string":
        _exact_keys(node, {"type", "value"}, path=path)
        return _string(node["value"], path=f"{path}.value")
    if node_type == "bytes":
        _exact_keys(node, {"type", "value"}, path=path)
        encoded = _string(node["value"], path=f"{path}.value")
        try:
            return base64.b64decode(encoded.encode("ascii"), validate=True)
        except (UnicodeEncodeError, ValueError) as error:
            raise GeometrySerializationError(f"{path} has invalid base64 bytes") from error
    if node_type == "integer":
        _exact_keys(node, {"type", "value"}, path=path)
        encoded = _string(node["value"], path=f"{path}.value")
        if _INTEGER_PATTERN.fullmatch(encoded) is None:
            raise GeometrySerializationError(f"{path} has an invalid integer")
        return int(encoded)
    if node_type == "real":
        _exact_keys(node, {"type", "value"}, path=path)
        return _finite_float(node["value"], path=f"{path}.value")
    if node_type == "complex":
        _exact_keys(node, {"type", "real", "imag"}, path=path)
        return complex(
            _finite_float(node["real"], path=f"{path}.real"),
            _finite_float(node["imag"], path=f"{path}.imag"),
        )
    if node_type in ("ndarray", "numpy_scalar"):
        _exact_keys(node, {"type", "array"}, path=path)
        values = reader.array(node["array"], path=f"{path}.array")
        if values.dtype.kind not in _SUPPORTED_ARRAY_KINDS:
            raise GeometrySerializationError(f"{path} has an unsupported array dtype")
        if values.dtype.kind in "fc" and not np.all(np.isfinite(values)):
            raise GeometrySerializationError(f"{path} has non-finite array values")
        if node_type == "numpy_scalar":
            if values.ndim != 0:
                raise GeometrySerializationError(f"{path} scalar array must be zero-dimensional")
            return values[()]
        return values
    if node_type == "tuple":
        _exact_keys(node, {"type", "items"}, path=path)
        return tuple(
            _decode_metadata_value(
                item,
                path=f"{path}[{index}]",
                reader=reader,
                depth=depth + 1,
            )
            for index, item in enumerate(_list(node["items"], path=f"{path}.items"))
        )
    if node_type == "mapping":
        _exact_keys(node, {"type", "items"}, path=path)
        result: dict[str, Any] = {}
        for index, item in enumerate(_list(node["items"], path=f"{path}.items")):
            pair = _list(item, path=f"{path}.items[{index}]")
            if len(pair) != 2:
                raise GeometrySerializationError(
                    f"{path}.items[{index}] must contain a key and value"
                )
            key = _string(pair[0], path=f"{path}.items[{index}][0]")
            if key in result:
                raise GeometrySerializationError(f"{path} has duplicate key {key!r}")
            result[key] = _decode_metadata_value(
                pair[1],
                path=f"{path}[{key!r}]",
                reader=reader,
                depth=depth + 1,
            )
        return result
    raise GeometrySerializationError(f"{path} has unknown metadata type {node_type!r}")


def _mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GeometrySerializationError(f"{path} must be a JSON object")
    if not all(isinstance(key, str) for key in value):
        raise GeometrySerializationError(f"{path} object keys must be strings")
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GeometrySerializationError(
                f"geometry manifest has duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise GeometrySerializationError(
        f"geometry manifest contains non-finite JSON constant {value!r}"
    )


def _list(value: Any, *, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise GeometrySerializationError(f"{path} must be a JSON array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, path: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise GeometrySerializationError(
            f"{path} has invalid fields; missing={missing!r}, extra={extra!r}"
        )


def _integer(value: Any, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GeometrySerializationError(f"{path} must be an integer")
    return value


def _optional_integer(value: Any, *, path: str) -> int | None:
    if value is None:
        return None
    return _integer(value, path=path)


def _boolean(value: Any, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise GeometrySerializationError(f"{path} must be a boolean")
    return value


def _string(value: Any, *, path: str) -> str:
    if not isinstance(value, str):
        raise GeometrySerializationError(f"{path} must be a string")
    return value


def _optional_string(value: Any, *, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path=path)


def _finite_float(value: Any, *, path: str) -> float:
    if not isinstance(value, str):
        raise GeometrySerializationError(f"{path} must be a hexadecimal float string")
    try:
        result = float.fromhex(value)
    except ValueError as error:
        raise GeometrySerializationError(f"{path} has an invalid float") from error
    if not np.isfinite(result):
        raise GeometrySerializationError(f"{path} must be finite")
    return result
