"""Leakage-safe handcrafted features for exact-physics dataset records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from toposc_lab.data import DatasetRecord
from toposc_lab.geometry import extract_geometry_descriptors

FEATURE_SCHEMA_VERSION = 1

_GRAPH_FEATURES = (
    "site_count",
    "edge_count",
    "mean_degree",
    "degree_variance",
    "component_count",
    "largest_component_site_count",
    "is_connected",
    "cycle_rank",
    "has_cycles",
    "triangle_count",
    "mean_local_clustering",
    "connected_pair_count",
    "reachable_pair_fraction",
    "mean_finite_shortest_path_length",
    "maximum_finite_shortest_path_length",
)

_GEOMETRY_FEATURES = (
    *tuple(f"graph::{name}" for name in _GRAPH_FEATURES),
    "geometry::path_length_missing",
    "geometry::boundary_fraction",
    "geometry::face_count",
    "geometry::embedding_dimension",
    "geometry::embedding_missing",
    "geometry::coordinate_extent_mean",
    "geometry::coordinate_extent_max",
    "geometry::coordinates_missing",
    "geometry::edge_length_mean",
    "geometry::edge_length_std",
    "geometry::edge_lengths_missing",
    "geometry::site_type_count",
)


@dataclass(frozen=True, slots=True)
class HandcraftedFeatureSchema:
    """A fixed feature contract fitted without consulting scientific labels."""

    model_name: str
    parameter_components: Mapping[str, tuple[str, ...]]
    feature_names: tuple[str, ...]
    version: int = FEATURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name:
            raise ValueError("model_name must be a non-empty string")
        components = {
            str(name): tuple(str(component) for component in values)
            for name, values in self.parameter_components.items()
        }
        if any(not name for name in components):
            raise ValueError("parameter names must be non-empty")
        if any(value not in {("value",), ("real", "imag")} for value in components.values()):
            raise ValueError("unsupported parameter component declaration")
        if self.version != FEATURE_SCHEMA_VERSION:
            raise ValueError("unsupported feature schema version")
        expected = _feature_names(components)
        if tuple(self.feature_names) != expected:
            raise ValueError("feature_names do not match parameter_components")
        object.__setattr__(
            self,
            "parameter_components",
            MappingProxyType(dict(sorted(components.items()))),
        )
        object.__setattr__(self, "feature_names", expected)

    @classmethod
    def fit(cls, records: Sequence[DatasetRecord]) -> HandcraftedFeatureSchema:
        """Infer only the numeric parameter names/components from training records."""
        selected = tuple(records)
        if not selected:
            raise ValueError("at least one training record is required")
        if any(not isinstance(record, DatasetRecord) for record in selected):
            raise TypeError("records must contain only DatasetRecord values")
        model_names = {record.model.model_name for record in selected}
        if len(model_names) != 1:
            raise ValueError("one feature schema cannot mix model identities")

        parameter_kinds: dict[str, set[str]] = {}
        for record in selected:
            for name, value in record.model.parameters.items():
                kind = _numeric_parameter_kind(value)
                if kind is not None:
                    parameter_kinds.setdefault(name, set()).add(kind)
        components = {
            name: ("real", "imag") if "complex" in kinds else ("value",)
            for name, kinds in sorted(parameter_kinds.items())
        }
        return cls(
            model_name=next(iter(model_names)),
            parameter_components=components,
            feature_names=_feature_names(components),
        )


@dataclass(frozen=True, slots=True)
class FeatureMatrix:
    """Immutable numeric matrix with row identities and an explicit schema."""

    values: NDArray[np.float64]
    record_ids: tuple[str, ...]
    schema: HandcraftedFeatureSchema

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=float, copy=True)
        if values.ndim != 2:
            raise ValueError("feature values must be a two-dimensional matrix")
        if values.shape != (len(self.record_ids), len(self.schema.feature_names)):
            raise ValueError("feature matrix shape does not match identities/schema")
        if not np.all(np.isfinite(values)):
            raise ValueError("feature values must be finite")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "record_ids", tuple(self.record_ids))


def extract_handcrafted_features(
    records: Sequence[DatasetRecord],
    *,
    schema: HandcraftedFeatureSchema,
) -> FeatureMatrix:
    """Transform records using geometry/model inputs only.

    The implementation deliberately never reads spectra, observables, topology,
    robustness, or provenance. Missing numeric parameters get a value of zero
    plus an explicit missingness indicator.
    """
    selected = tuple(records)
    if any(not isinstance(record, DatasetRecord) for record in selected):
        raise TypeError("records must contain only DatasetRecord values")
    rows = tuple(_record_features(record, schema) for record in selected)
    values = np.asarray(rows, dtype=float)
    if not rows:
        values = np.empty((0, len(schema.feature_names)), dtype=float)
    return FeatureMatrix(
        values=values,
        record_ids=tuple(record.record_id for record in selected),
        schema=schema,
    )


def records_for_ids(
    records: Iterable[DatasetRecord], record_ids: Sequence[str]
) -> tuple[DatasetRecord, ...]:
    """Return records in requested ID order and fail on missing/duplicate IDs."""
    selected = tuple(records)
    lookup = {record.record_id: record for record in selected}
    if len(lookup) != len(selected):
        raise ValueError("records contain duplicate identifiers")
    missing = [record_id for record_id in record_ids if record_id not in lookup]
    if missing:
        raise KeyError(f"unknown dataset record identifiers: {missing}")
    return tuple(lookup[record_id] for record_id in record_ids)


def _record_features(
    record: DatasetRecord, schema: HandcraftedFeatureSchema
) -> tuple[float, ...]:
    if record.model.model_name != schema.model_name:
        raise ValueError(
            f"record model {record.model.model_name!r} does not match schema model "
            f"{schema.model_name!r}"
        )
    geometry = record.geometry.to_geometry()
    descriptors = extract_geometry_descriptors(geometry)
    graph_values = [
        0.0 if descriptors[name] is None else float(descriptors[name])
        for name in _GRAPH_FEATURES
    ]
    path_missing = float(
        descriptors["mean_finite_shortest_path_length"] is None
        or descriptors["maximum_finite_shortest_path_length"] is None
    )
    coordinates = geometry.coordinates
    if coordinates is None:
        embedding_dimension = float(geometry.embedding_dimension or 0)
        embedding_missing = float(geometry.embedding_dimension is None)
        coordinate_extent_mean = 0.0
        coordinate_extent_max = 0.0
        coordinates_missing = 1.0
    else:
        extents = np.ptp(coordinates, axis=0)
        embedding_dimension = float(coordinates.shape[1])
        embedding_missing = 0.0
        coordinate_extent_mean = float(np.mean(extents))
        coordinate_extent_max = float(np.max(extents))
        coordinates_missing = 0.0
    lengths: list[float] = []
    for edge in geometry.edges:
        try:
            lengths.append(geometry.distance(edge.source, edge.target))
        except ValueError:
            continue
    geometry_values = [
        path_missing,
        len(geometry.boundary_sites) / geometry.n_sites,
        float(geometry.n_faces),
        embedding_dimension,
        embedding_missing,
        coordinate_extent_mean,
        coordinate_extent_max,
        coordinates_missing,
        float(np.mean(lengths)) if lengths else 0.0,
        float(np.std(lengths)) if lengths else 0.0,
        float(not lengths),
        float(len(set(geometry.site_types or ())) if geometry.site_types is not None else 0),
    ]
    parameter_values: list[float] = []
    for name, components in schema.parameter_components.items():
        raw = record.model.parameters.get(name)
        kind = _numeric_parameter_kind(raw)
        missing = kind is None
        if components == ("value",):
            if kind == "complex":
                raise ValueError(f"parameter {name!r} became complex after schema fitting")
            parameter_values.extend((0.0 if missing else float(raw), float(missing)))
        else:
            value = 0.0j if missing else complex(raw)
            parameter_values.extend((float(value.real), float(value.imag), float(missing)))
    return (*graph_values, *geometry_values, *parameter_values)


def _feature_names(components: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
    names = list(_GEOMETRY_FEATURES)
    for parameter, parameter_components in sorted(components.items()):
        names.extend(f"parameter::{parameter}::{name}" for name in parameter_components)
        names.append(f"parameter::{parameter}::missing")
    return tuple(names)


def _numeric_parameter_kind(value: object) -> str | None:
    if isinstance(value, bool):
        return "real"
    if isinstance(value, complex):
        return "complex" if np.isfinite(value.real) and np.isfinite(value.imag) else None
    if isinstance(value, Real):
        return "real" if np.isfinite(float(value)) else None
    return None
