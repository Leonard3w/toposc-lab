"""Basis-independent graph tensors for minimal message-passing surrogates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from toposc_lab.data import DatasetRecord
from toposc_lab.ml.features import HandcraftedFeatureSchema

GRAPH_INPUT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class GraphInputSchema:
    """Training-fitted categorical/dimensional contract for graph inputs."""

    model_name: str
    embedding_dimension: int
    site_types: tuple[str, ...]
    edge_types: tuple[str, ...]
    parameter_schema: HandcraftedFeatureSchema
    version: int = GRAPH_INPUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.version != GRAPH_INPUT_SCHEMA_VERSION:
            raise ValueError("unsupported graph input schema version")
        if self.embedding_dimension < 0:
            raise ValueError("embedding_dimension must be nonnegative")
        if self.parameter_schema.model_name != self.model_name:
            raise ValueError("parameter schema model identity does not match")
        object.__setattr__(self, "site_types", tuple(sorted(set(self.site_types))))
        object.__setattr__(self, "edge_types", tuple(sorted(set(self.edge_types))))

    @classmethod
    def fit(cls, records: Sequence[DatasetRecord]) -> GraphInputSchema:
        selected = tuple(records)
        if not selected:
            raise ValueError("at least one training record is required")
        parameter_schema = HandcraftedFeatureSchema.fit(selected)
        geometries = tuple(record.geometry.to_geometry() for record in selected)
        dimensions = tuple(geometry.embedding_dimension or 0 for geometry in geometries)
        site_types = {
            site_type
            for geometry in geometries
            for site_type in (geometry.site_types or ())
            if site_type is not None
        }
        edge_types = {
            edge.edge_type
            for geometry in geometries
            for edge in geometry.edges
            if edge.edge_type is not None
        }
        return cls(
            model_name=parameter_schema.model_name,
            embedding_dimension=max(dimensions, default=0),
            site_types=tuple(site_types),
            edge_types=tuple(edge_types),
            parameter_schema=parameter_schema,
        )


@dataclass(frozen=True, slots=True)
class GraphInput:
    """One graph with directed message edges and exact-record provenance."""

    node_features: NDArray[np.float64]
    edge_index: NDArray[np.int64]
    edge_features: NDArray[np.float64]
    global_features: NDArray[np.float64]
    record_id: str
    geometry_exact_id: str
    family_fingerprint: str
    dataset_schema_version: int

    def __post_init__(self) -> None:
        nodes = _immutable_array(self.node_features, float, ndim=2, name="node_features")
        indices = _immutable_array(self.edge_index, np.int64, ndim=2, name="edge_index")
        edges = _immutable_array(self.edge_features, float, ndim=2, name="edge_features")
        global_features = _immutable_array(
            self.global_features, float, ndim=1, name="global_features"
        )
        if indices.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, directed_edge_count)")
        if indices.shape[1] != edges.shape[0]:
            raise ValueError("edge_index and edge_features must align")
        if indices.size and (np.min(indices) < 0 or np.max(indices) >= nodes.shape[0]):
            raise ValueError("edge_index references an unknown node")
        object.__setattr__(self, "node_features", nodes)
        object.__setattr__(self, "edge_index", indices)
        object.__setattr__(self, "edge_features", edges)
        object.__setattr__(self, "global_features", global_features)


def graph_input_from_record(record: DatasetRecord, *, schema: GraphInputSchema) -> GraphInput:
    """Represent geometry/model inputs without Hamiltonian basis assumptions or labels."""
    if not isinstance(record, DatasetRecord):
        raise TypeError("record must be DatasetRecord")
    if record.model.model_name != schema.model_name:
        raise ValueError("record model identity does not match graph schema")
    geometry = record.geometry.to_geometry()
    dimension = schema.embedding_dimension
    coordinates = geometry.coordinates
    if coordinates is None:
        normalized_coordinates = np.zeros((geometry.n_sites, dimension), dtype=float)
        coordinates_missing = np.ones(geometry.n_sites, dtype=float)
    else:
        if coordinates.shape[1] > dimension:
            raise ValueError("record embedding exceeds training graph schema")
        centered = coordinates - np.mean(coordinates, axis=0)
        scale = float(np.sqrt(np.mean(np.square(centered))))
        if scale > 1e-12:
            centered = centered / scale
        normalized_coordinates = np.zeros((geometry.n_sites, dimension), dtype=float)
        normalized_coordinates[:, : coordinates.shape[1]] = centered
        coordinates_missing = np.zeros(geometry.n_sites, dtype=float)
    node_rows: list[list[float]] = []
    for site in geometry.site_indices:
        site_type = None if geometry.site_types is None else geometry.site_types[site]
        known_types = [float(site_type == name) for name in schema.site_types]
        unknown_type = float(site_type is not None and site_type not in schema.site_types)
        node_rows.append(
            [
                geometry.degree(site) / max(1, geometry.n_sites - 1),
                float(site in geometry.boundary_sites),
                *normalized_coordinates[site].tolist(),
                coordinates_missing[site],
                *known_types,
                unknown_type,
            ]
        )

    directed_indices: list[tuple[int, int]] = []
    edge_rows: list[list[float]] = []
    for edge in geometry.edges:
        for source, target in (
            (edge.source, edge.target),
            (edge.target, edge.source),
        ):
            displacement, missing = _edge_displacement(geometry, source, target, dimension)
            known_types = [float(edge.edge_type == name) for name in schema.edge_types]
            unknown_type = float(edge.edge_type is not None and edge.edge_type not in schema.edge_types)
            directed_indices.append((source, target))
            edge_rows.append(
                [
                    float(edge.boundary_crossing),
                    float(np.linalg.norm(displacement)) if not missing else 0.0,
                    float(missing),
                    *displacement.tolist(),
                    *known_types,
                    unknown_type,
                ]
            )
    edge_width = 3 + dimension + len(schema.edge_types) + 1
    edge_index = (
        np.asarray(directed_indices, dtype=np.int64).T
        if directed_indices
        else np.empty((2, 0), dtype=np.int64)
    )
    edge_features = (
        np.asarray(edge_rows, dtype=float)
        if edge_rows
        else np.empty((0, edge_width), dtype=float)
    )
    global_features = _parameter_features(record.model.parameters, schema.parameter_schema)
    return GraphInput(
        node_features=np.asarray(node_rows, dtype=float),
        edge_index=edge_index,
        edge_features=edge_features,
        global_features=global_features,
        record_id=record.record_id,
        geometry_exact_id=record.geometry.exact_id,
        family_fingerprint=record.geometry.family_fingerprint,
        dataset_schema_version=record.schema_version,
    )


def graph_inputs_from_records(
    records: Sequence[DatasetRecord], *, schema: GraphInputSchema
) -> tuple[GraphInput, ...]:
    return tuple(graph_input_from_record(record, schema=schema) for record in records)


def _parameter_features(
    parameters: Mapping[str, object], schema: HandcraftedFeatureSchema
) -> NDArray[np.float64]:
    values: list[float] = []
    for name, components in schema.parameter_components.items():
        raw = parameters.get(name)
        numeric = _finite_numeric(raw)
        missing = numeric is None
        value = 0.0j if missing else complex(numeric)
        if components == ("value",):
            if not missing and value.imag != 0.0:
                raise ValueError(f"parameter {name!r} became complex after schema fitting")
            values.extend((float(value.real), float(missing)))
        else:
            values.extend((float(value.real), float(value.imag), float(missing)))
    return np.asarray(values, dtype=float)


def _edge_displacement(geometry: object, source: int, target: int, dimension: int) -> tuple[np.ndarray, bool]:
    from toposc_lab.geometry import Geometry

    if not isinstance(geometry, Geometry):
        raise TypeError("geometry must be Geometry")
    try:
        raw = geometry.displacement_between(source, target)
    except ValueError:
        return np.zeros(dimension, dtype=float), True
    if len(raw) > dimension:
        raise ValueError("edge displacement exceeds training graph schema")
    padded = np.zeros(dimension, dtype=float)
    padded[: len(raw)] = raw
    return padded, False


def _finite_numeric(value: object) -> complex | float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, complex):
        return value if np.isfinite(value.real) and np.isfinite(value.imag) else None
    if isinstance(value, Real):
        return float(value) if np.isfinite(float(value)) else None
    return None


def _immutable_array(
    values: object, dtype: object, *, ndim: int, name: str
) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    if result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    result.setflags(write=False)
    return result
