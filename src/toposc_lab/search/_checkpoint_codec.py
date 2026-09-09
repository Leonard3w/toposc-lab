"""Closed, data-only codec for the version-1 search checkpoint object graph.

Only the explicitly listed record and enum classes can be reconstructed.
Dataclass nodes use backward references to preserve shared search identities.
Geometry archives reuse Phase 6; arrays use bounded raw bytes, never pickle.
"""

from __future__ import annotations

import base64
import math
from collections.abc import Mapping
from dataclasses import fields
from enum import Enum
from typing import Any

import numpy as np

from toposc_lab import evaluation as ev
from toposc_lab.core import results as core
from toposc_lab.geometry import geometry_from_bytes, geometry_to_bytes
from toposc_lab.observables.localization import LocalizationProfile
from toposc_lab.observables.majorana import MajoranaDiagnostics
from toposc_lab.search import diversity_preservation as diversity
from toposc_lab.search import generation_loop as loop
from toposc_lab.search import generation_population as generation
from toposc_lab.search import initial_population as initial
from toposc_lab.search import lexicographic_fitness as lexicographic
from toposc_lab.search import mutation_validity as validity
from toposc_lab.search import novelty_score as novelty
from toposc_lab.search import population_elitism as elitism
from toposc_lab.search import population_fitness as fitness
from toposc_lab.search import population_selection as selection
from toposc_lab.search.checkpoint import (
    DEFAULT_CHECKPOINT_MAX_UNCOMPRESSED_BYTES,
    SearchCheckpoint,
    SearchCheckpointError,
    _json_bytes,
    _parse_json,
)
from toposc_lab.search.geometry_genome import GEOMETRY_GENOME_SCHEMA_VERSION, GeometryGenome
from toposc_lab.search.search_benchmark import BenchmarkArmResult, SearchBenchmarkTrial
from toposc_lab.topology.results import NumericalConfidence, TopologyMethod, TopologyResult

_RECORDS: tuple[type[Any], ...] = (
    lexicographic.DerivedEvaluationRun,
    lexicographic.LexicographicFitnessDefinition,
    lexicographic.LexicographicFitness,
    BenchmarkArmResult,
    SearchBenchmarkTrial,
    SearchCheckpoint,
    core.BasisLayout,
    core.SimulationResult,
    ev.GeometryEvaluationRun,
    ev.GeometryEvaluation,
    ev.ReproducibilityRecord,
    ev.CandidateValidityReport,
    ev.CandidateValidityIssue,
    ev.CandidateStageFailure,
    ev.BasicScalarScore,
    ev.MultiObjectiveEvaluation,
    ev.ObjectiveSpec,
    ev.ObjectiveValue,
    LocalizationProfile,
    MajoranaDiagnostics,
    TopologyResult,
    NumericalConfidence,
    loop.GenerationLoopResult,
    loop.GenerationLoopConfig,
    loop.GenerationTransition,
    loop.GenerationOffspringRecord,
    loop.OffspringProposal,
    generation.GenerationPopulation,
    generation.GenerationPopulationMember,
    initial.InitialPopulation,
    initial.InitialPopulationMember,
    validity.MutationValidityPolicy,
    validity.MutationValidityReport,
    validity.MutationValidityIssue,
    fitness.PopulationFitnessResult,
    fitness.PopulationFitnessMember,
    fitness.PopulationFitnessFailure,
    fitness.ScalarFitnessDefinition,
    fitness.MultiObjectiveFitnessDefinition,
    selection.PopulationSelectionResult,
    selection.TournamentSelectionConfig,
    selection.TournamentSelectionRecord,
    elitism.PopulationElitismResult,
    elitism.ElitismConfig,
    elitism.EliteTier,
    diversity.DiversityPreservationPolicy,
    diversity.GeometrySearchFamily,
    diversity.PopulationFamilyMember,
    diversity.PopulationDiversityReport,
    diversity.DiversityPolicyViolation,
    novelty.NoveltyScoreConfig,
    novelty.NoveltyReferenceCohort,
    novelty.NoveltyComparison,
    novelty.PopulationNoveltyMember,
    novelty.PopulationNoveltyReport,
)
_ENUMS: tuple[type[Enum], ...] = (
    ev.CandidateIssueCategory,
    ev.CandidateIssueSeverity,
    ev.CandidateFailureStage,
    ev.BasicScoreComponent,
    ev.ObjectiveDirection,
    ev.ObjectiveQuantity,
    TopologyMethod,
    fitness.PopulationFitnessStatus,
    fitness.PopulationFitnessFailureStage,
    novelty.NoveltyComparisonSource,
)
_RECORD_BY_NAME = {cls.__name__: cls for cls in _RECORDS}
_ENUM_BY_NAME = {cls.__name__: cls for cls in _ENUMS}
_MAX_DEPTH = 100
_MAX_NODES = 1_000_000


class _Encoder:
    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = []
        self.memo: dict[int, int] = {}
        self.active: set[int] = set()

    def value(self, value: Any, depth: int = 0) -> Any:
        if depth > _MAX_DEPTH:
            raise SearchCheckpointError("checkpoint nesting exceeds depth limit")
        if isinstance(value, Enum):
            if type(value) not in _ENUMS:
                raise SearchCheckpointError("unsupported checkpoint enum")
            return ["enum", type(value).__name__, value.value]
        if value is None or type(value) in (bool, int, str):
            return value
        if type(value) is float:
            return ["float", value.hex()]
        if type(value) is complex:
            return ["complex", value.real.hex(), value.imag.hex()]
        if isinstance(value, (np.ndarray, np.generic)):
            array = np.asarray(value)
            if array.dtype.kind not in "biufcSU" or array.dtype.hasobject:
                raise SearchCheckpointError("unsupported checkpoint array dtype")
            return [
                "numpy_scalar" if isinstance(value, np.generic) else "array",
                array.dtype.str,
                list(array.shape),
                _b64(array.tobytes(order="C")),
            ]
        if isinstance(value, bytes):
            return ["bytes", _b64(value)]
        if isinstance(value, Mapping):
            return [
                "map",
                [
                    [self.value(key, depth + 1), self.value(item, depth + 1)]
                    for key, item in value.items()
                ],
            ]
        if isinstance(value, (tuple, list, frozenset, set)):
            tag = type(value).__name__
            prepared = [self.value(item, depth + 1) for item in value]
            if isinstance(value, (set, frozenset)):
                prepared.sort(key=_json_bytes)
            return [tag, prepared]
        if type(value) not in _RECORDS and type(value) is not GeometryGenome:
            raise SearchCheckpointError(f"unsupported checkpoint value: {type(value).__name__}")
        identity = id(value)
        if identity in self.active:
            raise SearchCheckpointError("cyclic checkpoint records are unsupported")
        if identity in self.memo:
            return ["ref", self.memo[identity]]
        self.active.add(identity)
        if isinstance(value, GeometryGenome):
            node = {
                "type": "genome",
                "schema_version": value.schema_version,
                "archive": _b64(geometry_to_bytes(value.to_geometry())),
            }
        else:
            node = {
                "type": type(value).__name__,
                "fields": {
                    item.name: self.value(getattr(value, item.name), depth + 1)
                    for item in fields(value)
                },
            }
        if len(self.nodes) >= _MAX_NODES:
            raise SearchCheckpointError("too many checkpoint records")
        index = len(self.nodes)
        self.nodes.append(node)
        self.memo[identity] = index
        self.active.remove(identity)
        return ["ref", index]


class _Decoder:
    def __init__(self, max_geometry_bytes: int) -> None:
        self.nodes: list[Any] = []
        self.used: set[int] = set()
        self.remaining_geometry_bytes = max_geometry_bytes

    def value(self, value: Any, depth: int = 0) -> Any:
        if depth > _MAX_DEPTH:
            raise SearchCheckpointError("checkpoint nesting exceeds depth limit")
        if value is None or type(value) in (bool, int, str):
            return value
        if not isinstance(value, list) or not value or not isinstance(value[0], str):
            raise SearchCheckpointError("invalid typed checkpoint value")
        tag = value[0]
        if tag == "ref" and len(value) == 2:
            index = value[1]
            if type(index) is not int or not 0 <= index < len(self.nodes):
                raise SearchCheckpointError("invalid or forward checkpoint reference")
            self.used.add(index)
            return self.nodes[index]
        if tag == "enum" and len(value) == 3:
            cls = _ENUM_BY_NAME.get(value[1])
            if cls is None:
                raise SearchCheckpointError("unsupported checkpoint enum")
            return cls(value[2])
        if tag == "float" and len(value) == 2 and isinstance(value[1], str):
            return float.fromhex(value[1])
        if tag == "complex" and len(value) == 3:
            return complex(float.fromhex(value[1]), float.fromhex(value[2]))
        if tag == "bytes" and len(value) == 2:
            return _unb64(value[1])
        if tag in ("array", "numpy_scalar") and len(value) == 4:
            return self.array(value)
        if tag in ("tuple", "list", "set", "frozenset", "map") and len(value) == 2:
            if not isinstance(value[1], list):
                raise SearchCheckpointError("invalid checkpoint collection")
            if tag == "map":
                result: dict[Any, Any] = {}
                for pair in value[1]:
                    if not isinstance(pair, list) or len(pair) != 2:
                        raise SearchCheckpointError("invalid checkpoint mapping entry")
                    key, item = (self.value(part, depth + 1) for part in pair)
                    if key in result:
                        raise SearchCheckpointError("duplicate checkpoint mapping key")
                    result[key] = item
                return result
            items = [self.value(item, depth + 1) for item in value[1]]
            if tag == "tuple":
                return tuple(items)
            if tag in ("set", "frozenset"):
                prepared = frozenset(items)
                if len(prepared) != len(items):
                    raise SearchCheckpointError("duplicate checkpoint set entry")
                return set(prepared) if tag == "set" else prepared
            return items
        raise SearchCheckpointError(f"unsupported checkpoint value tag: {tag}")

    def array(self, value: list[Any]) -> Any:
        dtype = np.dtype(value[1])
        shape = value[2]
        if dtype.kind not in "biufcSU" or dtype.hasobject or dtype.itemsize < 1:
            raise SearchCheckpointError("unsupported checkpoint array dtype")
        if (
            not isinstance(shape, list)
            or len(shape) > 32
            or any(type(size) is not int or size < 0 for size in shape)
        ):
            raise SearchCheckpointError("invalid checkpoint array shape")
        raw = _unb64(value[3])
        if math.prod(shape) * dtype.itemsize != len(raw):
            raise SearchCheckpointError("checkpoint array size disagrees with shape")
        array = np.frombuffer(raw, dtype=dtype).reshape(tuple(shape))
        if value[0] == "numpy_scalar":
            if shape:
                raise SearchCheckpointError("a numpy scalar requires empty shape")
            return array[()]
        return array

    def node(self, node: Any) -> Any:
        if not isinstance(node, dict) or not isinstance(node.get("type"), str):
            raise SearchCheckpointError("invalid checkpoint record")
        if node["type"] == "genome":
            if (
                set(node) != {"type", "schema_version", "archive"}
                or type(node["schema_version"]) is not int
                or node["schema_version"] != GEOMETRY_GENOME_SCHEMA_VERSION
            ):
                raise SearchCheckpointError("unsupported genome checkpoint schema")
            raw = _unb64(node["archive"])
            # Bound the aggregate expansion of all nested Phase-6 archives.
            from io import BytesIO
            from zipfile import ZipFile

            with ZipFile(BytesIO(raw)) as archive:
                expanded = sum(item.file_size for item in archive.infolist())
            if expanded > self.remaining_geometry_bytes:
                raise SearchCheckpointError("nested geometry archives exceed expansion limit")
            geometry = geometry_from_bytes(
                raw,
                max_archive_bytes=max(1, len(raw)),
                max_uncompressed_bytes=max(1, self.remaining_geometry_bytes),
            )
            self.remaining_geometry_bytes -= expanded
            return GeometryGenome.from_geometry(geometry)
        cls = _RECORD_BY_NAME.get(node["type"])
        if cls is None or set(node) != {"type", "fields"}:
            raise SearchCheckpointError("unsupported checkpoint record type or fields")
        schema = fields(cls)
        values = node["fields"]
        if not isinstance(values, dict) or set(values) != {item.name for item in schema}:
            raise SearchCheckpointError(f"unexpected fields for {cls.__name__}")
        decoded = {key: self.value(value) for key, value in values.items()}
        result = cls(**{item.name: decoded[item.name] for item in schema if item.init})
        for item in schema:
            if not item.init and (
                type(getattr(result, item.name)) is not type(decoded[item.name])
                or getattr(result, item.name) != decoded[item.name]
            ):
                raise SearchCheckpointError(f"incompatible or altered {cls.__name__}.{item.name}")
        return result


def encode_checkpoint(checkpoint: SearchCheckpoint) -> bytes:
    """Encode the closed record graph in dependency order."""
    try:
        encoder = _Encoder()
        root = encoder.value(checkpoint)
        return _json_bytes({"nodes": encoder.nodes, "root": root})
    except SearchCheckpointError:
        raise
    except (ValueError, TypeError, OverflowError, RecursionError) as error:
        raise SearchCheckpointError(f"cannot encode checkpoint: {error}") from error


def decode_checkpoint(
    payload: bytes,
    *,
    max_geometry_bytes: int = DEFAULT_CHECKPOINT_MAX_UNCOMPRESSED_BYTES,
) -> SearchCheckpoint:
    """Reconstruct through whitelisted constructors without importing file-specified code."""
    try:
        document = _parse_json(payload)
        if not isinstance(document, dict) or set(document) != {"nodes", "root"}:
            raise SearchCheckpointError("invalid checkpoint object table")
        nodes = document["nodes"]
        if not isinstance(nodes, list) or not 1 <= len(nodes) <= _MAX_NODES:
            raise SearchCheckpointError("invalid checkpoint record count")
        decoder = _Decoder(max_geometry_bytes)
        for node in nodes:
            decoder.nodes.append(decoder.node(node))
        if document["root"] != ["ref", len(nodes) - 1]:
            raise SearchCheckpointError("checkpoint root must be the final record")
        result = decoder.value(document["root"])
        if not isinstance(result, SearchCheckpoint) or len(decoder.used) != len(nodes):
            raise SearchCheckpointError("checkpoint contains wrong root or unused records")
        return result
    except SearchCheckpointError:
        raise
    except (
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
        RecursionError,
        AttributeError,
        IndexError,
    ) as error:
        raise SearchCheckpointError(f"invalid checkpoint payload: {error}") from error


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(value: Any) -> bytes:
    if not isinstance(value, str):
        raise SearchCheckpointError("checkpoint byte payload must be base64 text")
    return base64.b64decode(value, validate=True)
