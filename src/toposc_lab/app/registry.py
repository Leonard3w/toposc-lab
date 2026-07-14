"""Erweiterbares Verzeichnis der in der App verfuegbaren Modelle.

Ein neues Modell benoetigt fuer die Oberflaeche nur einen ``ModelSpec``.
Physik, Parameter-Validierung und Darstellung bleiben damit vom
Streamlit-Code getrennt.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, get_args

from pydantic import BaseModel as PydanticBaseModel

from toposc_lab.core.model import BaseModel
from toposc_lab.models.bhz_model import BHZModel, BHZModelParameters
from toposc_lab.models.graphene_model import GrapheneModel, GrapheneParameters
from toposc_lab.models.haldane_model import HaldaneModel, HaldaneModelParameters
from toposc_lab.models.kitaev_chain import KitaevChain, KitaevChainParameters
from toposc_lab.models.kitaev_ladder import KitaevLadder, KitaevLadderParameters
from toposc_lab.models.qwz_model import QWZModel, QWZModelParameters
from toposc_lab.models.ssh_chain import SSHChain, SSHChainParameters

ModelFactory = Callable[[Any], BaseModel]


@dataclass(frozen=True)
class ModelSpec:
    """Alles, was die App ueber ein wissenschaftliches Modell wissen muss."""

    key: str
    display_name: str
    category: str
    description: str
    parameter_model: type[PydanticBaseModel]
    factory: ModelFactory
    default_parameters: Mapping[str, Any]
    scan_defaults: Mapping[str, tuple[float, float, int]] = field(
        default_factory=dict
    )

    def validated_default_parameters(self) -> dict[str, Any]:
        """Liefere gueltige Defaultwerte in der Form der Pydantic-Parameter."""
        return self.parameter_model.model_validate(
            self.default_parameters
        ).model_dump()

    def build(self, parameter_values: Mapping[str, Any]) -> BaseModel:
        """Validiere UI-Werte und erstelle das physikalische Modell."""
        parameters = self.parameter_model.model_validate(parameter_values)
        return self.factory(parameters)

    def scannable_parameter_names(self) -> tuple[str, ...]:
        """Liefere alle kontinuierlichen Modellparameter fuer 1D-Scans."""
        names: list[str] = []

        for name, parameter in self.parameter_model.model_fields.items():
            annotation = parameter.annotation
            possible_types = (annotation, *get_args(annotation))

            if float in possible_types:
                names.append(name)

        return tuple(names)

    def scan_default(self, parameter_name: str) -> tuple[float, float, int]:
        """Liefere eine sinnvolle Startkonfiguration fuer einen UI-Scan."""
        if parameter_name not in self.scannable_parameter_names():
            raise ValueError(f"Parameter {parameter_name!r} is not scannable")

        if parameter_name in self.scan_defaults:
            return self.scan_defaults[parameter_name]

        center = float(self.validated_default_parameters()[parameter_name])
        span = max(abs(center), 1.0)
        return center - span, center + span, 61


class ModelRegistry:
    """Registriert Modelle eindeutig und macht sie fuer die App auffindbar."""

    def __init__(self, specs: Iterable[ModelSpec] = ()) -> None:
        self._specifications: dict[str, ModelSpec] = {}

        for specification in specs:
            self.register(specification)

    def register(self, specification: ModelSpec) -> None:
        """Fuege ein Modell hinzu; doppelte Schluessel werden verhindert."""
        if not specification.key:
            raise ValueError("Model specification key must not be empty")

        if specification.key in self._specifications:
            raise ValueError(
                f"A model is already registered as {specification.key!r}"
            )

        self._specifications[specification.key] = specification

    def get(self, key: str) -> ModelSpec:
        """Liefere ein Modell ueber seinen stabilen technischen Schluessel."""
        try:
            return self._specifications[key]
        except KeyError as error:
            raise ValueError(f"Unknown model key: {key!r}") from error

    def specifications(self) -> tuple[ModelSpec, ...]:
        """Liefere alle Modelle in der Reihenfolge ihrer Registrierung."""
        return tuple(self._specifications.values())


MODEL_REGISTRY = ModelRegistry(
    (
        ModelSpec(
            key="kitaev-chain",
            display_name="Kitaev chain",
            category="Superconducting",
            description="1D p-wave superconductor with optional onsite disorder.",
            parameter_model=KitaevChainParameters,
            factory=KitaevChain,
            default_parameters={
                "n_sites": 60,
                "hopping": 1.0,
                "chemical_potential": 0.0,
                "pairing": 0.7,
                "boundary": "open",
                "disorder_strength": 0.0,
                "disorder_seed": None,
            },
            scan_defaults={
                "hopping": (0.1, 2.0, 61),
                "chemical_potential": (-4.0, 4.0, 81),
                "pairing": (0.0, 2.0, 61),
                "disorder_strength": (0.0, 2.0, 41),
            },
        ),
        ModelSpec(
            key="ssh-chain",
            display_name="SSH chain",
            category="One-dimensional insulators",
            description="Dimerized chain with a chiral topological phase.",
            parameter_model=SSHChainParameters,
            factory=SSHChain,
            default_parameters={
                "n_cells": 30,
                "v": 0.5,
                "w": 1.0,
                "boundary": "open",
            },
            scan_defaults={"v": (0.0, 2.0, 81), "w": (0.0, 2.0, 81)},
        ),
        ModelSpec(
            key="kitaev-ladder",
            display_name="Kitaev ladder",
            category="Superconducting",
            description="Coupled p-wave Kitaev chains on a quasi-1D ladder.",
            parameter_model=KitaevLadderParameters,
            factory=KitaevLadder,
            default_parameters={
                "n_legs": 2,
                "length": 40,
                "hopping": 1.0,
                "chemical_potential": 0.0,
                "pairing": 0.7,
                "rung_hopping": 0.25,
                "rung_pairing": 0.0,
                "boundary_length": "open",
                "boundary_legs": "open",
            },
            scan_defaults={
                "hopping": (0.1, 2.0, 61),
                "chemical_potential": (-4.0, 4.0, 81),
                "pairing": (0.0, 2.0, 61),
                "rung_hopping": (0.0, 2.0, 61),
                "rung_pairing": (0.0, 2.0, 61),
            },
        ),
        ModelSpec(
            key="qwz-model",
            display_name="Qi-Wu-Zhang model",
            category="Two-dimensional Chern insulators",
            description="Two-band lattice Chern insulator on a square lattice.",
            parameter_model=QWZModelParameters,
            factory=QWZModel,
            default_parameters={
                "n_x": 12,
                "n_y": 12,
                "mass": -1.0,
                "boundary_x": "open",
                "boundary_y": "open",
            },
            scan_defaults={"mass": (-3.0, 3.0, 81)},
        ),
        ModelSpec(
            key="bhz-model",
            display_name="BHZ model",
            category="Two-dimensional topological insulators",
            description="Four-band lattice model of a quantum spin Hall insulator.",
            parameter_model=BHZModelParameters,
            factory=BHZModel,
            default_parameters={
                "n_x": 10,
                "n_y": 10,
                "mass": -1.0,
                "boundary_x": "open",
                "boundary_y": "open",
            },
            scan_defaults={"mass": (-3.0, 3.0, 81)},
        ),
        ModelSpec(
            key="graphene",
            display_name="Graphene",
            category="Honeycomb materials",
            description="Spinless nearest-neighbor tight-binding model on honeycomb.",
            parameter_model=GrapheneParameters,
            factory=GrapheneModel,
            default_parameters={
                "n_x": 8,
                "n_y": 8,
                "hopping": 1.0,
                "boundary_x": "open",
                "boundary_y": "open",
            },
            scan_defaults={"hopping": (0.1, 2.0, 61)},
        ),
        ModelSpec(
            key="haldane-model",
            display_name="Haldane model",
            category="Two-dimensional Chern insulators",
            description="Honeycomb Chern insulator with complex next-nearest hopping.",
            parameter_model=HaldaneModelParameters,
            factory=HaldaneModel,
            default_parameters={
                "n_x": 8,
                "n_y": 8,
                "nearest_neighbor_hopping": 1.0,
                "next_nearest_neighbor_hopping": 0.2,
                "phase": 1.5707963267948966,
                "sublattice_mass": 0.0,
                "boundary_x": "open",
                "boundary_y": "open",
            },
            scan_defaults={
                "nearest_neighbor_hopping": (0.1, 2.0, 61),
                "next_nearest_neighbor_hopping": (0.0, 0.5, 61),
                "phase": (0.0, 3.141592653589793, 61),
                "sublattice_mass": (-2.0, 2.0, 81),
            },
        ),
    )
)
