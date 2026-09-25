"""Explicit physical-domain contract over the existing Geometry and exact adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from toposc_lab.geometry import Geometry, validate_geometry
from toposc_lab.research.physics import (
    FiniteSystemEvaluator,
    PhysicsAdapterDefinition,
    PhysicsProtocol,
    register_physics_adapter,
)
from toposc_lab.research.validation_diagnostics import ValidationEvaluator, probe_positions
from toposc_lab.search.phase_9_8_evaluation import _clipped_voronoi_areas

EMBEDDED_ADAPTER_ID = "phase19.embedded-chiral-p-wave.v1"


@dataclass(frozen=True)
class EmbeddedDomain:
    """Coordinate box and separate integration cell; no inferred graph boundary."""

    bounds: tuple[float, float, float, float] = (0.0, 7.0, 0.0, 7.0)
    boundary_shell: float = 0.875
    bulk_inset: float = 2.0
    cell_padding: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "bounds", tuple(float(v) for v in self.bounds))
        if len(self.bounds) != 4 or not np.isfinite(self.bounds).all():
            raise ValueError("Four finite bounds required")
        x0, x1, y0, y1 = self.bounds
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Domain must have positive area")
        values = (self.boundary_shell, self.bulk_inset, self.cell_padding)
        if not np.isfinite(values).all() or any(v <= 0 for v in values):
            raise ValueError("Domain distances must be positive finite numbers")
        if self.bulk_inset * 2 >= min(x1 - x0, y1 - y0):
            raise ValueError("Domain must contain an interior probe region")

    @property
    def center(self) -> tuple[float, float]:
        x0, x1, y0, y1 = self.bounds
        return ((x0 + x1) / 2, (y0 + y1) / 2)

    def distances(self, coordinates: np.ndarray) -> np.ndarray:
        x0, x1, y0, y1 = self.bounds
        return np.min(
            np.column_stack(
                (
                    coordinates[:, 0] - x0,
                    x1 - coordinates[:, 0],
                    coordinates[:, 1] - y0,
                    y1 - coordinates[:, 1],
                )
            ),
            axis=1,
        )

    def boundary(self, coordinates: np.ndarray) -> frozenset[int]:
        return frozenset(
            np.flatnonzero(self.distances(coordinates) <= self.boundary_shell).tolist()
        )

    def validate(self, geometry: Geometry) -> None:
        validate_geometry(geometry, require_connected=True).raise_for_errors()
        coordinates = geometry.coordinates
        if coordinates is None or coordinates.shape != (geometry.n_sites, 2):
            raise ValueError("Explicit 2D coordinates required")
        if len(np.unique(coordinates, axis=0)) != geometry.n_sites:
            raise ValueError("Coincident positions are not supported by the embedded adapter")
        if np.any(self.distances(coordinates) < -1e-10):
            raise ValueError("Coordinates outside declared physical box")
        if geometry.boundary_sites != self.boundary(coordinates) or not geometry.boundary_sites:
            raise ValueError("Boundary must match the declared physical shell")
        for edge in geometry.edges:
            expected = coordinates[edge.target] - coordinates[edge.source]
            if edge.boundary_crossing or (
                edge.displacement is not None and not np.array_equal(edge.displacement, expected)
            ):
                raise ValueError("Only open physical displacements are supported")

    def probes(self) -> list[list[float]]:
        x0, x1, y0, y1 = self.bounds
        # Preserve the exact historical probe set for its square-domain defaults.
        if x0 == y0 == 0 and x1 == y1 and x1.is_integer() and self.bulk_inset == 2:
            return probe_positions(int(x1) + 1)
        cx, cy = self.center
        xs = (x0 + self.bulk_inset, cx, x1 - self.bulk_inset)
        ys = (y0 + self.bulk_inset, cy, y1 - self.bulk_inset)
        points = {(x, y) for x in xs for y in ys}
        points.update((x, cy) for x in (x0, x0 + 1, x1 - 1, x1))
        points.update((cx, y) for y in (y0, y0 + 1, y1 - 1, y1))
        return [list(p) for p in sorted(points)]

    def areas(self, coordinates: np.ndarray) -> np.ndarray:
        x0, x1, y0, y1 = self.bounds
        p = self.cell_padding
        # Existing validated clipping algorithm; input/order is unique-position order.
        return _clipped_voronoi_areas(coordinates, cell=(x0 - p, x1 + p, y0 - p, y1 + p))


@dataclass(frozen=True)
class EmbeddedPhysicsProtocol(PhysicsProtocol):
    adapter_id: str = EMBEDDED_ADAPTER_ID
    probe_rule: str = "declared_embedded_domain_center"
    domain: EmbeddedDomain = field(default_factory=EmbeddedDomain)

    def __post_init__(self) -> None:
        if isinstance(self.domain, dict):
            object.__setattr__(self, "domain", EmbeddedDomain(**self.domain))
        if not isinstance(self.domain, EmbeddedDomain) or self.adapter_id != EMBEDDED_ADAPTER_ID:
            raise ValueError("Invalid embedded adapter/domain")
        if self.probe_rule != "declared_embedded_domain_center":
            raise ValueError("Embedded probe rule is frozen")
        # Reuse every old numerical/physical validation; only applicability changes.
        legacy = asdict(self)
        for key in ("domain", "adapter_id", "probe_rule"):
            legacy.pop(key)
        checked = PhysicsProtocol(**legacy)
        for key in ("disorder_widths", "disorder_seeds", "kappas", "validators"):
            object.__setattr__(self, key, getattr(checked, key))


class EmbeddedSystemEvaluator(FiniteSystemEvaluator):
    adapter_id = EMBEDDED_ADAPTER_ID
    evidence_scope = "Finite embedded-domain center evidence; no phase or Majorana claim."
    geometry_family = "embedded_2d"

    def probe(self, geometry: Geometry) -> tuple[float, float]:
        domain = self.protocol.domain  # type: ignore[attr-defined]
        domain.validate(geometry)
        return domain.center


class EmbeddedValidationEvaluator(ValidationEvaluator):
    def __init__(self, protocol: EmbeddedPhysicsProtocol, provenance: Any = None) -> None:
        super().__init__(
            protocol,
            provenance,
            domain=protocol.domain,
            base=EmbeddedSystemEvaluator(protocol, provenance),
        )


register_physics_adapter(
    PhysicsAdapterDefinition(
        EMBEDDED_ADAPTER_ID,
        EmbeddedSystemEvaluator.evidence_scope,
        EmbeddedPhysicsProtocol,
        EmbeddedValidationEvaluator,
    )
)
