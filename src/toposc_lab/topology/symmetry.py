"""Declarative tenfold-way symmetry classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral
from typing import Literal, TypeAlias

from toposc_lab.observables.results import ObservableRecord

SymmetrySquare: TypeAlias = Literal[-1, 1]
SymmetrySignature: TypeAlias = tuple[
    SymmetrySquare | None,
    SymmetrySquare | None,
    bool,
]


class AntiunitarySymmetryKind(str, Enum):
    """Supported antiunitary symmetries in the tenfold way."""

    TIME_REVERSAL = "time_reversal"
    PARTICLE_HOLE = "particle_hole"


class AltlandZirnbauerClass(str, Enum):
    """The ten Altland-Zirnbauer symmetry classes."""

    A = "A"
    AIII = "AIII"
    AI = "AI"
    BDI = "BDI"
    D = "D"
    DIII = "DIII"
    AII = "AII"
    CII = "CII"
    C = "C"
    CI = "CI"


AZ_CLASS_ORDER: tuple[AltlandZirnbauerClass, ...] = tuple(AltlandZirnbauerClass)

_AZ_BY_SIGNATURE: dict[SymmetrySignature, AltlandZirnbauerClass] = {
    (None, None, False): AltlandZirnbauerClass.A,
    (None, None, True): AltlandZirnbauerClass.AIII,
    (1, None, False): AltlandZirnbauerClass.AI,
    (1, 1, True): AltlandZirnbauerClass.BDI,
    (None, 1, False): AltlandZirnbauerClass.D,
    (-1, 1, True): AltlandZirnbauerClass.DIII,
    (-1, None, False): AltlandZirnbauerClass.AII,
    (-1, -1, True): AltlandZirnbauerClass.CII,
    (None, -1, False): AltlandZirnbauerClass.C,
    (1, -1, True): AltlandZirnbauerClass.CI,
}


@dataclass(frozen=True, slots=True)
class AntiunitarySymmetry:
    """Declaration of an antiunitary symmetry and the sign of its square.

    The numerical unitary part of the antiunitary operator is intentionally not
    stored here. Operator validation belongs to the numerical-validation layer.
    """

    kind: AntiunitarySymmetryKind
    square: SymmetrySquare

    def __post_init__(self) -> None:
        if not isinstance(self.kind, AntiunitarySymmetryKind):
            raise TypeError("kind must be an AntiunitarySymmetryKind")
        square = _symmetry_square(self.square, name="square")
        object.__setattr__(self, "square", square)


@dataclass(frozen=True, slots=True)
class SymmetryClassification:
    """Consistent particle-hole, time-reversal, chiral, and AZ declaration."""

    time_reversal: AntiunitarySymmetry | None
    particle_hole: AntiunitarySymmetry | None
    chiral_symmetry: bool
    altland_zirnbauer_class: AltlandZirnbauerClass = field(init=False)

    def __post_init__(self) -> None:
        _validate_antiunitary_slot(
            self.time_reversal,
            expected_kind=AntiunitarySymmetryKind.TIME_REVERSAL,
            name="time_reversal",
        )
        _validate_antiunitary_slot(
            self.particle_hole,
            expected_kind=AntiunitarySymmetryKind.PARTICLE_HOLE,
            name="particle_hole",
        )
        if not isinstance(self.chiral_symmetry, bool):
            raise TypeError("chiral_symmetry must be a boolean")

        signature = self.signature
        try:
            symmetry_class = _AZ_BY_SIGNATURE[signature]
        except KeyError as error:
            raise ValueError(
                "symmetry combination is not a valid tenfold-way signature"
            ) from error
        object.__setattr__(self, "altland_zirnbauer_class", symmetry_class)

    @classmethod
    def from_signature(
        cls,
        *,
        time_reversal_square: int | None,
        particle_hole_square: int | None,
        chiral_symmetry: bool,
    ) -> SymmetryClassification:
        """Build a classification from ``T^2``, ``C^2``, and chiral presence."""
        time_reversal = (
            None
            if time_reversal_square is None
            else AntiunitarySymmetry(
                AntiunitarySymmetryKind.TIME_REVERSAL,
                _symmetry_square(time_reversal_square, name="time_reversal_square"),
            )
        )
        particle_hole = (
            None
            if particle_hole_square is None
            else AntiunitarySymmetry(
                AntiunitarySymmetryKind.PARTICLE_HOLE,
                _symmetry_square(particle_hole_square, name="particle_hole_square"),
            )
        )
        return cls(
            time_reversal=time_reversal,
            particle_hole=particle_hole,
            chiral_symmetry=chiral_symmetry,
        )

    @property
    def signature(self) -> SymmetrySignature:
        """Return the canonical ``(T^2, C^2, chiral)`` signature."""
        return (
            None if self.time_reversal is None else self.time_reversal.square,
            None if self.particle_hole is None else self.particle_hole.square,
            self.chiral_symmetry,
        )

    @property
    def has_time_reversal_symmetry(self) -> bool:
        return self.time_reversal is not None

    @property
    def has_particle_hole_symmetry(self) -> bool:
        return self.particle_hole is not None

    def to_observable_record(self) -> ObservableRecord:
        """Return a stable numerical encoding for datasets and ML."""
        labels = tuple(symmetry_class.value for symmetry_class in AZ_CLASS_ORDER)
        return ObservableRecord(
            kind="symmetry_classification",
            scalars={
                "time_reversal_square": (
                    0 if self.time_reversal is None else self.time_reversal.square
                ),
                "particle_hole_square": (
                    0 if self.particle_hole is None else self.particle_hole.square
                ),
                "chiral_symmetry": self.chiral_symmetry,
                "altland_zirnbauer_code": AZ_CLASS_ORDER.index(
                    self.altland_zirnbauer_class
                ),
            },
            metadata={"altland_zirnbauer_labels": labels},
        )


def classify_altland_zirnbauer(
    *,
    time_reversal_square: int | None,
    particle_hole_square: int | None,
    chiral_symmetry: bool,
) -> AltlandZirnbauerClass:
    """Return the AZ class for one valid tenfold-way signature."""
    return SymmetryClassification.from_signature(
        time_reversal_square=time_reversal_square,
        particle_hole_square=particle_hole_square,
        chiral_symmetry=chiral_symmetry,
    ).altland_zirnbauer_class


def _symmetry_square(value: object, *, name: str) -> SymmetrySquare:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be either +1 or -1")
    value = int(value)
    if value == 1:
        return 1
    if value == -1:
        return -1
    raise ValueError(f"{name} must be either +1 or -1")


def _validate_antiunitary_slot(
    symmetry: AntiunitarySymmetry | None,
    *,
    expected_kind: AntiunitarySymmetryKind,
    name: str,
) -> None:
    if symmetry is None:
        return
    if not isinstance(symmetry, AntiunitarySymmetry):
        raise TypeError(f"{name} must be an AntiunitarySymmetry or None")
    if symmetry.kind is not expected_kind:
        raise ValueError(f"{name} has the wrong antiunitary symmetry kind")
