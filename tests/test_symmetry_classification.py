import pytest

from toposc_lab.observables.results import StandardizedObservable
from toposc_lab.topology.symmetry import (
    AltlandZirnbauerClass,
    AntiunitarySymmetry,
    AntiunitarySymmetryKind,
    SymmetryClassification,
    classify_altland_zirnbauer,
)


@pytest.mark.parametrize(
    ("time_reversal_square", "particle_hole_square", "chiral", "expected"),
    [
        (None, None, False, AltlandZirnbauerClass.A),
        (None, None, True, AltlandZirnbauerClass.AIII),
        (1, None, False, AltlandZirnbauerClass.AI),
        (1, 1, True, AltlandZirnbauerClass.BDI),
        (None, 1, False, AltlandZirnbauerClass.D),
        (-1, 1, True, AltlandZirnbauerClass.DIII),
        (-1, None, False, AltlandZirnbauerClass.AII),
        (-1, -1, True, AltlandZirnbauerClass.CII),
        (None, -1, False, AltlandZirnbauerClass.C),
        (1, -1, True, AltlandZirnbauerClass.CI),
    ],
)
def test_all_ten_altland_zirnbauer_classes(
    time_reversal_square: int | None,
    particle_hole_square: int | None,
    chiral: bool,
    expected: AltlandZirnbauerClass,
) -> None:
    classification = SymmetryClassification.from_signature(
        time_reversal_square=time_reversal_square,
        particle_hole_square=particle_hole_square,
        chiral_symmetry=chiral,
    )

    assert classification.altland_zirnbauer_class is expected
    assert classification.signature == (
        time_reversal_square,
        particle_hole_square,
        chiral,
    )
    assert classify_altland_zirnbauer(
        time_reversal_square=time_reversal_square,
        particle_hole_square=particle_hole_square,
        chiral_symmetry=chiral,
    ) is expected


def test_antiunitary_symmetries_are_explicitly_typed() -> None:
    time_reversal = AntiunitarySymmetry(
        AntiunitarySymmetryKind.TIME_REVERSAL,
        -1,
    )
    particle_hole = AntiunitarySymmetry(
        AntiunitarySymmetryKind.PARTICLE_HOLE,
        1,
    )
    classification = SymmetryClassification(
        time_reversal=time_reversal,
        particle_hole=particle_hole,
        chiral_symmetry=True,
    )

    assert classification.altland_zirnbauer_class is AltlandZirnbauerClass.DIII
    assert classification.has_time_reversal_symmetry
    assert classification.has_particle_hole_symmetry


@pytest.mark.parametrize(
    ("time_reversal_square", "particle_hole_square", "chiral"),
    [
        (1, None, True),
        (None, 1, True),
        (1, 1, False),
        (-1, -1, False),
    ],
)
def test_inconsistent_tenfold_way_signatures_are_rejected(
    time_reversal_square: int | None,
    particle_hole_square: int | None,
    chiral: bool,
) -> None:
    with pytest.raises(ValueError, match="tenfold-way"):
        SymmetryClassification.from_signature(
            time_reversal_square=time_reversal_square,
            particle_hole_square=particle_hole_square,
            chiral_symmetry=chiral,
        )


@pytest.mark.parametrize("square", [0, 2, True])
def test_invalid_antiunitary_square_is_rejected(square: object) -> None:
    with pytest.raises((TypeError, ValueError), match="square"):
        AntiunitarySymmetry(
            AntiunitarySymmetryKind.PARTICLE_HOLE,
            square,  # type: ignore[arg-type]
        )


def test_wrong_antiunitary_kind_in_slot_is_rejected() -> None:
    particle_hole = AntiunitarySymmetry(
        AntiunitarySymmetryKind.PARTICLE_HOLE,
        1,
    )

    with pytest.raises(ValueError, match="wrong"):
        SymmetryClassification(
            time_reversal=particle_hole,
            particle_hole=None,
            chiral_symmetry=False,
        )


def test_symmetry_classification_has_stable_dataset_encoding() -> None:
    classification = SymmetryClassification.from_signature(
        time_reversal_square=1,
        particle_hole_square=1,
        chiral_symmetry=True,
    )

    assert isinstance(classification, StandardizedObservable)
    record = classification.to_observable_record()
    assert record.kind == "symmetry_classification"
    assert record.scalars == {
        "time_reversal_square": 1,
        "particle_hole_square": 1,
        "chiral_symmetry": True,
        "altland_zirnbauer_code": 3,
    }
    assert record.metadata["altland_zirnbauer_labels"] == [
        "A",
        "AIII",
        "AI",
        "BDI",
        "D",
        "DIII",
        "AII",
        "CII",
        "C",
        "CI",
    ]
