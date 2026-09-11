"""Deterministic family-grouped dataset splits that prevent geometry leakage."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np

from toposc_lab.data.dataset_storage import ExactPhysicsDataset


@dataclass(frozen=True, slots=True)
class DatasetSplitConfig:
    """Fractions and seed for one reproducible grouped split."""

    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 0

    def __post_init__(self) -> None:
        fractions = tuple(
            _fraction(getattr(self, name), name)
            for name in ("train_fraction", "validation_fraction", "test_fraction")
        )
        if not np.isclose(sum(fractions), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("split fractions must sum to one")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise TypeError("seed must be an integer")
        if int(self.seed) < 0:
            raise ValueError("seed must be nonnegative")
        object.__setattr__(self, "train_fraction", fractions[0])
        object.__setattr__(self, "validation_fraction", fractions[1])
        object.__setattr__(self, "test_fraction", fractions[2])
        object.__setattr__(self, "seed", int(self.seed))


@dataclass(frozen=True, slots=True)
class DatasetSplitDiagnostics:
    """Compact evidence about balance and leakage for a completed split."""

    seed: int
    record_counts: Mapping[str, int]
    group_counts: Mapping[str, int]
    target_fractions: Mapping[str, float]
    achieved_fractions: Mapping[str, float]
    leakage_groups: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """Record IDs and the family assignment that produced the split."""

    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    group_assignments: Mapping[str, str]
    diagnostics: DatasetSplitDiagnostics

    def __post_init__(self) -> None:
        groups = dict(self.group_assignments)
        if any(split not in {"train", "validation", "test"} for split in groups.values()):
            raise ValueError("group assignments contain an unsupported split")
        all_ids = (*self.train_ids, *self.validation_ids, *self.test_ids)
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("record identifiers occur in more than one split")
        object.__setattr__(self, "group_assignments", MappingProxyType(dict(sorted(groups.items()))))


def split_dataset(
    dataset: ExactPhysicsDataset,
    *,
    config: DatasetSplitConfig | None = None,
) -> DatasetSplit:
    """Split whole related/isomorphic geometry families as indivisible groups.

    Explicit ``family_label`` values take precedence. Otherwise the conservative
    relabeling-invariant graph fingerprint is used. Potential fingerprint
    collisions are therefore over-grouped, which can reduce balance but cannot
    leak a possible isomorphic family across train and evaluation data.
    """
    if not isinstance(dataset, ExactPhysicsDataset):
        raise TypeError("dataset must be ExactPhysicsDataset")
    if config is None:
        config = DatasetSplitConfig()
    if not isinstance(config, DatasetSplitConfig):
        raise TypeError("config must be DatasetSplitConfig")
    if not dataset.records:
        raise ValueError("dataset must contain at least one record")

    groups: dict[str, list[str]] = {}
    for record in dataset.records:
        family_key = (
            f"label:{record.geometry.family_label}"
            if record.geometry.family_label is not None
            else f"fingerprint:{record.geometry.family_fingerprint}"
        )
        groups.setdefault(family_key, []).append(record.record_id)
    if len(groups) < 3:
        raise ValueError("at least three independent geometry families are required")

    random_generator = random.Random(config.seed)
    group_keys = sorted(groups)
    random_generator.shuffle(group_keys)
    randomized_order = {key: index for index, key in enumerate(group_keys)}
    group_keys.sort(key=lambda key: (-len(groups[key]), randomized_order[key]))

    split_names = ("train", "validation", "test")
    fractions = {
        "train": config.train_fraction,
        "validation": config.validation_fraction,
        "test": config.test_fraction,
    }
    counts = {name: 0 for name in split_names}
    assignments: dict[str, str] = {}
    total = len(dataset.records)
    for key in group_keys:
        selected = min(
            split_names,
            key=lambda name: (
                counts[name] / (fractions[name] * total),
                split_names.index(name),
            ),
        )
        assignments[key] = selected
        counts[selected] += len(groups[key])

    identifiers: dict[str, list[str]] = {name: [] for name in split_names}
    for key, record_ids in groups.items():
        identifiers[assignments[key]].extend(record_ids)
    sorted_identifiers = {
        name: tuple(sorted(values)) for name, values in identifiers.items()
    }
    group_counts = {
        name: sum(value == name for value in assignments.values()) for name in split_names
    }
    achieved = {name: counts[name] / total for name in split_names}
    leakage = _leakage_groups(assignments)
    diagnostics = DatasetSplitDiagnostics(
        seed=config.seed,
        record_counts=MappingProxyType(counts),
        group_counts=MappingProxyType(group_counts),
        target_fractions=MappingProxyType(fractions),
        achieved_fractions=MappingProxyType(achieved),
        leakage_groups=leakage,
    )
    return DatasetSplit(
        train_ids=sorted_identifiers["train"],
        validation_ids=sorted_identifiers["validation"],
        test_ids=sorted_identifiers["test"],
        group_assignments=assignments,
        diagnostics=diagnostics,
    )


def _leakage_groups(assignments: Mapping[str, str]) -> tuple[str, ...]:
    # One mapping value per group makes leakage structurally impossible; retain
    # this explicit diagnostic for persisted reports and future split adapters.
    return tuple(
        sorted(key for key, split in assignments.items() if split not in {"train", "validation", "test"})
    )


def _fraction(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must be finite and strictly between zero and one")
    return result
