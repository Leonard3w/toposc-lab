from __future__ import annotations

from dataset_fixtures import representative_dataset_record

from toposc_lab.data.dataset_splitting import DatasetSplitConfig, split_dataset
from toposc_lab.data.dataset_storage import ExactPhysicsDataset
from toposc_lab.geometry import chain


def _split_fixture() -> ExactPhysicsDataset:
    records = []
    seed = 0
    for size in range(3, 9):
        for repeat in range(2 if size in (3, 5) else 1):
            records.append(
                representative_dataset_record(
                    geometry=chain(size),
                    seed=seed,
                    family_label=f"chain_size_{size}",
                )
            )
            seed += 1
    return ExactPhysicsDataset(tuple(records))


def test_fixed_seed_split_is_reproducible_and_order_independent() -> None:
    dataset = _split_fixture()
    config = DatasetSplitConfig(train_fraction=0.6, validation_fraction=0.2, test_fraction=0.2, seed=2026)

    first = split_dataset(dataset, config=config)
    second = split_dataset(ExactPhysicsDataset(tuple(reversed(dataset.records))), config=config)

    assert first.train_ids == second.train_ids
    assert first.validation_ids == second.validation_ids
    assert first.test_ids == second.test_ids
    assert first.group_assignments == second.group_assignments
    assert first.diagnostics.leakage_groups == ()


def test_related_family_members_never_cross_split_boundaries() -> None:
    dataset = _split_fixture()
    result = split_dataset(dataset, config=DatasetSplitConfig(seed=11))
    membership = {
        record_id: split
        for split, identifiers in (
            ("train", result.train_ids),
            ("validation", result.validation_ids),
            ("test", result.test_ids),
        )
        for record_id in identifiers
    }

    records_by_family: dict[str, list[str]] = {}
    for record in dataset.records:
        assert record.geometry.family_label is not None
        records_by_family.setdefault(record.geometry.family_label, []).append(record.record_id)
    assert all(len({membership[item] for item in identifiers}) == 1 for identifiers in records_by_family.values())
    assert sum(result.diagnostics.record_counts.values()) == len(dataset.records)


def test_unlabeled_isomorphic_snapshots_use_the_same_conservative_group() -> None:
    first = representative_dataset_record(geometry=chain(4), seed=1, family_label=None)
    second = representative_dataset_record(geometry=chain(4), seed=2, family_label=None)
    third = representative_dataset_record(geometry=chain(5), seed=3, family_label=None)
    fourth = representative_dataset_record(geometry=chain(6), seed=4, family_label=None)

    result = split_dataset(ExactPhysicsDataset((first, second, third, fourth)))
    membership = {
        record_id: split
        for split, identifiers in (
            ("train", result.train_ids),
            ("validation", result.validation_ids),
            ("test", result.test_ids),
        )
        for record_id in identifiers
    }

    assert membership[first.record_id] == membership[second.record_id]


def test_split_requires_three_independent_families() -> None:
    dataset = ExactPhysicsDataset(
        (
            representative_dataset_record(geometry=chain(3), seed=1, family_label="a"),
            representative_dataset_record(geometry=chain(4), seed=2, family_label="b"),
        )
    )

    try:
        split_dataset(dataset)
    except ValueError as error:
        assert "three independent" in str(error)
    else:
        raise AssertionError("split_dataset should reject fewer than three families")
