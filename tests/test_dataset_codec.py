from __future__ import annotations

from copy import deepcopy

import pytest
from dataset_fixtures import representative_dataset_record

from toposc_lab.data.dataset_codec import (
    DatasetCodecError,
    canonical_record_bytes,
    record_from_dict,
    record_to_dict,
)


def test_representative_record_round_trip_is_equal_and_deterministic() -> None:
    record = representative_dataset_record()

    restored = record_from_dict(record_to_dict(record))

    assert canonical_record_bytes(restored) == canonical_record_bytes(record)
    assert restored.record_id == record.record_id
    assert restored.model.parameters["pairing"] == 0.7 + 0.1j
    assert restored.geometry.archive == record.geometry.archive


def test_record_id_covers_scientific_and_provenance_content() -> None:
    first = representative_dataset_record(seed=1)
    second = representative_dataset_record(seed=2)

    assert first.record_id != second.record_id


def test_codec_rejects_tampered_record_content() -> None:
    payload = deepcopy(record_to_dict(representative_dataset_record()))
    payload["spectrum"]["eigenvalues"][1] = -0.3

    with pytest.raises(DatasetCodecError, match="record_id"):
        record_from_dict(payload)


def test_codec_rejects_unknown_or_incompatible_fields() -> None:
    payload = record_to_dict(representative_dataset_record())
    payload["unknown"] = True
    with pytest.raises(DatasetCodecError, match="fields mismatch"):
        record_from_dict(payload)

    payload = record_to_dict(representative_dataset_record())
    payload["schema_version"] = 99
    with pytest.raises(DatasetCodecError, match="incompatible"):
        record_from_dict(payload)
