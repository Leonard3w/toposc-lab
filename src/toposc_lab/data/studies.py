"""Reproduzierbares Speichern numerischer TopOSC-Lab-Studien.

Eine Studie trennt zwei Dinge klar voneinander:

* :class:`StudyMetadata` beschreibt, wie ein Ergebnis entstanden ist.
* :class:`StudyData` enthält die numerischen Arrays, die geplottet oder
  weiter analysiert werden sollen.

Das erste Dateiformat ist ein komprimiertes ``.npz``-Archiv. Die Metadaten
werden darin als JSON abgelegt, die Arrays als normale NumPy-Arrays. Dadurch
bleiben Dateien portabel und werden ohne Pickle geladen.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel as PydanticBaseModel, Field

from toposc_lab.core.results import ParameterScanResult

_METADATA_KEY = "__toposc_lab_metadata_json__"


def _package_version() -> str:
    """Liefere die installierte Paketversion, auch im Entwicklungsmodus."""
    try:
        return version("toposc-lab")
    except PackageNotFoundError:
        # Der Quellcode kann auch ohne Paketinstallation importiert werden.
        return "unknown"


def _utc_now() -> datetime:
    """Erzeuge einen eindeutigen, zeitzonenbewussten UTC-Zeitstempel."""
    return datetime.now(timezone.utc)


class StudyMetadata(PydanticBaseModel):
    """Beschreibung einer reproduzierbaren numerischen Studie.

    ``model_parameters`` enthält die festen physikalischen Modellparameter.
    Die tatsächlich gescannten Werte gehören als Array in :class:`StudyData`.
    ``scan_parameters`` hält Informationen über den Scan selbst, etwa den
    Namen der Scanachse, die k-Gitterauflösung oder einen Solver.
    """

    study_name: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    model_parameters: dict[str, Any] = Field(default_factory=dict)
    scan_parameters: dict[str, Any] = Field(default_factory=dict)
    random_seed: int | None = None
    created_at: datetime = Field(default_factory=_utc_now)
    package_version: str = Field(default_factory=_package_version)
    description: str | None = None
    tags: tuple[str, ...] = ()
    schema_version: int = Field(default=1, ge=1)


@dataclass(frozen=True)
class StudyData:
    """Metadaten und numerische Ergebnisse einer gespeicherten Studie."""

    metadata: StudyMetadata
    arrays: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, StudyMetadata):
            raise TypeError("metadata must be a StudyMetadata instance")

        if not self.arrays:
            raise ValueError("arrays must contain at least one result array")

        prepared_arrays: dict[str, np.ndarray] = {}

        for name, values in self.arrays.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError(
                    "Array names must be non-empty Python-style identifiers"
                )

            if name == _METADATA_KEY:
                raise ValueError(f"{name!r} is reserved for study metadata")

            array = np.asarray(values).copy()

            if array.dtype.hasobject:
                raise ValueError(
                    f"Array {name!r} must not use object dtype"
                )

            array.setflags(write=False)
            prepared_arrays[name] = array

        object.__setattr__(self, "arrays", prepared_arrays)


def _normalise_study_path(path: str | Path) -> Path:
    """Stelle sicher, dass Studien als ``.npz`` gespeichert werden."""
    destination = Path(path)

    if destination.suffix == "":
        destination = destination.with_suffix(".npz")
    elif destination.suffix.lower() != ".npz":
        raise ValueError("Study files must use the .npz suffix")

    return destination


def save_study(path: str | Path, study: StudyData) -> Path:
    """Speichere eine Studie komprimiert und gebe ihren Dateipfad zurück.

    Der Zielordner wird bei Bedarf angelegt. Die JSON-Metadaten und die
    Arrays bleiben vollständig getrennt; zum Laden wird nie Pickle benutzt.
    """
    if not isinstance(study, StudyData):
        raise TypeError("study must be a StudyData instance")

    destination = _normalise_study_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        metadata_json = study.metadata.model_dump_json()
    except (TypeError, ValueError) as error:
        raise ValueError("Study metadata must be JSON serializable") from error

    np.savez_compressed(
        destination,
        **{
            _METADATA_KEY: np.asarray(metadata_json),
            **study.arrays,
        },
    )

    return destination


def load_study(path: str | Path) -> StudyData:
    """Lade eine vorher mit :func:`save_study` gespeicherte Studie."""
    source = _normalise_study_path(path)

    if not source.is_file():
        raise FileNotFoundError(f"Study file does not exist: {source}")

    try:
        with np.load(source, allow_pickle=False) as archive:
            if _METADATA_KEY not in archive.files:
                raise ValueError("Study file does not contain TopOSC-Lab metadata")

            metadata_value = archive[_METADATA_KEY]
            if metadata_value.ndim != 0:
                raise ValueError("Study metadata must be a scalar JSON value")

            metadata = StudyMetadata.model_validate_json(str(metadata_value.item()))
            arrays = {
                name: archive[name].copy()
                for name in archive.files
                if name != _METADATA_KEY
            }
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not load study file: {source}") from error

    return StudyData(metadata=metadata, arrays=arrays)


def study_from_parameter_scan(
    scan: ParameterScanResult,
    metadata: StudyMetadata,
    *,
    observables: Mapping[str, np.ndarray] | None = None,
) -> StudyData:
    """Erzeuge eine speicherbare Studie aus einem 1D-Parameterscan.

    Es werden die Scanachse, alle Eigenwertspektren und der kleinste
    Energiebetrag gespeichert. ZusÃ¤tzliche Observablen, etwa ``bulk_gaps``,
    ``chern_numbers`` oder ``ipr``, kÃ¶nnen Ã¼ber ``observables`` hinzugefÃ¼gt
    werden. Die oft sehr groÃŸen Eigenvektoren werden absichtlich nicht
    automatisch gespeichert.
    """
    if not isinstance(scan, ParameterScanResult):
        raise TypeError("scan must be a ParameterScanResult instance")

    if not isinstance(metadata, StudyMetadata):
        raise TypeError("metadata must be a StudyMetadata instance")

    arrays: dict[str, np.ndarray] = {
        "parameter_values": scan.parameter_values,
        "spectra": scan.spectra,
        "minimum_abs_energy": scan.gaps,
    }

    if observables is not None:
        for name, values in observables.items():
            if name in arrays:
                raise ValueError(
                    f"Observable name {name!r} conflicts with a standard array"
                )
            arrays[name] = np.asarray(values)

    scan_parameters = {
        **metadata.scan_parameters,
        "parameter_name": scan.parameter_name,
        **scan.metadata,
    }
    enriched_metadata = metadata.model_copy(
        update={"scan_parameters": scan_parameters}
    )

    return StudyData(metadata=enriched_metadata, arrays=arrays)
