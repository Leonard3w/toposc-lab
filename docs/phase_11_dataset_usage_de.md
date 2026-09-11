# Phase 11: Versionierte Exact-Physics-Datasets

Die Dataset-API erweitert den bestehenden Namespace `toposc_lab.data`. Sie
speichert ausschließlich als exakt gekennzeichnete numerische Resultate; ein
späteres ML-Modell darf diese Records referenzieren, aber nicht als eigene
Ground Truth umdeuten.

## Record-Aufbau

Ein `DatasetRecord` enthält:

- einen verlustfreien `GeometryRecord`;
- `ModelParametersRecord` mit Modellname und Modellversion;
- ein `SpectrumRecord` mit sortierten Eigenwerten und expliziter
  Energiekonvention;
- versionierte Observable-, Topologie- und Robustness-Records;
- `ReproducibilityMetadata` mit Seed, Commit, explizitem Dirty-Flag, Paket-/Solverversion,
  Einstellungen, Toleranzen, UTC-Zeitstempel und Runtime-Angaben.

Große Eigenzustände werden nie automatisch inline gespeichert. Dafür ist
optional eine `ArtifactReference` mit URI, Format, SHA-256, Form und Datentyp
vorgesehen. Komplexe Modellparameter werden im JSON-Codec explizit als Real-
und Imaginärteil getaggt.

## Geometrie-Identitäten

Zwei Identitäten haben bewusst verschiedene Bedeutungen:

- `exact_id` hasht das vollständige Geometriearchiv. Labels, Koordinaten,
  Metadaten und Kantenorientierungen gehören zu dieser Identität.
- `family_fingerprint` ist der vorhandene relabelinginvariante 1-WL-Hash. Er
  ist nur ein Gruppierungs-/Vorauswahlwert und kein Isomorphiebeweis.

`assess_geometry_duplicate` bestätigt gleiche Fingerprints deshalb durch
einen exakten, begrenzten Graph-Isomorphietest. Erschöpft der Test sein Budget,
lautet das Ergebnis ausdrücklich `possible_fingerprint_collision` statt
„Duplikat“. Die optionale Koordinatenrelation vergleicht paarweise Abstände und
ist damit invariant unter Translation, Rotation, Spiegelung und erlaubtem
Site-Relabeling.

## Schreiben und Laden

```python
from toposc_lab.data import (
    DuplicateRecordPolicy,
    ExactPhysicsDataset,
    append_dataset_record,
    load_dataset,
    save_dataset,
)

save_dataset("results/exact_dataset.json", ExactPhysicsDataset((record,)))
dataset = load_dataset("results/exact_dataset.json")

append_dataset_record(
    "results/exact_dataset.json",
    another_record,
    duplicate_policy=DuplicateRecordPolicy.REJECT,
)
```

Writes werden zuerst vollständig validiert und anschließend über eine Datei im
Zielverzeichnis atomar publiziert. Vorhandene Dateien werden ohne explizites
`overwrite=True` nicht ersetzt. Append-Duplikate werden je nach expliziter
Policy verworfen, übersprungen oder ersetzt.

Der normale Loader akzeptiert nur die aktuelle Schema-Version. Die einzig
definierte Altversion ist der kompatible Container-Pilot v0, dessen Records
bereits v1 waren; er kann ausschließlich durch `migrate_dataset_bytes`
explizit auf v1 gehoben werden. Unbekannte Layouts werden nicht geraten.

## Leakage-freier Split

`split_dataset` weist immer ganze Geometriefamilien Train, Validation oder Test
zu. Ein explizites `family_label` hat Vorrang; andernfalls wird konservativ der
relabelinginvariante Familien-Fingerprint verwendet. Eine mögliche Hash-
Kollision kann dadurch höchstens zu Übergruppierung führen, nicht zu Leakage.
Seed, Ziel-/Ist-Anteile, Record-/Gruppenzahlen und Leakage-Diagnostik werden im
`DatasetSplitDiagnostics`-Record festgehalten.

Der kleine wissenschaftliche Acceptance-Lauf
`run_dataset_foundation_benchmark` diagonalisiert drei Kitaev-Ketten mit
fixiertem Seed und enthält sowohl bekannte topologische als auch triviale
Pfaffian-Referenzen. Er prüft anschließend Validierung, Write/Load-Round-trip
und familiengruppierten Split.
