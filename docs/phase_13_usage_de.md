# Phase 13: Active Learning

`toposc_lab.active_learning` priorisiert exakte Simulationen. Das Modul speichert
keine ML-Vorhersagen als physikalische Labels. Suchräume sind explizite endliche
Kataloge aus `Candidate(GeometryRecord, ModelParametersRecord)`; der Pool wird
ohne Wiederholung mit einem festen Seed gezogen. Andere Geometriefamilien können
über denselben Katalog und einen passenden exakten Evaluator angebunden werden.

## Discovery Gate A reproduzieren

Im Repository-Hauptverzeichnis, mit einem noch nicht belegten Ausgabeordner:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='src'
$env:BLIS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
.venv\Scripts\python.exe -m toposc_lab.active_learning.benchmark --output results/phase13-gate-reproduction
```

Die fünf Seeds und das Protokoll stehen in
[`phase_13_benchmark_protocol.md`](decisions/phase_13_benchmark_protocol.md).
`report.json` enthält den Vergleich; pro Seed liegen die exakten Datasets,
Bestätigungsrechnungen, das reservierte Testset und der AL-Checkpoint vor.
Der empirische Gate-Status muss zusammen mit der vollständigen Testsuite beurteilt
werden. Laufzeiten und Ausführungszeitstempel müssen bei Wiederholungen nicht gleich
sein; Auswahl, Spektren und Qualitätsmetriken müssen reproduzierbar sein.

## Eigene Kampagne

```python
from toposc_lab.active_learning import (
    AcquisitionConfig, CampaignConfig, CandidateSpace, CycleConfig,
    TrainingConfig, run_campaign,
)

# candidates: ungelabelte Candidate-Einträge mit Geometrie und Modellparametern
# train: ExactPhysicsDataset aus mindestens vier exakten Simulationen
# heldout: eingefrorene Validierungs-/Testrecords, z.B. aus data.split_dataset
# target: ml.RegressionTargetDefinition; ein größerer Wert wird bevorzugt
# exact_evaluator(candidate, seed) -> DatasetRecord aus der exakten Physikpipeline
config = CampaignConfig(
    cycle=CycleConfig(
        training=TrainingConfig(target),
        acquisition=AcquisitionConfig(batch_size=4, exploration_fraction=0.25),
        pool_size=40,
    ),
    seed=13001, cycles=4, exact_budget=16,
    evaluator_identifier="my-exact-evaluator.v1",
    code_version="commit-and-source-hash",
)
result = run_campaign(
    CandidateSpace(tuple(candidates)), train,
    config=config, evaluator=exact_evaluator,
    reserved_records=tuple(heldout),
    checkpoint_path="results/my-active-learning/checkpoint.json",
)
```

Mit denselben Argumenten wird der Checkpoint fortgesetzt. `stop_after=1` beendet
einen Lauf kontrolliert nach dem ersten Zyklus. Änderungen an Konfiguration,
Suchraum, initialem Dataset, Testset oder Quellversion werden beim Resume abgewiesen.
Das Budget zählt neue Evaluatorversuche; initiale Trainingslabels und unabhängige
Gewinnerprüfungen sind zusätzlich zu budgetieren. Ein Evaluatoraufruf soll genau
eine exakte Simulation ausführen; zusätzliche Robustheitsensembles sind separat
abzurechnen. Es gibt einen schreibenden Prozess pro Checkpoint.

OOD ist ein Warnsignal, keine Aussage über ungültige Physik. OOD-Kandidaten können
Explorationsplätze erhalten, treiben aber keine Exploitation durch ihren
vorhergesagten Score. Ungültige numerische Vorhersagen lösen explizit protokollierte,
reproduzierbare Zufallsexploration aus. Ein geometrisch ungültiger Katalog wird vor
der Simulation abgewiesen; ein inkompatibler Modellkandidat muss vom Evaluator
abgewiesen werden. Fehlgeschlagene exakte Versuche liefern keine Labels.

Die Speicherung erfolgt atomar. Das Journal sichert abgeschlossene Simulationen
vor dem Zyklusabschluss. Ein Prozessabbruch während einer Simulation ohne gesichertes
Ergebnis wird beim Resume konservativ als verbrauchter Fehlversuch behandelt;
der Versuch wird nicht wiederholt. Ein solcher Abbruch kann den weiteren Suchpfad
verändern. Ein Abbruch nach gesichertem exaktem Ergebnis reproduziert denselben Pfad.
Beschädigte temporäre Dateien werden ignoriert; beschädigte veröffentlichte
Checkpoints werden zurückgewiesen. Multiwriter-/verteiltes Scheduling ist nicht Teil
dieser Phase.

Die Unsicherheiten sind Schätzungen aus dem Phase-12-Bootstrapmodell und keine
garantierten Konfidenzintervalle. Reservierte Testfamilien werden niemals zum Fit
von Features, Regressor oder OOD-Schwelle verwendet. Der Gate-Benchmark berichtet
deshalb Testfehler und Fehler auf tatsächlich akquirierten Kandidaten getrennt.

## Phase 13R reproduzieren

Recovery und erneutes Gate sind abgeschlossen: **FAIL / NO-GO**, keine Phase 14.
Der [Entscheidungsbericht](decisions/phase_13r_recovery.md) enthält Diagnose,
Konfidenzintervalle, Simulationsbilanz und verbleibende Grenzen. Mit den obigen
Umgebungsvariablen und einem neuen Ausgabeordner:

```powershell
.venv\Scripts\python.exe -m toposc_lab.active_learning.recovery --output results/phase13r-reproduction
.venv\Scripts\python.exe scripts/phase_13r_audit.py results/phase13r-reproduction
```

Der erste Befehl führt das eingefrorene Protokoll mit 20 Seeds und 1.680 exakten
Simulationen aus. Der zweite prüft gespeicherte Artefakte und erstellt Fehlerdiagnosen
sowie Lernkurven ohne weitere Physiksimulationen. `report.json` enthält die
gepaarten statistischen Vergleiche; der vollständige Testlauf bleibt zusätzlich
erforderlich. Seed-Wiederholungen prüfen denselben begrenzten physikalischen Suchraum.

`recovery_config()` in `toposc_lab.active_learning.recovery` aktiviert
`TrainingConfig.flag_extrapolation=True`, `AcquisitionConfig.scaling="target_scale"`
und `diversity_radius_fraction=0.5`. Historische Defaults bleiben erhalten.
Die Separation verwendet ausschließlich Trainingsfeatures; unzureichend große
Pools führen zu protokollierter Lockerung statt stiller Budgetänderung.
Alte und neue Konfigurationen dürfen nicht im selben Checkpoint vermischt werden.
Die bestehende Bootstrap-Unsicherheit bleibt eine Schätzung ohne Deckungsgarantie.
