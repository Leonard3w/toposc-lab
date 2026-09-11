# Phase 12 — Surrogate Ready Gate

date: 2026-09-11
gate: SURROGATE_READY
result: PASS
base_commit: 68151d7

## Entscheidung

Phase 12 ist als Priorisierungs-Surrogate akzeptiert. Der Default ist das
handgefertigte Featuremodell mit Gradient Boosting über Entscheidungsstümpfe.
Die minimale GNN wird nicht als Default übernommen: Sie erreichte weder auf
Validation noch auf dem unangetasteten Testset einen verifizierten Mehrwert
gegenüber dem einfachen Modell. Das ist eine Modellwahl für Rechenpriorisierung
und keine wissenschaftliche Aussage über die zugrunde liegende Physik.

## Gate-Evidenz

- Fester Benchmark-Seed: `12120`.
- 30 exakt diagonalisierte offene Kitaev-Ketten, gruppiert nach unabhängigen
  Geometriefamilien; Split: 18 Train / 6 Validation / 6 Test.
- Test-RMSE Default: `0.1300120929`.
- Test-RMSE trivialer Trainingsmittelwert: `0.5957505327`.
- Die GNN bestand die vorab fixierten Effektgrößen- und gepaarten
  Bootstrap-Kriterien nicht; einfaches Modell blieb Default.
- Unsicherheit gegen Held-out-Fehler: Spearman-Rangkorrelation
  `0.9428571429`; MAE der hoch-unsicheren Hälfte `0.5594695320` gegenüber
  `0.1696421156` in der niedrig-unsicheren Hälfte.
- 90%-Intervalle: empirische Coverage `0.5` auf nur sechs Testfällen. Das ist
  unzureichend für eine starke Coverage-Aussage und wird von der API als grobe
  Kleinstichprobe markiert. Die Intervalle bleiben ein messbares
  Akquisitionssignal, keine Garantie.
- Ein stark extrapolierter Referenzgraph wurde als Feature-OOD markiert.
- Jede Testvorhersage verweist auf ihre exakte Dataset-Record-ID, Schema v1 und
  den Dataset-Fingerprint
  `exact-dataset-v1-sha256:a909a49bbb0f7312fd9c9e49eed233a24e4f758f85f7421d15d8af8d245ecb94`.

## Methodischer Vertrag

Das Feature-Schema wird nur aus Train-Records bestimmt. Modell-/Schwellenwahl
nutzt Validation; Testdaten dienen ausschließlich der eingefrorenen
Abschlussprüfung. Geometriefamilien bleiben über `DatasetSplit` ungeteilt.
Targets werden nur aus versionierten exakten Observable-, gültigen Topologie-
oder Robustheitsrecords gelesen; unzulässige Topologieresultate werden maskiert
und nie in Labels umgedeutet.

OOD bedeutet ausschließlich „ungewöhnlich im trainierten Featureraum“. Es ist
weder ein Beleg für schlechte noch für ungültige Physik. Kandidaten müssen vor
jeder Aufnahme als Label oder Gewinner exakt simuliert werden.

## Bekannte Grenzen

Der Gate-Benchmark ist klein und auf eine kontrollierte Kitaev-Kettenfamilie
begrenzt. Besonders die Coverage-Schätzung ist wegen sechs Testfällen grob.
Vor einem breiteren Discovery-Einsatz müssen Unsicherheit und Kalibrierung auf
größeren, vielfältigeren exakten Datensätzen erneut geprüft werden. Feature-
Wichtigkeit, OOD-Score und ML-Fehler dürfen nicht kausal oder physikalisch
interpretiert werden.
