# Phase 13 — Discovery Gate A

Datum: 2026-09-11. Block C, Aufgaben 13.1–13.12 implementiert.
**Empirisches Ergebnis: FAIL / NO-GO. Phase 14 wird nicht begonnen.**
Vollständige Testsuite: **2748 passed in 384.55 s**. Kommando:
`PYTHONDONTWRITEBYTECODE=1 BLIS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=results/pytest-phase13-final-full`.
Ruff für geänderte Quellen/Tests und `git diff --check`: PASS.

## 1. Exakte Simulationen und Vergleichsrahmen

Der maßgebliche, korrigierte Gate-Lauf umfasst **420 exakte Simulationen**:
360 Suchsimulationen (5 Seeds × 3 Verfahren × 24), 15 reservierte Testrechnungen
und 45 Bestätigungs-/Parametersensitivitätsrechnungen. Die acht initialen Labels
pro Arm sind in dessen 24 Simulationen enthalten; kein kostenloses Vortraining.
Seeds: 13101–13105. Identischer Suchraum: offene Kitaev-Ketten N=6..65,
t=1, Delta=0.5, mu=1.8; identischer Startpool pro Seed.

Zusätzlich: 420 Simulationen im dokumentierten Vorlauf und 84 für die unabhängige
Wiederholung von Seed 13101, somit **924 Benchmark-Simulationen insgesamt**,
ohne Entwicklungs-/Regressionstests. Der Vorlauf wurde wegen zeitstempelabhängiger
Trainingssortierung korrigiert, nicht wegen seines negativen Ergebnisses.
Seeds/Parameter/Kriterien wurden nicht angepasst; Läufe werden nicht gepoolt.

## 2. Best Candidate und physikalische Metriken

Bester AL-Kandidat: **N=29**, Seed 13104, Score **0.0737186631123**.
Das ist ein transparenter endlicher Qualitätsindikator, kein Majorana-Zertifikat.

| Exakte Metrik | Wert |
|---|---:|
| Niedrigste Paarenergie / Splitting (Einheit t) | 0.0002347072050 |
| Nächste Anregung (t) | 0.2243553125 |
| Isolation des niedrigsten Paars (t) | 0.2241206053 |
| Gewicht auf je zwei Randplätzen | 0.4120930548 |
| Mittlere lokale Polarisationsnorm | 0.9981954053 |
| Spektraler PHS-Fehler (t) | 1.78e-15 |
| Pfaffian-Bulkreferenz | topologisch |

Alle 15 Arm-/Seed-Gewinner stimmen bei unabhängiger exakter Wiederholung in
Spektrum und gespeicherten Metriken bis zur fixierten Toleranz 1e-10 überein.
Für N=29 ergeben mu=1.75/1.85 die Scores 0.1032038144/0.0429421819:
deutliche Parametersensitivität. Das sind zwei kontrollierte Perturbationen,
kein Disorderensemble oder Nachweis thermodynamischer Robustheit.

## 3–4. Active Learning gegen beide Baselines

Die Werte sind Mittel über fünf gepaarte Seeds; die Medianzeile mittelt die
Medianqualität der 24 ausgewerteten Kandidaten jedes Arms.

| Kennzahl | Active Learning | Random | Evolutionary |
|---|---:|---:|---:|
| Exakte Suchsimulationen pro Seed | 24 | 24 | 24 |
| Verschiedene Kandidaten pro Seed | 24 | 24 | 17 im Mittel |
| Beste Qualität, Mittel | 0.0736898114 | 0.0736892569 | 0.0736803892 |
| Medianqualität, Mittel | 0.0702857164 | 0.0705979702 | 0.0723293708 |
| Best-so-far-AUC, Mittel | 0.0727387129 | 0.0727019783 | 0.0727702703 |
| Gesamtzeit pro Seed, Mittel (s) | 17.026 | 0.536 | 1.027 |
| Zeit exakter Evaluatoraufrufe, Mittel (s) | 0.497 | 0.531 | 0.549 |
| Summe Basisdimension³, Mittel | 15,206,670 | 12,918,859 | 11,709,186 |

AL minus Random: mittlere Bestqualitätsdifferenz **+5.54e-7**,
gepaartes 95%-Bootstrapintervall **[-3.94e-5, +4.00e-5]**;
AL besser/schlechter/gleich in **2/1/2** Seeds.

AL minus Evolution: **+9.42e-6**, Intervall **[-5.60e-5, +9.93e-5]**;
AL besser/schlechter/gleich in **1/3/1** Seeds.

Evolution verwendet den bestehenden Phase-10-Generationsloop mit Tournament,
Elitismus und expliziter begrenzter Mutation. Erneute Elite-/Duplikatsbewertungen
verbrauchen gemäß dessen Vertrag Budget. Random und AL haben keine Duplikate.
Das exakte Aufrufbudget ist gleich; Rechenzeit ist nicht gleichgesetzt. Der
Dimension³-Proxy liegt bei AL 17.7% über Random und 29.9% über Evolution.
Auf diesen kleinen Matrizen dominiert der ML-/Orchestrierungsaufwand. Die Spalte
Evaluatorzeit misst den gesamten exakten Evaluator einschließlich Diagnostik
und Record-Erstellung, nicht isolierte LAPACK-Zeit. Laufzeiten sind deskriptiv.

## 5. Learning / Sample Efficiency

Die vorab fixierte Schwelle 0.02 erreichen alle Verfahren nach **[1,1,1,1,2]**
Simulationen, also schon im gemeinsamen Startpool. Sie liefert hier keinen
trennenden Sample-Efficiency-Nachweis. Auch Best-so-far-AUC und Medianqualität
belegen keinen stabilen Vorteil gegenüber beiden Baselines.

RMSE auf den drei reservierten Testgrößen N=68/70/72: im Mittel
**0.00080610 → 0.00054966**, verbessert in **2/5** Seeds, verschlechtert in 3/5.
Die Verbesserung des Mittelwerts ist kein konsistenter Suchvorteil. Diese drei
Geometrien werden über Seeds wiederholt; es sind nicht 15 unabhängige Testfamilien.
Pro AL-Seed: 12 Exploitations- und 4 Explorationsakquisitionen.

## 6. Stabilität und Reproduzierbarkeit

AL-Bestqualität: 0.0736493063 bis 0.0737186631, Stichproben-Standardabweichung
3.70e-5. Seed 13101 wurde mit frischem Zeitstempel vollständig wiederholt:
identische Auswahl- und wissenschaftliche Ergebnisse. Entwicklungsseed 13001
prüft zusätzlich Reproduzierbarkeit bei verändertem Datum. Unterbrechung nach
gesichertem exaktem Ergebnis reproduziert denselben Suchpfad; Abbruch ohne
gesichertes Ergebnis bleibt ein budgetierter Fehlversuch.

Artefaktaudit: fünf Seeds, 40 JSON-Artefakte; exakte Schema-/Recordvalidierung,
gleiche Startpools, korrekte Budgets, keine Testfamilien im Training, keine späteren
Labels in früheren Vorhersagen, keine OOD-Exploitation, keine AL-Duplikate.

## 7. Schwächen / Failure Modes

- Enger, nahezu gesättigter Ketten-Suchraum; keine neue Geometriefamilie oder
  allgemeine Graph-Discovery belegt. Die niedrige First-hit-Schwelle trennt nicht.
- OOD-Fehler werden sichtbar: MAE **0.04348** auf 3 OOD-Akquisitionen gegenüber
  **0.00704** auf 77 übrigen Akquisitionen. Das sind selektierte Daten, kein
  unabhängiger Kalibrationstest; die Phase-12-Coverage-Grenze bleibt bestehen.
- Kein Simulations-/Zeitvorteil. Begrenzte Feature-/Modellkapazität und die
  relative Skalierung von Wert, Unsicherheit und Neuheit sind mögliche Ursachen,
  aber dieser Versuch identifiziert keine eindeutige Reparatur.
- Rangfolgen ändern sich unter alternativen Splitting-Skalen 0.005/0.02
  (bester Kandidat in Seed 13104: N=33/23 statt N=29 bei 0.01);
  Parametersensitivität ist erheblich. Keine umfassende Robustheits- oder
  Finite-size-Validierung einer neuen physikalischen Familie.
- Checkpoints setzen einen schreibenden Prozess voraus. Solverabbrüche ohne
  gesichertes Ergebnis kosten Budget; korrupte veröffentlichte Checkpoints werden
  zurückgewiesen. Null numerische Ausfälle im eigentlichen Gate-Lauf.

## 8. Entscheidung und Nachweise

**Discovery Gate A nicht bestanden.** Die vorab fixierte positive untere
Konfidenzgrenze und mindestens vier positive Seeds werden gegen keine Baseline
erreicht. Ein minimaler, eindeutig verantwortlicher Reproduzierbarkeitsfehler
wurde behoben; weitere Gewichtungs-/Modelländerungen wären neue Experimente.
Block C endet hier. Keine Phase-14-Implementierung.

Protokoll: [phase_13_benchmark_protocol.md](phase_13_benchmark_protocol.md).
Maschinenlesbarer Report: [phase_13_discovery_gate_a.json](phase_13_discovery_gate_a.json).
Rohdaten/Checkpoints/Audit: `results/phase13-gate-a-final/`;
Wiederholung: `results/phase13-gate-a-repeat/`; Vorlauf: `results/phase13-gate-a/`.
Die `results/`-Artefakte sind lokal und gemäß bestehender Repository-Regel Git-ignoriert.

Quellbasis: `c628ed3e4ae3f4944ba2dfed09abd1d7e9fdfe82` plus Phase-13-Arbeitsbaum.
SHA-256 aller Python-Quellen:
`07f16b366fd38ad2201e70a8aa4322f28ddd4e004bdd625bf623d3c4bbce34bd`.
Quellarchiv: `results/phase13-source-snapshot.zip`, Archiv-SHA-256
`32608e04b47933240f615ae44af6558ed8c845ec259aae4162b4c47ffdaaaedf`.

Öffentliche API: neues `toposc_lab.active_learning`; additive ungelabelte
`ml.features.FeatureInput`-Eingabe. Bestehende Physik-/Basis-/Topologiekonventionen
unverändert; neuer endlicher Benchmarkscore mit explizitem Vertrag.
