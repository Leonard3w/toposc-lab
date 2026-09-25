# Phase 19: eingebettete Graphen

Die Graphdarstellung bleibt `Geometry` mit `GeometryEdge`: Site-IDs sind Indizes,
Koordinaten und Kanten sind getrennt, Randinformationen und Metadaten bleiben
erhalten. Es gibt keinen zweiten Hamiltonian, Score oder Experiment-Speicher.

`research.embedded.EmbeddedDomain` legt Koordinatenbox, Randstreifen,
Innenbereich und Integrationszelle explizit fest. Der neue versionierte Adapter
verwendet dieselbe BdG- und Disorder-Auswertung wie der bisherige Adapter.
Die allgemeine Modellklasse unterstützt Richtungs-Pairing bereits; übernommen
wird Delta_ij = Delta * (dx + i*dy)/r mit Delta_ji = -Delta_ij.
Es wird kein neuer Abstandsexponent für Hopping oder Pairing eingeführt.

Die vier Familien sind reguläres Quadratgitter, umverdrahtetes Quadratgitter,
amorphe planare Geometrie und räumlich eingebetteter Zufallsgraph mit begrenzten
Kantenlängen. Die letzten beiden teilen den bestehenden Hard-Core-Punktprozess;
ihre Kantenpools unterscheiden sich. Sie sind keine unabhängigen Tests aller
denkbaren amorphen Ensembles.

PowerShell-Aufruf für ein **neues, ausdrücklich geplantes** Experiment:

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:BLIS_NUM_THREADS='1'
$env:PYTHONDONTWRITEBYTECODE='1'
.venv/Scripts/python.exe -B -m toposc_lab.research.embedded_study prepare results/NEUE-kohorte.json --per-family 50
.venv/Scripts/python.exe -B -m toposc_lab.research.embedded_study create results/NEUER-lauf --cohort results/NEUE-kohorte.json
.venv/Scripts/python.exe -B -m toposc_lab.research.embedded_study run results/NEUER-lauf
```

`run --max-stages 20` pausiert nach höchstens 20 zusätzlichen Realisierungen.
Ein erneutes `run` setzt anhand des vorhandenen SQLite-Speichers fort und
berechnet gespeicherte Realisierungen nicht erneut. Quellcode, Laufzeitumgebung,
Protokoll, Kohorte und Konfiguration werden bei Wiederaufnahme überprüft.
Bei numerischen Validierungsfehlern wird der Lauf unterbrochen; zuerst ist der
gespeicherte Fehler zu untersuchen. Die Budgetgrenzen bleiben bestehen.

`report results/NEUER-lauf` exportiert gespeicherte Ergebnisse ohne neue Physik:
vollständige Realisierungen als JSONL, skalare Rohdiagnostiken als CSV,
Kandidaten- und Familienstatistiken, gepaarte Baseline-Differenzen,
deskriptive Spearman-Korrelationen und Bilder. Für die Beweiskette sind
`source.zip`, `manifest.json`, `protocol.md`, `study.json`, `cohort.json` und
die Datenbank gemeinsam aufzubewahren. Ein geänderter Quellcode darf nicht
durch Überschreiben des alten Manifests als ursprünglicher Lauf ausgegeben werden.

Die drei gemeinsamen Disorder-Seeds koppeln die iid-Vektoren nach Site-Index.
Bei unterschiedlichen Koordinaten ist das kein identisches räumliches Feld.
Familienverteilungen zeigen Geometrievariation; alle Realisierungen gemeinsam
sind wegen geteilter Seeds keine unabhängigen Stichproben für Signifikanztests.
Ein nicht separierter Bulk-Spektralgap bleibt ausdrücklich nicht verfügbar.
Der Interior-Gap bezeichnet den spektralen Localizer.

Siehe [Gap-Analyse](decisions/phase19_architecture_gap_analysis.md),
[eingefrorenes Protokoll](decisions/phase19_exploratory_protocol.md) und
[Forschungsbericht](decisions/phase19_exploratory_report_de.md).
