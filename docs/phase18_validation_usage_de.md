# Phase 18: unabhängige Fixed-Site-Validierung

Die neue Studie verwendet eine eingefrorene Kohorte und die bestehende
Research-Persistenz. Sie ist keine neue Suchkampagne. Der konkrete Einstieg
ist zunächst die CLI; der allgemeine Research-Suchstarter weist diese Studien
ausdrücklich zurück, damit keine unbeabsichtigte Suche beginnt.

- [Gesamtplan](roadmap/phase18_user_plan.md)
- [Bearbeitungsstand](roadmap/phase18_status.md)
- [Protokoll vor neuen Ergebnissen](decisions/phase18_validation_protocol.md)
- Kohorte: `examples/phase18_validation_cohort.json`
- Pilot: `examples/phase18_validation_pilot.json`

## Kleine Studie starten oder fortsetzen

Alle Befehle aus dem Repository-Verzeichnis. Die Threadvariablen vor Python
setzen, damit der gemessene Aufwand und die Herkunft reproduzierbar bleiben:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:BLIS_NUM_THREADS = '1'

.venv/Scripts/python.exe -B -m toposc_lab.research.validation create results/phase18-pilot --config examples/phase18_validation_pilot.json --cohort examples/phase18_validation_cohort.json
.venv/Scripts/python.exe -B -m toposc_lab.research.validation run results/phase18-pilot
```

`create` verlangt einen neuen/leeren Ordner. `run` nimmt dieselbe Studie unter
strengem Quell-, Konfigurations- und Laufzeitabgleich wieder auf. Auf einem
abgeschlossenen Lauf werden nur Exporte rekonstruiert, keine Exakt-Stufen ergänzt.
`run --max-stages 1` begrenzt einen Aufruf, ohne das Gesamtbudget zu ändern.

## Pause, Stop, Checkpoints und Auswertung

In einem zweiten Terminal:

```powershell
.venv/Scripts/python.exe -B -m toposc_lab.research.validation pause results/phase18-pilot
.venv/Scripts/python.exe -B -m toposc_lab.research.validation checkpoint results/phase18-pilot
.venv/Scripts/python.exe -B -m toposc_lab.research.validation stop results/phase18-pilot
```

Anfragen werden in der vorhandenen Control-Tabelle gespeichert. Ein laufender
Worker verarbeitet sie zwischen vollständigen Realisierungen. Bei einem
ruhenden Worker erfolgt die Verarbeitung beim nächsten `run`. Pause ist
wiederaufnehmbar; Stop beendet die Studie. Ein bereits begonnenes numerisches
Paket wird nicht hart abgebrochen. Ein harter Prozessabbruch bleibt im
Versuchsjournal verbucht; Wiederholung kostet einen weiteren Versuch.

```powershell
.venv/Scripts/python.exe -B -m toposc_lab.research.validation report results/phase18-pilot
```

`report` wertet eine konsistente lesende Datenbank-Momentaufnahme aus; keine
Physikrechnung. Es schreibt abgeleitete Dateien in den neuen Studienordner:

- `reports/realizations.jsonl`: Einzelresultate inklusive Felder, Spektren,
  Index/Lücke/Gültigkeit jeder Probe, Chern-Marker und Randzustandsprofile.
- `reports/summary.json`, `reports/curves.csv`: Statistik und alle Nenner.
- `plots/`: Geometrien, Robustheitskurven, Localizer-Karten, räumliche Schnitte,
  Chern-Marker, Randprofile und Qualität/Randgewicht-Zielkonflikt.
- `final_report.md`: automatisch erzeugter deutscher Forschungsbericht.
- `confirmation_proposal.json`: erst nach vollständigem Pilot; neues Raster,
  50 reservierte Seeds und noch kein autorisiertes Budget.

## Bestätigung nach ausdrücklicher Budgetentscheidung

Die startbereite Konfiguration wird als
`examples/phase18_validation_confirmation.json` dokumentiert. Sie enthält
absichtlich `null` für Versuchslimit und Zeitlimit. Erst explizite Werte
aktivieren die Erstellung:

```powershell
# Werte erst nach der vereinbarten Budgetentscheidung einsetzen:
.venv/Scripts/python.exe -B -m toposc_lab.research.validation create results/phase18-confirmation --config examples/phase18_validation_confirmation.json --cohort examples/phase18_validation_cohort.json --exact-budget <VERSUCHE> --wall-seconds <SEKUNDEN>
.venv/Scripts/python.exe -B -m toposc_lab.research.validation run results/phase18-confirmation
```

Die Bestätigung wurde nach ausdrücklichem Nutzerauftrag mit `--exact-budget 2560`
und `--wall-seconds 7200` vollständig ausgeführt: 2510 Realisierungen, keine Ausfälle.
Der [Bestätigungsbericht](decisions/phase18_confirmation_report_de.md) enthält
die geprüften Ergebnisse. Keine neue Studie automatisch starten.

Die ergänzende Statistik lässt sich ohne neue Physikrechnungen reproduzieren:

```powershell
.venv/Scripts/python.exe -B scripts/phase18_confirmation_analysis.py results/phase18-confirmation --output results/phase18-confirmation-analysis
```

Der automatische alte Pipeline-Bericht enthält noch Pilot-Textbausteine;
für die Interpretation der Bestätigung gilt der oben verlinkte Forschungsbericht.
Quellstand nach dem Pilot einfrieren oder spätere Änderungen als neue Version
prüfen. Vorhandene Studien niemals durch manuelles Ersetzen von Fingerabdrücken
„kompatibel“ machen. Pilot und Bestätigung werden nicht zusammengepoolt.

## Datenhaltung und nachgelagerte Forschung

`results/` ist Git-ignoriert. Den gesamten neuen Studienordner separat sichern,
insbesondere SQLite einschließlich eines noch vorhandenen WAL, source.zip,
Konfiguration und Kohorte. Eine normale Quellcode-Sicherung enthält diese Daten
nicht. Historische Experimente werden nur gelesen.

Hopping-Unordnung und Kantenausfälle existieren im allgemeinen Robustheitspaket,
werden aber vom eingefrorenen Phase-17-Research-Adapter nicht unterstützt.
Geplante separate Folgestudien: gleicher Kohorten-/Seed-Abgleich, explizite
Verteilung und korrelierte Teilchen-Loch-Einbettung für Hopping beziehungsweise
identische vorab definierte Kanten-Zufallsvariablen bei unterschiedlichen
Kantenmengen; Konnektivitätsverlust als eigener Status. Diese Protokolle wurden
hier nicht implementiert oder gestartet. Größenfamilien, Kopplungsabfall mit
Entfernung, Nachbarparameter und Chiralitäts-/Transportdiagnostik bleiben vertagt.
