# Phase 15: Autonome Discovery Engine

Die Engine automatisiert den numerischen Ablauf. Exakte Rechnungen bleiben die
wissenschaftliche Grundlage. Sie zertifiziert weder Majorana-Zustände noch eine
thermodynamische Phase selbstständig.

```python
from toposc_lab.discovery import DiscoveryConfig, DiscoveryEngine

config = DiscoveryConfig(seed=15201, cycles=3)
discovery = DiscoveryEngine("results/my-discovery", config)
single_cycle = discovery.run(iterations=1)

# Neuer Prozess oder neue Python-Sitzung, dieselbe Konfiguration:
summary = DiscoveryEngine("results/my-discovery", config).run()
```

`iterations` bezeichnet zusätzliche Zyklen. `cycles` ist die vorab festgelegte
Gesamtgrenze. Ohne `iterations` werden die verbleibenden Zyklen ausgeführt. Ein
bereits abgeschlossener Lauf startet keine neuen Rechnungen. Konfiguration,
Quellcode und numerische Laufzeit müssen beim Wiederaufnehmen übereinstimmen.

CLI, nachdem die vier Thread-Variablen vor dem Python-Start gesetzt wurden:

```powershell
$env:PYTHONPATH = 'src'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:BLIS_NUM_THREADS = '1'
.venv/Scripts/python.exe -B -m toposc_lab.discovery results/my-discovery --iterations 1
.venv/Scripts/python.exe -B -m toposc_lab.discovery results/my-discovery
```

Eine eigene JSON-Konfiguration kann über `--config config.json` übergeben werden.
Beim CLI-Resume ohne diese Option wird die gespeicherte Konfiguration geladen.
Die Standardwerte stehen in `DiscoveryConfig`; unbekannte oder inkompatible
wissenschaftliche Einstellungen werden abgewiesen.

Der erste Zyklus verwendet einen gemeinsamen Random-Warmstart. Danach ist Patch
nur im validierten 36-Site/60-Kanten-Stratum der Standard. `generator="random"`
und `generator="evolution"` bleiben Vergleichsoptionen; `coverage` ist optional.
Andere Suchräume benötigen einen eigenen exakten Adapter und kontrollierte
Benchmarks. Die Policy dort bevorzugt Evolution als etablierte Baseline; die
aktuelle Engine weist nicht implementierte Räume ausdrücklich zurück.

`surrogate=True` aktiviert die optionale Vorhersage/Akquisition. Die Trainingsbasis
wird gemäß `retrain_every` nur aus abgeschlossenen exakten Zyklen erweitert. Es
gibt keine automatisch behauptete ML-Überlegenheit oder Kalibrierungsgarantie.
OOD-Kandidaten werden vor der Rechnung markiert und können nur durch exakte
Validierung wissenschaftliche Evidenz erhalten. Ohne Warmstart-Referenz lautet
der OOD-Status unbekannt und wird vorsichtig als Warnung gezählt.

Jeder ausgewählte Kandidat erhält eine exakte Ausgangsrechnung, eine unabhängige
Bestätigung und ein explizit gesätes Onsite-Unordnungsensemble. Das eingefrorene
Erfolgskriterium bleibt Qualität >=0,20. Das Ensemble ist klein und prüft nur
skalare Onsite-Unordnung. Für diesen festen Suchraum ist keine sinnvolle
kandidatenspezifische Größenskalierung definiert; das Ergebnis lautet deshalb
`unavailable`, niemals `passed`.

Die atomare `dataset.json` enthält ausschließlich vollständig exakt validierte
Kandidaten. Rohe exakte Zwischenresultate und fehlgeschlagene Rechnungen bleiben
im Journal erhalten. Optional kann `seed_dataset=load_dataset(...)` als externe
Duplikat-Sperrliste übergeben werden. Dieses Archiv wird nicht automatisch zum
Training oder als neue Discovery-Evidenz importiert. Beim API-Resume dasselbe
Archiv übergeben. Die Duplikatprüfung umfasst frühere gültige Vorschlagspools,
das externe Archiv und den aktuellen Pool; die Grenze wird nie automatisch
gelockert.

Wichtige Dateien im Laufverzeichnis:

- `manifest.json`, `source.zip`, `exclusions.json`: eingefrorene Konfiguration,
  Basis-Commit, Dirty-Status, Quellcode, Laufzeit und externe Sperrliste.
- `cycle-*/plan.json`: Vorschläge, Ablehnungsgründe, OOD, Vorhersagen und Auswahl
  vor der exakten Auswertung.
- `cycle-*/base-*.json`, `confirmation-*.json`, `disorder-*.json`: exakte Ergebnisse
  einschließlich Spektrum, Eigenzuständen im Basisergebnis und Diagnostik.
- `attempts/`: dauerhaft gezählte gestartete Rechnungen; auch ein unterbrochener
  Versuch verbraucht Budget. `retry_reserve` begrenzt Wiederholungen.
- `checkpoint.json`, `cycle-*/commit.json`: atomare Wiederaufnahmepunkte und
  Prüfsummen. Eine Betriebssystem-Sperre verhindert parallele Schreibzugriffe
  und wird beim Prozessabbruch freigegeben.
- `leaderboard.json`: exakte Qualität, Randgewicht, mittlere Unordnungsqualität,
  Pareto-Status und alternative Gewichtung; Neuheit kompensiert keine Physik.
- `reports/`: reproduzierbare Kandidatenberichte mit expliziten Koordinaten und
  Kanten sowie vollständigem Dataset-Record; `discovery.png` zeigt Verlauf und
  saubere versus gestörte Qualität ohne erneute Physikrechnung.

Engine-JSONs besitzen eine Hülle aus `sha256` und `payload` und werden über
`toposc_lab.discovery.storage.read_json` geprüft. `dataset.json` bleibt im
bestehenden Dataset-Format und wird mit `load_dataset` gelesen.

Nach einem Prozessabbruch den identischen Lauf erneut öffnen. Gespeicherte
Ergebnisse werden wiederverwendet; eine vor dem Speichern unterbrochene Rechnung
wird mit demselben Seed wiederholt und zusätzlich gezählt. Veränderte oder
beschädigte Artefakte werden nicht stillschweigend repariert. Fehlende
Dataset-Einträge oder ein hinter dem Zyklus-Commit zurückliegender Checkpoint
werden aus dem gültigen Journal rekonstruiert. Laufzeiten nach Abbrüchen sind
Untergrenzen, da nicht gespeicherte Zeit nicht rekonstruiert werden kann.

Gate-Reproduktion (neues, noch nicht existierendes Verzeichnis erforderlich):

```powershell
.venv/Scripts/python.exe -B scripts/phase_15_gate.py results/phase15-gate-repeat
```

Der Gate-Test führt sieben kleine Kampagnen einschließlich echter
Prozessbeendigung aus. Er startet keine Phase 16 und keine große Suchkampagne.
