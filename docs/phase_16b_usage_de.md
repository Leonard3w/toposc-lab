# Phase 16B: unabhängige Prüfung der fünf Hypothesen

Phase 16B verwendet die eingefrorenen Fragen aus 16A. Der neue Testplan wählt
Geometrien und Eingriffe ohne Kenntnis ihrer physikalischen Ergebnisse aus.
Die Erfolgsschwelle bleibt Q >= 0,20. Ein einzelner endlicher Graph begründet
weiterhin weder eine thermodynamische Phase noch einen separierten Majorana-Modus.

## Ergebnisse und Umfang

- [Prüfprotokoll vor den Ergebnissen](decisions/phase_16b_protocol.md)
- [Wissenschaftlicher Bericht](decisions/phase_16b_validation.md)
- [Maschinenlesbarer Bericht](decisions/phase_16b_validation.json)
- [Übergabe und nächster Schritt](roadmap/NEXT_CODEX_RUN.md)

Die 16 Eltern stammen aus neuen Seeds 16201–16216, abwechselnd Random und Patch.
Alle bisherigen 16A-Strukturen und die zum Stichtag vorhandenen Vorschläge der
Discovery-Kampagnen bilden einen eingefrorenen Ausschlusssatz. Innerhalb einer
Elternfamilie sind Eingriffe absichtlich verwandt; verschiedene Familien und
alte Strukturen müssen auch nach Gittersymmetrien ausreichend getrennt sein.

Nicht verfügbare streng passende Vergleiche werden nicht durch locker passende
ersetzt. Ergebnisse bei mu=1,9 und 2,1 sind Parametersensitivität innerhalb des
gleichen Modells. Ein validierter Vergleich mit einer anderen Modellfamilie
steht aus.

## Reproduktion

Aus dem Repository-Verzeichnis in PowerShell:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:BLIS_NUM_THREADS = '1'

# Vorhandene gespeicherte Ergebnisse prüfen, keine neuen Eigenwertberechnungen:
.venv/Scripts/python.exe -B scripts/phase_16b_validation.py --audit-only

# Bericht und Abbildung aus gespeicherten Ergebnissen neu erstellen:
.venv/Scripts/python.exe -B scripts/phase_16b_report.py
```

Der vollständige Treiber wird so ausgeführt:

```powershell
.venv/Scripts/python.exe -B -u scripts/phase_16b_validation.py
```

Bei einem abgeschlossenen Lauf verwendet er alle exakten Ergebnisse erneut und
benötigt keine weiteren numerischen Aufrufe. Quelle, Laufzeit, Protokoll und
Plan müssen exakt dem Manifest entsprechen. Eine Abweichung wird abgelehnt;
Fingerabdrücke nicht manuell korrigieren. Die Quellen im ZIP sind maßgeblich,
weil der aufgezeichnete Git-Commit zusätzlich lokale Änderungen enthielt.
Der Plan kann separat mit `--plan-only` erstellt werden. Das plant einen neuen
Studienordner, wenn dort noch kein Plan vorliegt, und ist kein reiner Audit.

## Aufbewahrung

`results/phase16b/` enthält den Plan, den Ausschlusssatz samt Herkunftsprüfsummen,
alle Versuche, exakte Datensätze und Eigenvektoren, Auswertung, Manifest,
Quellenarchiv und Inventar. Der Ordner ist von Git ausgeschlossen und muss
zusätzlich gesichert werden. Logs und vollständige Tests liegen unter
`results/phase16b-*.log`. Bestehende 16A-Daten werden nicht überschrieben.

Der neue Ablauf ändert keine Discovery-Kampagne und keine Funktion der Live-App.
Block G endet nach der dokumentierten Gate-16-Entscheidung.
