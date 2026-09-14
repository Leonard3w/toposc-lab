# Phase 16A: Musteranalyse und kontrollierte Ablationen

Block F ist am Internal Mechanism Checkpoint abgeschlossen. Der
[Muster-/Ablationsbericht](decisions/phase_16a_pattern_ablation.md) dokumentiert
auch Nullbefunde. Kein Motiv besteht das vorab festgelegte Ablationskriterium;
die Erfolgsschwelle bleibt 0,20. Phase 16B wurde nicht begonnen.

## Vorhandene Ergebnisse pruefen

Die folgenden Befehle werden im Repository-Verzeichnis in PowerShell ausgefuehrt:

```powershell
$env:PYTHONPATH = 'src'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:BLIS_NUM_THREADS = '1'
.venv/Scripts/python.exe -B scripts/phase_16a_audit.py
.venv/Scripts/python.exe -B -m toposc_lab.patterns.campaign --output results/phase16a --analysis-only
.venv/Scripts/python.exe -B scripts/phase_16a_report.py
```

Der Audit prueft gespeicherte Eigenzustaende und Diagnostik ohne neue Eigenwert-
rechnungen. `--analysis-only` reproduziert die eingefrorenen Statistiken und
bricht bei Abweichungen ab. Der Berichtstreiber regeneriert JSON und PNG aus
diesen Ergebnissen. Er fuehrt keine Simulation aus.

## Daten und Herkunft

`results/phase16a/` enthaelt:

- `manifest.json`, `source.zip`, `source-archive.json`: Konfiguration, Runtime,
  Quellstand und Pruefsummen; Protokoll und relevante Tests sind im Archiv.
- `inputs/`: unveraenderte vollstaendige Datensaetze der vier Phase-15-Arme.
- `fresh-plan.json`: alle 176 Rohvorschlaege mit Annahme/Ablehnung vor Auswertung.
- `fresh-dataset.json`, `features.json`, `distances.json`: 64 neue exakte Labels,
  96 eindeutige Analysegeometrien, Motifinstanzen und vollstaendige Distanzmatrix.
- `intervention-plan.json`: festgelegte Eltern, Edits, Kontrollpaare,
  Perturbationsseeds, nicht verfuegbare Vergleiche und Vereinfachungen.
- `attempts/`, `exact/`: 464 gezaehlte Rechnungen; jeweils vollstaendiger
  Datensatz, gespeichertes Eigensystem, Localizerdaten und Majorana-Diagnostik.
- `experiments.json`, `simplifications.json`, `analysis.json`: alle 79
  Ablationspaare, acht Ein-Kanten-Vereinfachungen, Statistiken und Unsicherheiten.
- `reproduction.json`, `accounting.json`, `inventory.json`: Wiederholungspruefung,
  Budgetzaehlung und Inventar aller 946 unveraenderlichen Dateien.

`results/phase16a-audit.json` liegt ausserhalb des eingefrorenen Verzeichnisses.
Die vollstaendigen Logs liegen als `results/phase16a-*.log` daneben.
`results/` ist git-ignoriert. Beim Uebertragen des Projekts muessen die Rohdaten
und Archive deshalb separat kopiert/gesichert werden; die Markdown-/JSON-
Berichte allein ersetzen die Rohdaten nicht.

Die zugewiesenen 700 Versuche gelten fuer diesen Block insgesamt. Verbraucht
sind 464; eine neue Ausgabedatei oder ein neues Verzeichnis hebt diese Grenze
nicht auf. Keine weitere Kampagne ist Teil des abgeschlossenen Blocks.

## Wiederaufnahme und Tests

Eine Wiederaufnahme des vorhandenen Verzeichnisses verwendet gespeicherte
Rechnungen. Quellstand, Protokoll, Runtime und Eingaben muessen exakt passen:

```powershell
.venv/Scripts/python.exe -B -m toposc_lab.patterns.campaign --output results/phase16a
```

Dieser vollstaendige Resume wurde geprueft: alle wissenschaftlichen Ergebnisse
und das Inventar bleiben gleich, der Zaehler bleibt bei 464. Fehlende committed
Ergebnisse, beschaedigte Dateien oder Konfigurationsabweichungen fuehren zum
Abbruch. Unterbrochene ungespeicherte Rechnungen werden beim erneuten Versuch
zusaetzlich gezaehlt. Der Prozess haelt eine Betriebssystem-Schreibsperre.

Die Block-Abschlusspruefung war:

```powershell
.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider --basetemp=results/pytest-phase16a-final
.venv/Scripts/python.exe -m ruff check src/toposc_lab/patterns tests/test_pattern*.py scripts/phase_16a_report.py scripts/phase_16a_audit.py
.venv/Scripts/python.exe -m mypy --python-version 3.14 --follow-imports silent --ignore-missing-imports src/toposc_lab/patterns
```

Ergebnis: 2842 Tests bestanden; Ruff und die isolierte Typpruefung bestanden.
Die explizite Python-3.14-Einstellung folgt der vorhandenen Umgebung; die
SciPy-Stubs fehlen und werden in dieser isolierten Pruefung nicht geprueft.

Fuer Block G sind neue unabhaengige Geometrien erforderlich. Seeds 16101-16108,
alle ausgewerteten Eltern und Edits sowie ihre Symmetrie-/Nahduplikatfamilien
gelten als bereits gesehen. Phase16B erfordert eine ausdrueckliche neue Anweisung.
