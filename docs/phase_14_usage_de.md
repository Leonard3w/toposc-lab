# Phase 14: Generative Geometrie

Die neue Schnittstelle erzeugt verbundene, kreuzungsfreie Graphen mit 36 Orten
und 60 Kanten auf einem 6x6-Gitter. Zufall, Evolution, räumliche Patch-Vorschläge,
strukturelle Coverage und optionale Active-Learning-Auswahl nutzen dieselben
Ressourcen, Gültigkeitsregeln und exakten Physikprüfungen.

Evolution verwendet die vorhandene Phase-10-Fitness, Turnierauswahl und
Kantenmutation. Patch und Coverage sind einfache, nicht lernende Verfahren.
Active Learning bleibt optional. Eine bessere Suchleistung wird nicht vorausgesetzt.
Die vorhandene REINFORCE-Geometriedemo ist kein physikalisch validierter Generator.

Im eingefrorenen Vergleich über 20 Seeds erfüllt **Patch** das Kriterium für einen
reproduzierbaren Vorteil beim besten kontinuierlichen Qualitätswert: 16/20 Siege
gegen Random, 18/20 gegen Evolution, jeweils positive korrigierte Konfidenzintervalle.
Für genau diesen Suchraum und dieses Qualitätsziel ist Patch die bevorzugte einfache
Methode. Die Erfolgsschwelle 0.20 wurde von keinem Verfahren erreicht. Evolution
bleibt die etablierte Baseline außerhalb dieses begrenzten Ergebnisses.

Die reine Geometrieerzeugung berechnet noch keine Physik:

```python
from toposc_lab.generative.generators import GeometryGenerator

generator = GeometryGenerator("patch")
geometries = generator.propose((), seed=14001, count=4)
print(generator.audit.invalid, generator.audit.duplicate)
```

Für Evolution und AL wird bereits bezahlte exakte `Evidence` als Historie übergeben.
Der Benchmark übernimmt Erzeugung, OOD-Warnungen, exakte Auswertung und Archivierung.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='src'
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:BLIS_NUM_THREADS='1'

# Kleiner Entwicklungslauf: zwei Seeds, insgesamt 274 exakte Auswertungen
.venv\Scripts\python.exe -B -m toposc_lab.generative.benchmark --mode development --output results/phase14-local-development

# Eingefrorener Gate-Vergleich: 20 Seeds, insgesamt 2704 exakte Auswertungen
.venv\Scripts\python.exe -B -m toposc_lab.generative.benchmark --mode gate --output results/phase14-local-gate

# Unabhängige Reproduktion des ersten Gate-Seeds: 135 exakte Auswertungen
.venv\Scripts\python.exe -B -m toposc_lab.generative.benchmark --mode repeat --output results/phase14-local-repeat

.venv\Scripts\python.exe -B scripts/phase_14_audit.py --directory results/phase14-local-gate --repeat results/phase14-local-repeat --output results/phase14-local-audit.json
```

Jeder Outputpfad muss neu sein. Die Befehle benötigen keine Netzverbindung.
Der Runner überschreibt keine Kampagne und bietet kein Resume. Bei einem Fehler
bleiben bereits bezahlte Versuche und gespeicherte Records im Archiv; es wird kein
vollständiger Gate-Erfolg ausgegeben. Unterbrechungen erfordern eine Prüfung der
Versuchsliste, bevor ein neuer, separat verbuchter Lauf gestartet wird.

`exact-attempts.jsonl` protokolliert jeden gestarteten und abgeschlossenen Versuch.
Pro Kandidat bleiben exakte Dataset-Records erhalten. `selected-ood-*.json` hält
OOD-Warnungen vor der Simulation fest; `proposal-audit.json` enthält auch verworfene
Vorschläge und die getrennt markierten AL-Vorhersagen. `summary.json`, `report.json`,
`manifest.json`, `inventory.json` und `source.zip` ermöglichen Auswertung und Audit.

Nähe wird über die Kanten-Jaccard-Distanz unter acht Symmetrien des quadratischen
Gitters gemessen. Distanz <=0.06 wird verworfen. Die Metrik berücksichtigt die
Einbettung und beseitigt damit Relabeling und starre Bildänderungen; sie ist kein
allgemeiner Beweis graphentheoretischer oder wissenschaftlicher Neuheit.

Die Physik bleibt das vorhandene chirale p-Wellen-Modell. Die Qualität ist der
kleinste lokale Spektrallokalisator-Gap bei drei Kappa-Werten, sofern alle Indizes
übereinstimmen und ungleich null sind. Widersprüche werden als unaufgelöst
gespeichert. Der Erfolgsschwellwert bleibt 0.20. Eigenwerte, Randgewichte und
Majorana-Diagnostik bleiben separat verfügbar. Daraus folgt weder ein
thermodynamischer Phasenbeweis noch eine Majorana-Behauptung.

Siehe [eingefrorenes Protokoll](decisions/phase_14_protocol.md),
[Datenaudit](decisions/phase_14_feasibility.json) und
[Phase-14 Gate Report](decisions/phase_14_generator_gate.md).
Phase 15 startet nicht automatisch.
