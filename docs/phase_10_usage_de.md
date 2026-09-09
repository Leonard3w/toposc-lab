# Phase 10 benutzen: evolutionäre Geometriesuche

## Was jetzt vorhanden ist

Phase 10 ergänzt eine kontrollierbare evolutionäre Suche über Geometrien. Sie
trainiert kein neuronales Netz: Kandidaten werden durch deine Simulationspipeline
bewertet. Ausgewählte Eltern liefern neue Kandidaten; die Suche wiederholt das
über eine festgelegte Zahl von Generationen.

```text
Startpopulation → Simulation/Fitness → Auswahl + Eliten → neue Geometrien
                         ↑                                     │
                         └──────── Gültigkeitsprüfung ──────────┘
```

Du legst Suchraum, physikalisches Modell, Zielfunktion, Gültigkeitsregeln, Seeds
und Fortpflanzungsregeln ausdrücklich fest. Das System erfindet diese
wissenschaftlichen Entscheidungen nicht. Allgemeine Suchen verwenden die
Python-Schnittstelle. Für den unten beschriebenen eingefrorenen Forschungsversuch
gibt es zusätzlich `toposc phase-10-research` mit Live-Fortschritt und Resume.

| Baustein | Kann er | Wichtige Grenze |
| --- | --- | --- |
| Genome | Geometrien unveränderlich als Suchkandidaten darstellen und zurückwandeln | Keine Physikparameter oder Fitness im Genom |
| Mutationen | Kanten/Knoten hinzufügen oder entfernen, Knoten bewegen, Kanten umverdrahten | Du bestimmst Operation und Parameter |
| Validierung | U. a. Zusammenhang, Größen-, Grad-, Abstands- und Raumgrenzen prüfen | Unzulässige Kandidaten werden nicht heimlich repariert |
| Fitness | Skalare Engineering-Scores oder getrennte wissenschaftliche Ziele | Hoher Score allein beweist keine Topologie |
| Auswahl und Elitismus | Turnierauswahl, vollständige beste Gleichstandsgruppen/Pareto-Fronten erhalten | Eine komplette Elite kann alle Plätze belegen |
| Crossover | Ganze orientierte Kanten kompatibler Eltern kombinieren | Nur wissenschaftlich definierter gemeinsamer fester Aufbau |
| Generationen | Eltern, Operationen, Seeds und Bewertungen nachvollziehbar führen | Auch Eliten werden erneut bewertet |
| Diversität | Explizite Familienbelegung prüfen | Keine erfundene Familienklassifikation; Verletzung stoppt |
| Novelty | k-Nachbar-Abstände mit deiner Abstandsfunktion auswerten | Separater Bericht, nicht automatisch Fitness oder Selektionsdruck |
| Checkpoint/Resume | Abgeschlossene Generationen speichern und exakt zum gespeicherten Ziel fortsetzen | Passende externe Funktionen und Versionen erforderlich |
| Benchmark | Evolution mit Zufallssuche unter gleichem Bewertungsbudget vergleichen | Kein automatischer wissenschaftlicher Überlegenheitsnachweis |

Die Einbettungsdimension bleibt während einer Suche erhalten. Phase 10.8 hat
Dimensionswechsel bewusst untersucht und nicht blind als Mutation freigegeben.
Abstrakte Graphen und andere Einbettungsdimensionen bleiben auf API-Ebene möglich;
das bedeutet nicht, dass jedes physikalische Modell darauf anwendbar ist.

## Sofort ausprobieren: sichere kleine Demo

Im Projektverzeichnis in PowerShell:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='src'
.venv\Scripts\python.exe examples/phase_10_search_demo.py
```

Die Demo nutzt sechs Knoten, einen festen Ring und zwei zusätzliche Kanten.
Mutation tauscht eine dieser Zusatzkanten aus. Ein kleiner reeller
Hopping-Hamiltonian wird numerisch diagonalisiert; die normierte endliche
Spektrallücke dient als Engineering-Score. Das ist ausdrücklich KEIN
Supraleiter-/BdG-Modell und KEIN wissenschaftlicher Phase-9.8-Lauf.

Fünf Seeds vergleichen jeweils Evolution und Zufallssuche. Mit sechs Mitgliedern
und vier Übergängen werden pro Methode `6 × (4 + 1) = 30` Bewertungen versucht:
insgesamt 300. Die Startpopulation gehört zum Budget. Die Demo schreibt keine
Ergebnisdateien und erzeugt keine gelöschten Phase-9.8-Ordner neu.

Die Ausgabe bedeutet:

- `evolution_best` / `random_best`: bester bis zum Budgetende beobachteter Score,
  nicht nur der beste Kandidat der letzten Generation.
- `delta`: richtungsbereinigter Unterschied. Positiv begünstigt Evolution,
  negativ die Zufallssuche; null ist Gleichstand.
- `first_hit`: Nummer des ersten Bewertungsversuchs mit Demo-Score ≥ 0.65.
- `None`: kein Treffer bzw. kein verfügbarer skalarer Score, nicht der Wert null.

Die Demo ist klein genug, um die Bedienung zu prüfen. Fünf Versuche und eine
Spielzeug-Zielfunktion reichen nicht für Aussagen über physikalische Vorteile.
Insbesondere ist die endliche Spektrallücke keine automatisch topologische Lücke.

## Eine eigene Suche aufsetzen

Für unseren ersten physikalischen Vergleich ist jetzt das
[Forschungsprotokoll TOPOSC-P10-EVO-RS-001](decisions/pre_phase_10_research_protocol_v1.md)
ausgearbeitet. Es legt 32 gepaarte Versuche fest: jeweils Evolution und
Zufallssuche mit denselben acht Startgeometrien, 64 Knoten, 112 Kanten und
32 Bewertungsversuchen pro Arm. Einschließlich Referenzen sind das 2.083
Bewertungsversuche im Hauptlauf. Ziel ist ein Vergleich der Suchtrefferrate;
Disorder-Robustheit und größere Systeme folgen in separaten Versuchen.

Das Protokoll ist unter `71e159f9eeb67de17b551cb820c4f3599830ca3b` separat
eingefroren. Der neue Befehl `toposc phase-10-research` implementiert Vorlauf,
Live-Fortschritt und Wiederaufnahme. Die Umsetzung muss ebenfalls committed
sein, bevor du einen Lauf startest. Der Runner prüft Code, Protokoll und Umgebung.
Vorlauf und Hauptlauf startest du selbst; sie schreiben in ein neues
Phase-10-Ergebnisverzeichnis.

### Nach dem abgeschlossenen Erstversuch

**Aktuell: Der Rand-/Innenhauptlauf ist mit 82/82 Bewertungen abgeschlossen.**
Alle Kontrollen sind gültig. Die
[geprüfte Auswertung](decisions/phase_10_edge_location_v1_analysis.md) erklärt,
warum zentrale Chern-Nähe, lokale Markeränderung und das bisherige Screening
unterschiedliche Bewertungen ergeben. Der
[Entwurf zum Messvertrag](decisions/phase_10_measurement_contract_draft_v1.md)
beschreibt den nächsten methodischen Schritt. Die bestehenden Läufe sind fertig;
ein neuer Lauf benötigt ein eigenes Validierungsprotokoll. Die folgenden
Start-/Resume-Befehle dokumentieren die bisherigen Durchführungen.

**Vorheriger Schritt: Der Größen-/Methodenhauptlauf ist mit 150/150 Bewertungen abgeschlossen.**
Die [räumliche Auswertung mit Grafik](analysis/phase_10_size_methods_spatial_v1/report_de.md)
zeigt, wo sich die Marker nach Positions- und Kantenänderungen vom Quadrat
unterscheiden. Alle sechs Kontrollbewertungen sind gültig. Für diesen fertigen
Lauf ist kein `--resume` und kein erneuter Start nötig; die folgenden Befehle
dokumentieren die Durchführung.

**Abgeschlossener Rand-/Innenvergleich.** Der
[Rand-/Innenvergleich](decisions/pre_phase_10_edge_location_protocol_v1.md)
verglich denselben graderhaltenden Zwei-Kanten-Eingriff an verschiedenen Orten
eines 12×12-Quadrats: 6 technische Vorlauf- und 82 Hauptlaufbewertungen.
Das Protokoll ist unter `acf122f9f8cc4e8848f3508766c127b435083f6e`
eingefroren, die verwendete Implementierung unter
`62fab4249d22823cf546c0553aca766a09297619` committed. Der bisherige einzelne
Kantentausch kann im unveränderten Quadrat nicht ausschließlich innen beginnen;
deshalb wurde der neue Zwei-Kanten-Eingriff ausdrücklich separat untersucht.

Der technische Vorlauf mit sechs Bewertungen wurde folgendermaßen gestartet:

```powershell
.\.venv\Scripts\python.exe -B -c "from toposc_lab.cli import main; main()" phase-10-edge-location --preflight --output results\phase_10_edge_location_v1
```

Nach bestandenem Vorlauf wurde der Hauptlauf mit 82 Bewertungen gestartet:

```powershell
.\.venv\Scripts\python.exe -B -c "from toposc_lab.cli import main; main()" phase-10-edge-location --full --output results\phase_10_edge_location_v1
```

Nur nach einem Abbruch bei unveränderter Umgebung und Code-Revision; für den
abgeschlossenen Lauf nicht erforderlich:

```powershell
.\.venv\Scripts\python.exe -B -c "from toposc_lab.cli import main; main()" phase-10-edge-location --resume --output results\phase_10_edge_location_v1
```

**Vorheriger Schritt, Quadratkalibration:** Der Lauf
`results/phase_10_calibration_v1/full` ist mit 386/386 Bewertungen abgeschlossen.
Von 384 veränderten Fällen bestehen 99 alle bisherigen Gates; zwei gültige
Nullkontrollen kommen hinzu. Die [geprüfte Auswertung](decisions/phase_10_calibration_v1_analysis.md)
erklärt Chern-Marker, Bulk-Masken und die Grenzen der Interpretation.
Der [damalige Größen-/Methodenplan](decisions/pre_phase_10_size_methods_protocol_v1.md)
legte den inzwischen abgeschlossenen Vergleich von 8×8, 10×10 und 12×12 fest.
Für den abgeschlossenen Kalibrationslauf ist kein `--resume` erforderlich.

Der Größen-/Methodenplan ist unter
`795124b0d89cf755f575a0de3049a2f3983aba1b` eingefroren. Der getrennte Befehl
`phase-10-size-methods` wertet pro Geometrie fünf vorab festgelegte Bulkregionen
auf 8×8, 10×10 und 12×12 aus. Vorlauf und Hauptlauf schreiben ausschließlich
in den neuen Stamm `results/phase_10_size_methods_v1`. Erst die Implementierung
committen; danach startest du selbst zunächst nur den Vorlauf:

```powershell
.\.venv\Scripts\python.exe -B -c "from toposc_lab.cli import main; main()" phase-10-size-methods --preflight --output results\phase_10_size_methods_v1
```

Nur wenn dessen Bericht 24/24 verfügbare Bewertungen und gültige Kontrollen
ausweist, wird der Hauptlauf separat gestartet:

```powershell
.\.venv\Scripts\python.exe -B -c "from toposc_lab.cli import main; main()" phase-10-size-methods --full --output results\phase_10_size_methods_v1
```

Nach einem Abbruch dient bei unveränderter Umgebung und Code-Revision:

```powershell
.\.venv\Scripts\python.exe -B -c "from toposc_lab.cli import main; main()" phase-10-size-methods --resume --output results\phase_10_size_methods_v1
```

Die nachfolgenden Kalibrationsbefehle dokumentieren den bereits abgeschlossenen
Vorgängerlauf und müssen dafür nicht erneut ausgeführt werden.

Der Hauptlauf in `results/phase_10_research_v1_storage_fix1/full` ist abgeschlossen:
Evolution und Zufallssuche hatten jeweils 0/32 Versuchstreffer. Das ist kein
Nachweis, dass geeignete Geometrien grundsätzlich unmöglich sind.

Als nächsten Schritt beschreibt das
[Kalibrationsprotokoll TOPOSC-P10-CAL-001](decisions/pre_phase_10_calibration_protocol_v1.md)
kontrollierte Änderungen an der erfolgreichen Quadrat-Referenz: nur Positionen,
nur Kanten, dann deren Kombinationen. Geplant sind 50 Bewertungen im technischen
Vorlauf und 386 im Hauptlauf, mit unveränderten Physik- und Gateparametern.
Das Protokoll ist separat unter `a1c5a087ed1383f496ebe6a14e1bc7eecaea568a`
eingefroren. Der neue Befehl `phase-10-calibration` implementiert den getrennten
Runner mit Live-Anzeige und Resume. Er darf erst nach einem separaten
Implementierungs-Commit gestartet werden. Alte Ergebnisse bleiben unangetastet.

Im Projektverzeichnis setzt du in einer neuen PowerShell zuerst die Umgebung:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='src'
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:BLIS_NUM_THREADS='1'
```

Danach startest du ausschließlich den technischen Vorlauf:

```powershell
.\.venv\Scripts\python.exe -B -c "from toposc_lab.cli import main; main()" phase-10-calibration --preflight --output results\phase_10_calibration_v1
```

Parallel kannst du den Fortschritt verfolgen:

```powershell
Get-Content results\phase_10_calibration_v1\events.jsonl -Wait
```

Erst wenn der Vorlauf bestanden ist, startest du ausdrücklich den Hauptlauf:

```powershell
.\.venv\Scripts\python.exe -B -c "from toposc_lab.cli import main; main()" phase-10-calibration --full --output results\phase_10_calibration_v1
```

Nach einem Abbruch verwendest du mit exakt derselben Umgebung und Code-Revision:

```powershell
.\.venv\Scripts\python.exe -B -c "from toposc_lab.cli import main; main()" phase-10-calibration --resume --output results\phase_10_calibration_v1
```

Die Anzeige nennt Block, Trajektorie, Positionsamplitude `amplitude`,
Kantenstufe `edge_step` und `clean_eligible`. `True` heißt, dass diese Variante
alle bisherigen sauberen Gates erfüllt; es ist noch kein Robustheits- oder
Geometrievorteil. Der Plan wird vor der ersten Physikbewertung vollständig
versiegelt. Die folgenden `phase-10-research`-Befehle dokumentieren den bereits
abgeschlossenen älteren Suchlauf und starten **nicht** diese Kalibration.

### Unseren Forschungslauf starten

Im Projektverzeichnis, nach dem Implementierungs-Commit, in PowerShell:

**Nach dem Windows-Speicherfix vom 09.09.2026:** zuerst den Fix committen.
Die beiden abgebrochenen Vorlaufversuche in `results/phase_10_research_v1`
bleiben unverändert als Fehlernachweis erhalten. Mit dem neuen Code dort nicht
`--resume` verwenden: Der Code-Commit passt absichtlich nicht mehr. Die Befehle
unten starten den dokumentierten technischen Wiederholungsvorlauf in einem
neuen Ordner; wissenschaftliche Parameter und Seeds bleiben unverändert.
Siehe [Speicher-Amendment A1](decisions/phase_10_research_storage_amendment_a1.md).

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='src'
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:BLIS_NUM_THREADS='1'
.\.venv\Scripts\python.exe -B -c "from toposc_lab.cli import main; main()" phase-10-research --preflight --output results\phase_10_research_v1_storage_fix1
```

Diese Thread-Einstellungen müssen vor dem Pythonstart gelten. Sie gehören zum
reproduzierbaren Protokoll; der Versuch nutzt bewusst einen numerischen Thread.
Es wird kein OpenAI-Modell aufgerufen, und der Physiklauf verbraucht keine
API-Tokens. Die Modellwahl in Codex betrifft nur unsere Arbeit am Code.

Der Vorlauf prüft zwei vollständige kleine Versuchspaare mit reservierten Seeds,
das Quadrat als Referenz und acht zusätzliche Geometrien: 129 Physikbewertungen
plus acht Geometriechecks. Er muss technisch bestehen; gute Treffer oder ein
Evolutionsvorteil sind dafür nicht erforderlich. Laufzeit und ETA entstehen aus
den Messungen auf deinem PC. Hauptsuch-Seeds und spätere Disorder-Seeds bleiben
bei diesem Start unbenutzt.

In einer zweiten PowerShell kannst du parallel mitlesen:

```powershell
Get-Content results\phase_10_research_v1_storage_fix1\events.jsonl -Wait
```

Die erste Konsole zeigt Versuch, Arm, Generation, Bewertungsslot, Treffer,
abgeschlossene/verbleibende Arbeit, Zeit/ETA, CPU/RAM und letzten versiegelten
Stand. Ein Heartbeat läuft auch während längerer Rechnungen. Eine hohe Zahl
erfolgreicher Bewertungen kann erneut bewertete Eliten enthalten; sie bedeutet
nicht ebenso viele verschiedene Entdeckungen.

Wenn der Vorlauf bestanden ist, startest du im selben vorbereiteten Terminal
den Hauptlauf ausdrücklich:

```powershell
.\.venv\Scripts\python.exe -B -c "from toposc_lab.cli import main; main()" phase-10-research --full --output results\phase_10_research_v1_storage_fix1
```

Das berechnet 32 Versuchspaare mit jeweils 32 Bewertungen pro Arm und das feste
Panel aus 35 Referenzen: 2.083 Bewertungsversuche. Ein einzelner Versuchsslot
enthält die komplette Topologie- und Randdiagnostik, nicht nur eine
Diagonalisierung. Keine Disorder- oder Größenvalidierung wird angehängt.

Bei `Ctrl+C`, Stromausfall oder geschlossenem Terminal setzt du mit denselben
Umgebungsvariablen und unverändertem Code fort:

```powershell
.\.venv\Scripts\python.exe -B -c "from toposc_lab.cli import main; main()" phase-10-research --resume --output results\phase_10_research_v1_storage_fix1
```

Versiegelte Referenzen und Versuchspaare werden geladen und geprüft. Ein
unvollständiges Paar wird mit denselben Seeds wiederholt; seine alten Dateien
bleiben erhalten, und widersprüchliche bereits gespeicherte Ergebnisse stoppen
die Wiederaufnahme. Zusätzliche Arbeit wird separat ausgewiesen. Ein bereits
vollständiger Lauf wird nur geprüft. Resume startet keinen neuen Hauptlauf
nach einem fertigen Vorlauf. Dafür verwendest du ausdrücklich `--full`.

Die Berichte liegen unter `preflight/report.md` bzw. `full/report.md` innerhalb
des Outputverzeichnisses. `summary.json` enthält die Trefferraten, Unsicherheit,
den gepaarten Vergleich, Rohwertvergleiche, deskriptive Sensitivitätsauswertung
und höchstens acht nominierte Geometrien je Arm. `hit_curves.png` zeigt, wie viele
Versuche bis zu jedem Bewertungsslot mindestens einen Treffer gefunden haben.
Kandidatenabbildungen verwenden die gespeicherten Geometrien; Orange markiert
die expliziten Randknoten. Leere Kandidatenlisten sind gültige negative Ergebnisse.

Die einzelnen Ordner enthalten verlustfreie Eingaben, wissenschaftliche Grids,
Pipelineergebnisse und Fehler sowie Generationen-Checkpoints. Der Bericht
beantwortet die Frage nach Suchleistung bei 64 Knoten. Eine statistische
Unterscheidung der Arme belegt noch keinen physikalischen Robustheitsvorteil.

Der Forschungsrunner veröffentlicht seit Amendment A1 je Generation eine eigene
`checkpoint_generation_0000.zip` bis `checkpoint_generation_0003.zip`.
Keine vorhandene Checkpoint-Datei wird ersetzt. Das benötigt mehr Speicherplatz
als ein einzelner letzter Checkpoint, erhält aber alle abgeschlossenen Präfixe.
Die unten beschriebene allgemeine API behält ihr bisheriges Ersetzungsverhalten;
mit `save_search_checkpoint(..., overwrite=False)` ist exklusives Speichern möglich.

### Eine andere Suche über die Python-API definieren

Diese fünf Entscheidungen kommen vor dem Start:

1. **Suchraum:** Welche Geometrien sind erlaubt? Ressourcen wie Knotenanzahl,
   Kopplungsbudget, räumliche Ausdehnung und Randbedingungen kontrollieren.
2. **Evaluator:** `evaluate_geometry(...)` mit dem passenden Modell, festen
   Parametern, Solvereinstellungen und anwendbaren Diagnostiken verbinden.
3. **Ziel:** Gewichte/Skalen oder getrennte Ziele vorab festlegen. Für Forschung
   Topologie, Schutz, Randzustände und Robustheit nicht zu einem beliebigen Score
   vermischen; die [Research Charter](decisions/pre_phase_9_research_charter.md)
   beschreibt die Evidenzgrenzen.
4. **Fortpflanzung:** `offspring_producer(request)` liefert exakt die verlangte
   Anzahl `OffspringProposal` mit Elternbezügen und Operator-Identifier.
   Mutationstyp, Wahrscheinlichkeiten und Crossover werden nicht automatisch gewählt.
5. **Budget und Seeds:** Population, Generationen und Such-Seeds einfrieren.
   Validierungs- und Bestätigungs-Seeds bleiben von der Suche getrennt.

Ein vollständiges Beispiel für diese Verbindungen steht in
[phase_10_search_demo.py](../examples/phase_10_search_demo.py). `demo_protocol()`
enthält die Konfiguration; `sample_genome`, `evaluate_genome` und
`produce_offspring` sind die drei austauschbaren Verbindungen. Für einen echten
physikalischen Versuch müssen sie zusammen einen neuen, dokumentierten Vertrag
erfüllen. Ein Austausch nur des Modells macht die Graph-Demo noch nicht zu einem
zulässigen Class-D-Experiment.

Im Python-Kontext mit Projektwurzel als Arbeitsverzeichnis:

```python
from examples.phase_10_search_demo import run_demo

comparison = run_demo()
trial = comparison.trials[0]
curve = trial.evolution_arm.progress  # ein Eintrag pro Bewertungsversuch
history = trial.evolution.fitness_history  # alle Generationen samt Fehlschlägen
last_population = trial.evolution.final_fitness.population
geometry = last_population.members[0].genome.to_geometry()
```

Mitglied 0 ist hier nur ein Beispiel, keine Zusage, dass es der beste Kandidat
über die gesamte Historie ist. Der vollständige Bewertungsdatensatz eines
Mitglieds bleibt unter `history[g].members[i].evaluation` erhalten; bei
Callback-Fehlern kann er None sein. Vor der Nutzung `is_available` prüfen.

## Checkpoints und Fortsetzen einer einzelnen Suche

Der Benchmark selbst besitzt keinen Suite-Checkpoint. Seine einzelnen
Evolutionsresultate können mit Phase 10.18 gespeichert werden. Für eine längere
einzelne Suche verwendet man direkt `run_generation_loop` und dessen Callback.
Das folgende Beispiel ist ebenfalls nur die Engineering-Demo. Es schreibt
ausschließlich nach `results/phase10_manual_demo`, das noch nicht existieren darf:

```python
from pathlib import Path
from examples.phase_10_search_demo import (
    demo_protocol, sample_genome, evaluate_genome, produce_offspring,
)
from toposc_lab.search import (
    create_initial_population, create_search_checkpoint,
    save_search_checkpoint, run_generation_loop,
)

protocol = demo_protocol()
initial = create_initial_population(
    [sample_genome(20000 + i) for i in range(protocol.population_size)],
    validity_policy=protocol.validity_policy,
)

def evaluate_member(member):
    seed = 30000 + member.generation_index * protocol.population_size + member.member_index
    return evaluate_genome(member.genome, seed)

output = Path('results/phase10_manual_demo')
output.mkdir(parents=True, exist_ok=False)
checkpoint_path = output / 'checkpoint.zip'

def save_prefix(prefix):
    checkpoint = create_search_checkpoint(
        prefix,
        requested_generation_count=protocol.generation_config.generation_count,
        evaluator_identifier=protocol.evaluator_identifier,
        code_version=protocol.code_version,
    )
    save_search_checkpoint(checkpoint_path, checkpoint)

result = run_generation_loop(
    initial, definition=protocol.definition, evaluator=evaluate_member,
    config=protocol.generation_config, seed=40000,
    offspring_producer=produce_offspring,
    producer_identifier=protocol.producer_identifier,
    checkpoint_callback=save_prefix,
)
```

Der Callback ersetzt atomar den eigenen Checkpoint nach Generation null und
jeder abgeschlossenen Generation. Bei einem Abbruch bleibt die letzte vollständig
gespeicherte Generation verfügbar. Zum Fortsetzen müssen dieselben Funktionen und
dieselbe Protokollversion wieder definiert/importiert sein:

```python
from toposc_lab.search import load_search_checkpoint, resume_search

checkpoint = load_search_checkpoint('results/phase10_manual_demo/checkpoint.zip')
continued = resume_search(
    checkpoint,
    evaluator=evaluate_member,
    evaluator_identifier=protocol.evaluator_identifier,
    offspring_producer=produce_offspring,
    producer_identifier=protocol.producer_identifier,
    code_version=protocol.code_version,
    checkpoint_callback=lambda saved: save_search_checkpoint(
        'results/phase10_manual_demo/checkpoint.zip', saved,
    ),
)
result = continued.result
```

Ein bereits abgeschlossener Lauf wird nicht neu gerechnet. Resume verlängert das
Ziel nicht und erlaubt keinen heimlichen Wechsel von Fitness, Seeds, Operatoren
oder Laufzeitversionen. Auf einem anderen PC müssen Python-/Paketversionen passen;
identische Hardware-/BLAS-Ergebnisse werden dadurch nicht garantiert. Private
Zufallszustände in eigenen Callbacks werden nicht mitgespeichert.

## Was „Phase 10 fertig“ bedeutet

Die Software-Bausteine bis einschließlich Vergleich mit Zufallssuche sind fertig.
Ein wissenschaftlich belastbarer Suchvorteil für die Forschungsfrage ist damit
noch nicht nachgewiesen. Dafür braucht es einen eigenen eingefrorenen physikalischen
Versuchsplan, ausreichend unabhängige Läufe, Unsicherheitsanalyse sowie unabhängige
Topologie-, Robustheits- und Größenvalidierung. Schlechte oder negative Ergebnisse
dürfen nicht durch zusätzliche Seeds ersetzt werden.

Phase 11 (Dataset-Infrastruktur) und spätere ML-Schritte werden dadurch nicht
automatisch begonnen. Details zum Vergleichsvertrag stehen in der
[Phase-10.20-Entscheidung](decisions/phase_10_20_search_benchmark.md).
