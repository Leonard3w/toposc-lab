# Freier Graphbauer mit neuronalem Lernen und Liveansicht

Vorhanden: freie Punktplatzierung, Kantenbau, neuronaler REINFORCE-Actor,
Zufallsbaseline, feste Punkte als einfacherer Modus, Checkpoints, Resume und Replay.
Das lokale Netz benötigt nur NumPy. Keine API-Schlüssel, kein PyTorch, kein Streamlit.

**Die ausführbare Aufgabe ist zunächst eine geometrische Demo:** kurze Verbindungen
bei festen Ressourcen. Sie berechnet keine Supraleitung und ist kein neuer
Phase-10-Physiklauf. Architektur, Grenzen und Forschungsschritte stehen im
[Entscheidungsentwurf](decisions/constructive_graph_learning_v1.md).

## 1. PowerShell vorbereiten

In `C:\Users\Leonard\Documents\GitHub\toposc-lab`:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='src'
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:BLIS_NUM_THREADS='1'
```

Alle Befehle starten bewusst durch dich. Die folgenden Outputnamen sind Beispiele;
für einen neuen Lauf muss das Verzeichnis neu sein. Alte Archive nicht löschen.

## 2. Liveansicht zuerst öffnen

In einer zweiten vorbereiteten PowerShell:

```powershell
.\.venv\Scripts\python.exe -B -m toposc_lab.design watch --output results\graph_builder_demo_v1
```

Im Browser `http://127.0.0.1:8766` öffnen. Die Ansicht wartet, bis das Training Daten
schreibt. Sie zeigt Punkte, Kanten, aktuellen Schritt, Belohnung und Bewertungsstatus.
Orange markiert die letzte Änderung. `Anzeige pausieren` hält nur die Ansicht an.
`Episode abspielen` verwendet die **nullbasierte** Episodennummer, z. B. 0 für
die erste Episode. Livezustände können bei schnellen Schritten übersprungen werden;
Replay spielt jeden gespeicherten Schritt ab.

## 3. Neuronale Geometrie-Demo starten

In der ersten PowerShell:

```powershell
.\.venv\Scripts\python.exe -B -m toposc_lab.design train-demo --episodes 100 --seed 20260911 --policy neural --output results\graph_builder_demo_v1
```

Standard: 12 frei platzierte Punkte, 16 Verbindungen in einer 4×4-Fläche.
Der Actor wählt zuerst kontinuierlich vorgeschlagene Punkte, danach Kanten.
Das ist kein verstecktes Quadrat und keine direkte kontinuierliche Koordinatenpolicy:
je Schritt stehen bis zu 32 zufällige kontinuierliche Punktvorschläge zur Auswahl.

Die Konsole meldet jede abgeschlossene Episode. −1 bedeutet geometrische Ablehnung.
Bei gültigen Endgraphen bedeutet eine höhere Demo-Belohnung kürzere mittlere
Verbindungslänge. `None` bedeutet einen Evaluatorfehler ohne Lernupdate.
Weder positives Reward noch ein ungewöhnliches Bild beweisen Topologie.

## 4. Unterbrechen und fortsetzen

Windows-Liveanzeige: Ein vorübergehender `PermissionError` beim Ersetzen von
`live.json` bricht den Lerner nicht mehr ab. Das nächste Ereignis versucht die
Anzeige erneut zu aktualisieren; Ereignisarchiv und Checkpoints bleiben strikt.
Für den bekannten ursprünglichen Codehash ist Resume nach dieser reinen
Anzeige-Korrektur zugelassen. Das Originalmanifest bleibt erhalten, neue
Episoden speichern zusätzlich ihren tatsächlichen Ausführungscodehash.

Ctrl+C beendet den Lerner kontrolliert. Mit denselben Optionen fortsetzen:

```powershell
.\.venv\Scripts\python.exe -B -m toposc_lab.design train-demo --episodes 100 --seed 20260911 --policy neural --output results\graph_builder_demo_v1 --resume
```

Versiegelte Episoden werden nicht neu gerechnet. Eine unterbrochene Episode
wird mit denselben Seeds in einem neuen Ausführungsordner wiederholt. Änderungen
an Regeln, Budget, Kern-Code, Umgebung oder Startpolicy verhindern Resume.
Ein fertiger Lauf wird nur geprüft. Nach hartem Prozessabbruch kann `writer.lock`
stehenbleiben: erst prüfen, dass kein Lerner mehr läuft, dann diese Sperre gezielt
entfernen; keine Archive oder Ausführungen entfernen.

## 5. Einfachere Aufgaben und andere Regeln

Regeln als JSON speichern (PowerShell UTF-8-BOM wird unterstützt):

```powershell
.\.venv\Scripts\python.exe -B -m toposc_lab.design config | Set-Content -Encoding UTF8 graph_builder_rules.json
```

Darin lassen sich Punktzahl, Kantenbudget, Flächengröße, Abstand, Reichweite und
Gradgrenzen verändern. Die Standardwerte sind Engineeringparameter, keine
wissenschaftliche Empfehlung für freie Class-D-Geometrien.

Nur Verbindungen auf festen 3×3-Quadratpunkten lernen:

```powershell
.\.venv\Scripts\python.exe -B -m toposc_lab.design config --square 3 | Set-Content -Encoding UTF8 graph_builder_square_rules.json
.\.venv\Scripts\python.exe -B -m toposc_lab.design train-demo --rules graph_builder_square_rules.json --episodes 100 --seed 20260911 --output results\graph_builder_square_demo_v1
```

`fixed_points` kann auch eine eigene Liste unregelmäßiger Koordinaten enthalten.
Beim festen Modus entfällt nur die Punktplatzierung; gestartet wird weiterhin ohne
Verbindungen. Große Systeme erst geometrisch profilieren, nicht direkt auf 16×16
hochskalieren und eine kurze Laufzeit voraussetzen.

## 6. Zufallsbaseline und andere Bewertungen

### Neuer Dreiarm-Pilot: Lernen gegen eingefrorenes Netz und Zufall

Der gemeinsame Pilot verwendet zehn neue Seeds `20261001` bis `20261010`,
100 Episoden pro Arm und Seed, insgesamt **3.000 Bauversuche**. Keine Physik.
Die Standardregeln der bisherigen freien 12-Punkte-Demo bleiben gleich.
`neural` lernt, `frozen` verwendet dieselben initialen Gewichte ohne Updates,
`random` zieht gleichverteilt aus den zulässigen Aktionen. Das eingefrorene Netz
ist nicht das bereits trainierte Netz aus dem alten Archiv.

In der wie oben vorbereiteten PowerShell selbst starten:

```powershell
.\.venv\Scripts\python.exe -B -m toposc_lab.design.benchmark --output results\graph_builder_three_arm_v1
```

Liveansicht in einer zweiten vorbereiteten PowerShell:

```powershell
.\.venv\Scripts\python.exe -B -m toposc_lab.design watch --output results\graph_builder_three_arm_v1 --port 8768
```

Browser: `http://127.0.0.1:8768`. Die gemeinsame Liveansicht folgt dem gerade
laufenden Arm und zeigt dessen Namen und Seed. Die Rewardkurve gehört jeweils
zum aktuellen Arm, nicht zu allen Armen zusammen. Für Replay eines bestimmten
Arms den Viewer auf dessen Unterordner richten, z. B.
`results\graph_builder_three_arm_v1\seed_20261001\frozen`.

Nach Ctrl+C mit unveränderten Einstellungen fortsetzen:

```powershell
.\.venv\Scripts\python.exe -B -m toposc_lab.design.benchmark --output results\graph_builder_three_arm_v1 --resume
```

Bereits fertige Arme werden geprüft, nicht erneut trainiert. Der Stammordner
enthält `benchmark_manifest.json` und nach Abschluss `benchmark_complete.json`;
darunter liegen vollständige Einzelarchive pro Seed und Arm. Die Ausführungsreihenfolge
rotiert, damit nicht immer derselbe Arm zuerst läuft. Kein Datencache wird zwischen
Armen geteilt. Alle Arme starten pro Seed mit frischem Zustand.

Primäre deskriptive Größe: Differenz der mittleren Belohnung aller 100 Episoden
zwischen `neural` und `frozen`, getrennt auch gegen `random`. Sackgassen mit −1
bleiben darin enthalten. Bei operativen fehlenden Bewertungen ist der jeweilige
Mittelwert nicht verfügbar; solche Paare werden ausdrücklich gezählt. Gültigkeitsrate,
mittlere Qualität gültiger Graphen, Bestwert, Evaluatoraufrufe, Cachetreffer und
zusätzliche Ausführungen bleiben separat sichtbar. Episoden sind keine unabhängigen
statistischen Wiederholungen; die zehn Seeds sind die Vergleichseinheiten.

Das ist ein begrenzter Engineering-Pilot, keine bereits ausreichend gepowerte
Überlegenheitsstudie. Es werden keine guten Seeds ausgewählt, schlechten Seeds
ersetzt oder Parameter während des Piloten angepasst. Die zuvor betrachteten
Einzelläufe zählen als Entwicklungsmaterial, nicht zu diesen zehn neuen Paarungen.
Die laufende Trainingsleistung ist auch kein Holdout-Test des fertig trainierten
Netzes. Eine spätere Transfer-/Generalisationsthese braucht eine eigene eingefrorene
Policy und neue Testepisoden.

### Einzelne Zufallsbaseline oder eigene Bewertung

Ein separater Lauf unter denselben Regeln:

```powershell
.\.venv\Scripts\python.exe -B -m toposc_lab.design train-demo --episodes 100 --seed 20260911 --policy random --output results\graph_builder_random_demo_v1
```

Zum Anzeigen einen zweiten Viewer mit `--port 8767` auf diesen Ordner richten.
Diese zwei Demoaufrufe sind noch kein statistisch ausreichender Benchmark.

Eigene deterministische Engineeringbewertung über die Python-API:

```python
from pathlib import Path
from toposc_lab.design import BuildRules, Evaluation, NeuralPolicy, TrainingConfig, run_training

class MyEvaluation:
    identifier = 'my_geometric_objective_v1'
    deterministic = True  # Nur wenn der Seed das Ergebnis wirklich nicht beeinflusst.

    def __call__(self, state, rules, seed):
        geometry = state.geometry()
        # Beispiel: Ausdehnung in x. Engineering, kein physikalischer Schutz.
        extent = max(x for x, y in state.points) - min(x for x, y in state.points)
        return Evaluation(extent / rules.width, 'engineering_only', {'extent_x': extent})

run_training(BuildRules(), TrainingConfig(100, 123), NeuralPolicy(seed=123),
             MyEvaluation(), Path('results/my_geometry_demo_v1'))
```

`state.geometry()` kann an bestehende Geometry-/Genome-APIs übergeben werden.
Ein wissenschaftlicher Physikevaluator benötigt jedoch zuerst einen eigenen
geprüften Rand-, Mess-, Ressourcen- und Referenzvertrag für freie Koordinaten.
Die alte Quadratmessung darf nicht einfach als neues Reward verwendet werden.

## Artefakte

- `manifest.json`: Regeln, Seeds, Budget, Code-/Umgebungsbindung.
- `attempts_XXXXXX/execution_XXXX/events.jsonl`: jeder Aufbau- und Bewertungsstatus.
- `episode_XXXXXX.json`: versiegelter Endzustand, Bewertung und vollständige Netzgewichte.
- `complete.json`: festes Budget abgeschlossen; Historie und letzte Checksumme.
- `live.json`: austauschbarer aktueller Anzeigestand.

Snapshots enthalten Koordinaten und orientiert rekonstruierbare Kanten, keine
animierte Behauptung eines optimalen Ergebnisses. Alte Forschungsarchive und
geschützte Dateien werden vom Graphbauer nicht verwendet.
