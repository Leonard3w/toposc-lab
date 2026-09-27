# TOPOSC Research Studio

Das Studio erweitert TOPOSC LIVE. Es verwendet dessen Prozesse, Datenbank,
Checkpoints und exakte Physikauswertung. Modell-/Lehrlabore, alte Kampagnen und
Gruppen bleiben im selben Anwendungsfenster erreichbar.

## Start unter Windows

Die vorhandene Projektumgebung einmalig entsprechend README einrichten:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[live,app]"
```

Danach `start_toposc.bat` doppelklicken. Alternativ:

```powershell
.venv\Scripts\python.exe -B -m toposc_live --root results
```

`toposc-live` und `toposc-ui` sind Aliasse dieser Anwendung. Der Launcher legt
keine Umgebung an und installiert nichts automatisch. Lehrlabore werden erst
beim Öffnen als lokaler, eingebetteter Streamlit-Dienst gestartet. Physikworker
sind unabhängig vom Fenster; Schließen der Anwendung beendet laufende
Forschungsexperimente nicht.

## Ein Experiment konfigurieren

Unter **Research Studio → New Experiment** ein Preset laden. Die zehn Bereiche
sind Experiment, Geometry, Physics, Disorder / Ensemble, Search Space,
Search Method, Search Settings, Objectives, Numerical Settings und Output / Storage.

Standard Mode zeigt häufige Einstellungen. Expert Mode zeigt zusätzliche
Parameter und die vollständige JSON-Konfiguration. Der Wechsel des Modus
ändert keine wissenschaftlichen Werte. Eine JSON-Liste hat beispielsweise die
Form `[3, 6, 9]`; Seeds sind explizite ganze Zahlen.
Gepaarte Seeds koppeln Potentiale nach Site-Index, bei freien Geometrien nicht
nach identischen räumlichen Punkten. Bei unabhängiger Seed-Politik wird der
realisierte Seed deterministisch aus Geometrie und angefordertem Seed
abgeleitet; beide werden gespeichert.

Die Presets `quick_test`, `phase19_like`, `disorder_scan`,
`candidate_validation` und `broad_random_search` sind normale, bearbeitbare
Konfigurationen. Ihre Namen starten keinen versteckten Speziallauf.
`phase19_like` ist insbesondere keine identische Wiederholung der historischen
Phase-19-Kohorte. Es verwendet neue Seeds und ein kleineres konfigurierbares
Budget. Save config und Load config dienen auch eigenen Presets.

**Locked / variable:** Beim Sampling freier Graphen müssen Koordinaten und
Konnektivität ausdrücklich variabel sein. Umverdrahtete Quadratgitter ändern
nur Konnektivität. Reguläre Referenzen und feste Folge-Kandidaten sperren ihre
Geometrie. t, mu und Delta können als feste Experimentwerte geändert werden;
die derzeitigen Suchstrategien mutieren diese Werte nicht. Nicht unterstützte
Variabilität wird abgelehnt. Parametrische Modellsweeps sind außerdem in den
vorhandenen Model Labs verfügbar.

Sampling bietet reguläre Quadratgitter, umverdrahtete Quadratgitter,
amorph-planare Graphen und räumlich eingebettete Radiusgraphen. Die vorhandenen
Zufalls-, Evolutions-, MAP-Elites- und Surrogat-MAP-Elites-Strategien gelten für
den Suchraum `fixed_connectivity`; freie Graphfamilien bieten outcome-blind
Sampling. Das Studio fügt keine ML-Verfahren hinzu.

Die Domain muss zur Geometrie passen. Quadratgitter haben Einheitsabstand und
Box `[0, side-1]²`. Freie Generatoren verwenden eine quadratische Box mit
explizitem `box_maximum`. Physischer Randstreifen, Localizer-Inset und
Voronoi-Padding sind getrennte Begriffe. Grad-, Kanten-, Abstands- und
Kreuzungsbedingungen werden vor einer exakten Auswertung geprüft.

## Vorschau und Ausführung

**Review Run / Preview** prüft die Konfiguration, zeigt gesperrte Größen,
Variablen, Seeds, Budget, Ressourcenabschätzung und den Konfigurationshash.
Nur **START** aus dieser Ansicht erzeugt und startet den Lauf. Nachträgliche
Änderungen erfordern eine neue Vorschau. Verzeichnisse mit vorhandenen Daten
werden nicht überschrieben.

Geplante Realisierungen sind bei zufälliger Konstruktion eine Obergrenze:
verworfene und doppelte Vorschläge können sie reduzieren. Das Versuchslimit
zählt auch unterbrochene und fehlgeschlagene exakte Versuche. Ein Kandidat
wird nur begonnen, wenn das Restbudget seine geplanten Stufen abdecken kann;
zusätzliche Wiederholungen können dennoch zu einem unvollständigen Lauf führen.

Pause und Graceful Stop greifen nach der laufenden exakten Stufe. Resume
verwendet den vorhandenen SQLite-Fortschritt. Geometrieerzeugung und eine
einzelne numerische Operation sind nicht präemptiv unterbrechbar. Das Zeitlimit
ist daher eine kooperative Grenze, kein harter Prozessabbruch.

## Ergebnisse und Candidate Explorer

Gespeicherte Experimente öffnen oder aus der Liste auswählen. Tabellen lassen
sich sortieren und nach ID, Familie, Status und gespeicherten Kennzahlen
filtern. Kandidatenvergleich, Geometrie, Rohdaten, Herkunft und Checkpoints
bleiben im vorhandenen Dashboard verfügbar. Die Detailansicht liest Spektrum
und Ortswahrscheinlichkeiten aus gespeicherten Stufen; sie berechnet keine
Physik beim Öffnen nach.
Die ausgewählten Observablen bestimmen die angebotenen numerischen Filter;
die vollständigen Rohdaten bleiben erhalten. Ein Filter trifft zu, wenn
mindestens ein entsprechender gespeicherter Wert innerhalb seiner Grenzen
liegt. Er fordert keine Übereinstimmung aller Seeds oder W-Werte.
Chern filtert den Bulk-Mittelwert des Chern-Markers, der Interior-Localizer-
Filter den kleinsten gültigen inneren Gap. Die Spalte Exact score folgt
dagegen dem gespeicherten Experimentziel.

Q, Interior-Localizer-Gap, Randgewicht, Zentrumgewicht, Chern-Marker und
Majorana-Diagnostik bleiben getrennt. Ein fehlender Wert ist nicht Null.
Ein Interior-Localizer-Gap ist kein Bulk-Spektralgap. Die numerischen Daten
begründen allein weder eine thermodynamische Phase noch getrennte Majoranas.
Prognosen bestehender Surrogatstrategien sind als solche gekennzeichnet.

Historische Phase-18/19-Studien erscheinen über einen lesenden Adapter. Ihre
Datenbanken werden nicht migriert; Laufkontrollen sind dafür deaktiviert.
Für eine identische historische Fortsetzung sind der archivierte Quellstand
und dessen Laufzeitumgebung maßgeblich.

## Folgeexperimente

Kandidaten auswählen, **Create follow-up experiment**, W-Werte und frische
Seeds angeben. Daraus entsteht eine neue normale `ExperimentConfig` mit
verlustfreien Geometriearchiven, ursprünglichen Kandidaten-IDs, physischen
Hashes und Quelllauf. Erst Preview → START beginnt die Rechnung. Die Auswahl
aus vorherigen Ergebnissen bleibt dokumentiert. Ein geklonter historischer
Lauf mit denselben Seeds ist ausdrücklich keine unabhängige Bestätigung.

## CLI mit derselben Konfiguration

```powershell
.venv\Scripts\python.exe -B -m toposc_lab.research preview --config experiment.json
.venv\Scripts\python.exe -B -m toposc_lab.research create results\mein_lauf --config experiment.json
.venv\Scripts\python.exe -B -m toposc_lab.research run results\mein_lauf
.venv\Scripts\python.exe -B -m toposc_lab.research inspect results\mein_lauf
```

`output_directory` muss beim Schema 2 mit dem angegebenen Laufverzeichnis
übereinstimmen. `create` erzeugt nur den Lauf; `run` startet ihn ausdrücklich.
CLI und GUI-Worker rufen `ResearchService.run_experiment()` und dieselbe
`ResearchEngine` auf. Kontrollbefehle: `pause`, `resume`, `stop`, `checkpoint`.

## Daten und Erweiterungen

`research.sqlite3` bleibt autoritativ. Gespeichert werden aufgelöste
Konfiguration samt Hash, Manifest, Quellarchiv, Geometrien, Seeds,
Vorschläge einschließlich Ablehnungen, exakte Stufen, Laufstatus, Zeiten und
Checkpoints. Studio-Läufe exportieren JSONL-Rohdaten und CSV-Diagnostik unter
`reports/`; optionale Plots sind davon abgeleitet.

Schema 1 behält seine Serialisierung und Konfigurationshashes. Schema 2 ergänzt
das vorhandene Konfigurationsobjekt, ersetzt es nicht. Der Adapter
`phase20.configurable-chiral-p-wave.v1` verwendet dieselben Hamiltonian-,
Disorder-, Localizer- und Diagnostikfunktionen. Historische Adapter bleiben
eingefroren.

Neue Suchverfahren registrieren sich an der bestehenden Strategie-Schnittstelle
mit `initialize`, `propose`, `select`, `observe`, `checkpoint`, `resume` und
`summarize`. Ihre Fähigkeit zu Räumen und Zielen muss ausdrücklich validiert
und als Metadaten angegeben werden. Keine zweite Engine oder UI einführen.

Aktuelle Grenzen: ein exakter Worker pro Lauf, vollständiger dichter Solver,
offene 2D-p+ip-Geometrien im Research-Adapter. Keine gewichteten oder
Pareto-Optimierungsziele, kein neuer Bulk-Gap-Schätzer, keine Speicherung aller
komplexen Eigenvektoren. Vorhandene Ortsprofile und Eigenwerte werden erhalten.
Die Versionierung dokumentiert unveränderliche mathematische Konventionen.

Unverändert festgelegt sind insbesondere die Q-Zulässigkeitsregel und
Erfolgsschwelle 0,20, die p+ip-Konvention und Nambu-Basis, die uniforme skalare
Onsite-Verteilung `[-W/2,W/2]`, offene Ränder, die Wilson-Intervallmethode und
das vorhandene räumliche Probenrezept. Ein Konfidenzniveau ist einstellbar;
die Methode wird damit nicht zu einem simultanen Intervall über Kandidaten.
Die Exportspalten `edge_weight` und `center_weight` behalten ihre festen
Vergleichsdefinitionen: Streifenbreite 1 beziehungsweise ein Quadrat mit
Halbbreite 1 um das Domain-Zentrum. Andere einstellbare Randbreiten stehen
in den Rohdaten; fehlt Breite 1, bleibt `edge_weight` leer. Die geometrische
Domain-Toleranz 1e-10 und der Voronoi-Algorithmus bleiben unabhängig von der
einstellbaren Eigensystemtoleranz. Das Studio ändert diese Definitionen nicht.
