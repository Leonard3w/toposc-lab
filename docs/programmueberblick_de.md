# TopoSC Lab
## Programmüberblick und Funktionsstand
Stand: 25. September 2026 | Version im Paket: 0.1.0 | Entwicklungsstand: Phase 17.2
Quellstand: Snapshot 37c9279, fachlicher Vorgänger eb7b3c3.

TopoSC Lab ist ein lokales Forschungs- und Lernprogramm für topologische Materialien, Supraleitung und ausgewählte Themen der statistischen Physik. Es berechnet endliche Modelle, zeigt ihre Energien und Zustände und kann selbstständig nach interessanten Verbindungsstrukturen suchen.

Der Funktionsumfang ist inzwischen deutlich größer als die ursprüngliche Kitaev-Kette. Zum Projekt gehören eine Python-Bibliothek, eine Kommandozeile, eine Streamlit-Oberfläche im Browser und die native Windows-Anwendung TOPOSC LIVE. Sie haben unterschiedliche Aufgaben und verwenden gemeinsame Rechenbausteine.

### Was bereits vorhanden ist
- Simulationen mit sieben direkt in der Streamlit-Modellauswahl registrierten Modellen; zusätzliche geometriebasierte Modelle in der Python-API.
- Spektren, Wellenfunktionen, Lokalisierung, Parameterstudien und mehrere Verfahren zur topologischen Analyse.
- Geometriegeneratoren, Unordnungstests, evolutionäre Suche, Datensatzverwaltung und lernende Priorisierung teurer Rechnungen.
- Eine persistente Forschungswerkbank mit Budgets, exakter Validierung, Archiv, Pause, Wiederaufnahme und Berichten.
- Interaktive Lernlabore für Quantengase, Landau-Niveaus und den ganzzahligen Quanten-Hall-Effekt.

### So ist dieser Bericht zu lesen
„Implementiert“ bedeutet: im aktuellen Quellcode vorhanden. Testzahlen belegen jeweils den angegebenen Prüfumfang. Ein erfolgreich getesteter Programmablauf ist noch kein Nachweis einer neuen physikalischen Phase oder einer überlegenen Suchmethode.

Dieser Bericht beschreibt Funktionen aus Nutzersicht und die wichtigsten technischen Zusammenhänge. Er ist kein vollständiges Verzeichnis jeder internen Python-Funktion. Historische Messwerte sind als solche gekennzeichnet; für diesen Bericht wurde kein neues wissenschaftliches Langzeitexperiment gestartet.

### Inhalt
2 Bedienung und Einstieg · 3 Modelle und Rechenablauf · 4 Geometrien · 5 Auswertung und Robustheit · 6 Daten, Lernen und Suche · 7 Forschungswerkbank · 8 Lernlabore · 9 Ergebnisse und Nachweise · 10 Grenzen und Quellen

[[PAGE]]
# 2 | Bedienung und Einstieg
### Streamlit: Modelle erkunden und Physik verstehen
Die Browseroberfläche hat fünf Hauptbereiche: Topological superconductors, Quantum Hall / Landau levels, Quantum gases, Research studies und Autonomous Research. Im topologischen Bereich stehen Single simulation, Parameter scan und Model guide bereit.

- Single simulation: Modell wählen, Parameter einstellen und berechnen. Die Oberfläche zeigt Spektrum, Zustände, geometriebezogene Lokalisierung und Messgrößen. Ergebnisse lassen sich als NPZ-Datei herunterladen.
- Parameter scan: einen kontinuierlichen Modellparameter über ein Intervall variieren. Das Programm berechnet für jeden Punkt dieselben definierten Größen und zeigt Veränderungen von Spektrum und Observablen.
- Model guide: modellbezogene Erklärungen unterstützen die Interpretation der Parameter und Resultate.
- Study explorer: gespeicherte Studien laden, ihre Metadaten prüfen und kompatible Resultate vergleichen. Studien bewahren Parameter und Arrays gemeinsam auf.

### TOPOSC LIVE: länger laufende Forschung verwalten
Die native Qt-Anwendung dient zum Konfigurieren, Starten und Beobachten von Kampagnen sowie zur Kandidateninspektion. Diagramme und Tabellen zeigen gespeicherte exakte Evidenz, Budgets, Fortschritt und Diagnostik. Campaign Groups vergleichen kompatible unabhängige Seeds und exportieren Zusammenfassungen.

Die ältere Discovery-Kampagnenansicht und die neuere Autonomous-Research-Werkbank haben verschiedene Steuerungsmöglichkeiten: Die neue Werkbank unterstützt kooperative Pause und Safe Stop. Die ältere Discovery-Ansicht bietet Start, Beobachtung und geprüfte Wiederaufnahme, aber keine echte Pause mitten im Lauf.

### Start aus dem Projektordner in PowerShell
```text
.venv/Scripts/toposc-ui.exe
.venv/Scripts/toposc-live.exe --root results
.venv/Scripts/python.exe -m toposc_lab kitaev-scan --L 60 --mu-min -4 --mu-max 4
```
Die ersten beiden Befehle benötigen die optionalen Pakete für app beziehungsweise live. Die Programme können bei Bedarf mit uv in der vorhandenen Umgebung installiert werden:
```text
uv pip install --python .venv/Scripts/python.exe -e ".[app,live]"
```
Ein guter erster Ablauf: Kitaev chain öffnen, ein kleines System berechnen, das chemische Potential variieren und anschließend Spektrum und Randlokalisierung gemeinsam betrachten. Das Öffnen einer Oberfläche startet für sich noch keine Forschungskampagne.

Quellen: pyproject.toml; src/toposc_lab/app/streamlit_app.py; docs/toposc_live.md; docs/campaign_groups.md.

[[PAGE]]
# 3 | Modelle und Rechenablauf
### Sieben Modelle in der allgemeinen Modellauswahl
- Kitaev chain: eindimensionaler p-Wellen-Supraleiter. Länge, Hopping, chemisches Potential, Paarung, Randbedingungen und Onsite-Unordnung erlauben Untersuchungen von Endzuständen und Spektren.
- Kitaev ladder: gekoppelte Kitaev-Ketten. Zusätzliche Verbindungen und Paarung zwischen den Ketten machen quasi-eindimensionale Strukturen zugänglich.
- SSH chain: dimerisierte Kette mit zwei alternierenden Kopplungen. Sie ist ein anschauliches Referenzmodell für chirale Topologie und Randzustände.
- Qi-Wu-Zhang: zweibandiges Modell eines Chern-Isolators auf einem quadratischen Gitter. Der Massenparameter steuert das Phasenverhalten.
- BHZ: vierbandiges Gittermodell eines Quanten-Spin-Hall-Isolators. Es erweitert den Vergleich auf eine andere Symmetrie- und Modellstruktur.
- Graphene: spinloses Tight-Binding-Modell mit nächster Nachbarschaft auf dem Honigwabengitter.
- Haldane: Honigwabenmodell mit komplexem Hopping zu übernächsten Nachbarn und Sublattice-Masse für Chern-Physik.

Zusätzlich enthält die API GeometryKitaevChain und ChiralPWaveModel. Letzteres ist die physikalische Grundlage der aktuellen festplatzbasierten Forschungswerkbank. Diese zusätzlichen Klassen sind keine weiteren Einträge der allgemeinen Sieben-Modell-Auswahl.

### Wie eine Simulation funktioniert
1. Parameter validieren: typisierte Pydantic-Modelle prüfen die Eingabe.
2. Geometrie bauen: Orte, Koordinaten, Verbindungen und Randinformationen festlegen.
3. Hamiltonmatrix erzeugen: Kopplungen und lokale Terme beschreiben die Energie. Supraleitende Modelle verwenden eine Bogoliubov-de-Gennes-Matrix mit Teilchen- und Lochkomponenten.
4. Eigenproblem lösen: numerische Diagonalisierung liefert Energien und Eigenzustände.
5. Auswerten: Spektrallücken, Zustandsgewichte, Lokalisierung und anwendbare Topologieverfahren berechnen.
6. Darstellen und speichern: Diagramme, Arrays, Parameter und Herkunft bilden ein nachvollziehbares Ergebnis.

### Wiederverwendbare Hamilton-Bausteine
Die Bibliothek enthält Tight-Binding, Flussphasen, Nambu-Basis, BdG-Aufbau, s-, p- und d-Wellen-Paarung, Rashba- und Zeeman-Terme sowie Unordnungsbausteine. Ein vorhandener Baustein bedeutet nicht, dass jede beliebige Kombination bereits als fertiges GUI-Modell angeboten wird.

Quellen: src/toposc_lab/app/registry.py; models/; hamiltonians/; solvers/ unter src/toposc_lab.

[[PAGE]]
# 4 | Geometrien und Verbindungen
### Welche Strukturen erzeugt werden können
Die allgemeine Geometriebibliothek trennt die Form eines Systems vom physikalischen Modell. Nicht jeder Generator ist mit jedem Modell oder jeder Suchwerkbank kombinierbar; die jeweilige Validierung entscheidet über die Zulässigkeit.

- Regelmäßige Strukturen: Kette, Ring, Quadrat-, Dreiecks-, Honigwaben- und Kagome-Geometrien; kubische und raumzentriert-kubische Gitter. Leiter- und Bandgeometrien ergänzen die Modell- und Gitterbausteine.
- Bäume und Netze: allgemeine Bäume, Cayley-Bäume, Zufallsgraphen, zufällig reguläre Graphen, Small-World- und Scale-Free-Netze.
- Quasiperiodische Strukturen: Fibonacci- und Silver-Mean-Ketten sowie Ammann-Beenker-Patches.
- Fraktale: Sierpinski-Dreieck, Sierpinski-Teppich und Menger-Schwamm.
- Koordinatenbasierte Geometrien: unregelmäßige Cluster, Abstands-Cutoff- und k-Nächste-Nachbarn-Graphen, Hard-Core-Planargeometrien und regelbasiert erzeugte Graphen.

### Was die Geometrieschicht übernimmt
Validierung kontrolliert strukturelle Bedingungen. Serialisierung speichert eine Geometrie, Hashes unterstützen ihre Identifikation. Deskriptoren beschreiben zum Beispiel Grade, Verbindungsstruktur, Randanteile oder Bindungslängen. Die allgemeine Suche besitzt Mutationen für Knoten, Kanten und Koordinaten; konkrete Suchräume schränken diese gezielt ein.

### Der aktuelle Forschungsraum ist enger
Phase 17 untersucht Verbindungen auf festen quadratischen Gitterplätzen, standardmäßig 10 × 10 = 100 Orte. Die Orte bleiben fest, während Kanten umverdrahtet werden. Zusammenhang, Gradgrenzen, Kantenbudget, maximale Länge und Kreuzungsregeln werden geprüft. Vier allgemeine Suchstrategien arbeiten in diesem festgelegten Raum.

Seit Phase 17.1 können explizit konfigurierte Bindungen der Länge 2 über einen festen Gitterplatz hinwegführen. Dieser überquerte Ort wird dabei nicht elektrisch verbunden. Überlappende Strecken und unerlaubte Kreuzungen zwischen Gitterplätzen bleiben verboten. Die Option ist eine deklarierte Modellkonvention.

Lokale Mutationen kombinieren wenige Änderungen; explorative Mutationen versuchen größere Umbauten. Neu erzeugte Kandidaten durchlaufen erneut die Gültigkeits- und Duplikatprüfung. Deskriptoren messen den Anteil ersetzter regulärer Kanten und den Anteil langer Bindungen. Eine größere Mutation garantiert keinen besseren physikalischen Zustand.

Im Geometrie-Audit von Phase 17.1 wurden 1.100 gültige 100-Orte-Geometrien ohne exakte Physikrechnung erzeugt. Das belegt Erreichbarkeit unterschiedlicher Strukturen, keine physikalische Überlegenheit.

Quellen: geometry/generators/__init__.py; research/space.py; docs/decisions/phase_17_1_implementation.md.

[[PAGE]]
# 5 | Auswertung und Robustheit
### Spektrum und Zustände
Eigenenergien zeigen die erlaubten Energien des endlichen Modells. Zustandsdichten und lokale Zustandsdichten beschreiben deren Verteilung. Ortsgewichte, Randgewichte und Lokalisierungsmaße zeigen, wo sich ein Zustand befindet. Finite-Size-Splitting untersucht die Aufspaltung bei veränderter Systemgröße, sofern eine passende Modellfamilie definiert ist.

Majorana-Diagnostik umfasst unter anderem Polarisation, Selbstkonjugation und räumliche Gewichte. Diese Größen helfen beim Untersuchen von Kandidaten. Eine kleine Energie allein beweist keinen geschützten Majorana-Zustand.

### Topologische Analyse
- Symmetrien: numerische Prüfung von Teilchen-Loch-, Zeitumkehr- und chiraler Symmetrie sowie Altland-Zirnbauer-Klassifikation.
- Pfaffian-Invariante: eindimensionale topologische Auswertung in dafür geeigneten Systemen.
- Realraum-Winding: Diagnose chiraler eindimensionaler Systeme unter ihren jeweiligen Voraussetzungen.
- Bott-Index und lokaler Chern-Marker: realraumbasierte topologische Information für passende zweidimensionale Systeme.
- Spectral Localizer: verbindet Hamiltonmatrix und Ortsinformation, um lokale Indizes und eine zugehörige Localizer-Lücke zu bestimmen.

Der Dispatcher berücksichtigt die Anwendbarkeit. Nicht verfügbare oder numerisch ungültige Ergebnisse müssen als solche erhalten bleiben; sie sind nicht automatisch „topologisch trivial“.

### Was bei Störungen untersucht werden kann
Die Robustheitsbibliothek unterstützt Onsite-, Hopping- und Paarungsunordnung, zufälliges Entfernen von Kanten oder Knoten, Koordinatenstörungen und Parameteränderungen. Gesäte Zufallszahlen erlauben reproduzierbare Ensembles und vergleichbare Störungsrealisierungen.

Aus Ensemble-Ergebnissen entstehen Erfolgsanteile, Robustheitsmaße, statistische Unsicherheiten und Berichte. Die allgemeine Bibliothek enthält außerdem Werkzeuge für Größenfamilien und Finite-Size-Auswertung. Der aktuelle Fixed-Site-Forschungsadapter stellt jedoch keine kandidatenerhaltende Größenfortsetzung bereit.

### Einheitliche Bewertung
evaluate_geometry verbindet Geometrie, Modelladapter, Spektrum, Zustände, Topologie und Diagnostik zu einem Ergebnis. Gültigkeitsberichte, skalare oder mehrdimensionale Ziele und Reproduzierbarkeitsinformationen machen Kandidaten vergleichbar. Aussagekraft und Voraussetzungen der einzelnen Größen bleiben dabei entscheidend.

Quellen: evaluation/; observables/; topology/; robustness/ unter src/toposc_lab; docs/research_workbench.md.

[[PAGE]]
# 6 | Daten, maschinelles Lernen und Suche
### Datensätze statt unbeschrifteter Ergebnisdateien
Die Datenschicht speichert wissenschaftliche Resultate mit Schema, Geometrie, Modellparametern und Herkunft. Sie enthält Validierung, Migration, Duplikatprüfung und Aufteilung in Trainings-, Validierungs- und Testsätze. Studien speichern Messreihen und ihre Metadaten für spätere Vergleiche.

### Wozu maschinelles Lernen dient
Exakte Rechnungen sind teurer als eine Vorhersage aus Graphmerkmalen. Surrogatmodelle schätzen daher, welche Kandidaten als Nächstes exakt gerechnet werden sollten. Merkmale stammen aus Geometrie und ausdrücklich angegebenen Modellparametern. Bereits berechnete Zielgrößen dürfen nicht unbemerkt als Eingabemerkmale dienen.

Vorhanden sind einfache Regressions- und Klassifikationsverfahren, Bootstrap-Ensembles für Unsicherheit, Intervallkalibrierung, Erkennung großer Abstände zum Trainingsbereich und eine kleine GNN-Referenz. Benchmarks vergleichen einfache Modelle und GNN auf denselben Datensätzen. Ein GNN ist kein automatisch überlegener Standard.

Ein OOD-Hinweis bedeutet: Der Kandidat liegt außerhalb oder weit vom bisher gelernten Bereich. Das ist ein Warnsignal für die Vorhersage, kein Beweis ungültiger Physik. Nur die anschließende exakte Rechnung kann wissenschaftliche Evidenz liefern.

### Mehrere Suchgenerationen sind implementiert
- Zufallssuche: gültige Geometrien und Modellparameter ziehen, auswerten, speichern und sortieren; Baselines und Top-Kandidaten darstellen.
- Evolutionäre Suche: Population, Fitness, Auswahl, Elitismus, Mutationen, eingeschränktes Crossover, Diversität, Neuheit und Generationenablauf. Checkpoints erlauben Wiederaufnahme.
- Generative und konstruktive Ansätze: gültige Strukturen gezielt vorschlagen, mit vergleichbaren exakten Budgets bewerten und gegen Referenzen prüfen.
- Discovery Engine: Vorschläge, Vorhersage, exakte Bestätigung, Unordnungsprüfungen und Berichte in einer persistenten Kampagne verbinden.
- Musteranalyse und Interventionen: strukturelle Zusammenhänge untersuchen und durch gezielte Änderungen beziehungsweise unabhängige Prüfungen hinterfragen.

### Was die neueste Suche verändert
Phase 17.2 kann Eltern häufiger aus dem exakt besten Archivanteil auswählen. Die Option batch_novelty berücksichtigt bereits im selben Batch ausgewählte Kandidaten. Kompakte Bitmasken beschleunigen symmetriebezogene Distanzprüfungen; kleinere Checkpoints und sparsamere Dashboard-Abfragen reduzieren Verwaltungsaufwand. Eine bessere wissenschaftliche Trefferquote ist damit noch nicht nachgewiesen.

Quellen: docs/phase_10_usage_de.md; phase_11_dataset_usage_de.md; phase_12_ml_usage_de.md; phase_15_usage_de.md; decisions/phase_17_2_interrupted_run_review.md.

[[PAGE]]
# 7 | So arbeitet Autonomous Research
### Ein nachvollziehbarer Suchkreislauf
1. Ein neues Experiment speichert Konfiguration, Seed, physikalisches Protokoll, Grenzen und Quellherkunft.
2. Random, Evolution, MAP-Elites oder Surrogate MAP-Elites erzeugt Kandidaten. Gültigkeit und Ähnlichkeit werden vor teuren Rechnungen geprüft.
3. Die Auswahl kombiniert je nach Konfiguration hohe Vorhersagewerte, Unsicherheit und strukturelle Neuheit.
4. Der exakte Adapter berechnet sauberes System, angeforderte Bestätigung und einzelne Unordnungsrealisierungen.
5. Vollständige gültige Ergebnisse erhalten Zielwerte. Ein Qualitäts-Diversitäts-Archiv bewahrt gute Kandidaten in unterschiedlichen Deskriptorbereichen auf.
6. Neue exakte Daten verbessern die Priorisierung. SQLite, Journale und Checkpoints sichern den Ablauf; die Oberfläche zeigt gespeicherte Ergebnisse.

### Was hier physikalisch gerechnet wird
Der Adapter phase17.fixed-sites-chiral-p-wave.v1 verwendet ChiralPWaveModel mit Hopping 1, chemischem Potential 2, Paarung 1 und Chiralität +1. Offene Grenzen und ein Probeort im Zentrum sind festgelegt. Eine Exakt-Stufe umfasst das BdG-Spektrum und Localizer-Rechnungen bei drei Skalen: 0,1; 0,2; 0,3.

Die Qualität ist die kleinste Localizer-Lücke, sofern die vorgesehenen Index-, Invertierbarkeits- und Symmetriebedingungen erfüllt sind; andernfalls ist sie null. Das eingefrorene Erfolgskriterium lautet Qualität mindestens 0,20. Onsite-Unordnung wird gleichverteilt zwischen -W/2 und W/2 gezogen.

Wählbare Ziele sind beispielsweise clean_quality, robustness_success_fraction und robustness_quality_mean. Experiment 002 verwendet den kontinuierlichen Qualitätsmittelwert. Rohdaten und Ergebnisse je Störungsbreite bleiben zusätzlich erhalten.

### Budget, Steuerung und Wiederherstellung
Jede saubere Rechnung, Bestätigung und einzelne Störungsrealisierung kostet einen Exakt-Versuch. Beim Standard mit drei Breiten und je vier Seeds sind das 2 + 3 × 4 = 14 Stufen pro vollständig geprüftem Kandidaten. Fehlversuche und Wiederholungen beanspruchen ebenfalls Budget.

Pause und Safe Stop greifen zwischen Stufen; eine bereits laufende Stufe endet zunächst. Ein abgetrennter Worker kann nach dem Schließen der Oberfläche weiterlaufen. Wiederaufnahme prüft Quelle und Herkunft streng. Ein alter Lauf darf deshalb nicht einfach mit geändertem Programmcode fortgesetzt werden.

Dashboard, MAP-Elites-Ansicht, Suchfortschritt, Kandidatenexplorer, Baselines, Checkpoints und Abschlussbericht machen die einzelnen Schritte sichtbar. Der Adapter unterstützt einen exakten Worker pro Experiment; das Zeitlimit ist kooperativ.

Quelle: docs/research_workbench.md; src/toposc_lab/research/.

[[PAGE]]
# 8 | Die interaktiven Lernlabore
### Quantengase
Das Gaslabor vergleicht klassische Maxwell-Boltzmann- und Bose-Einstein-Statistik unter gemeinsamen äußeren Bedingungen. Bei fester Teilchenzahl bestimmt es das chemische Potential aus der Teilchenzahlgleichung. Besetzungen von Impulszuständen und der ideale dreidimensionale Bose-Kondensationsübergang werden sichtbar.

Der Bereich Ensembles and dynamics trennt kanonische, großkanonische und mikrokanonische Fragestellungen. Für das klassische Gas gibt es unter anderem Teilchenzahlfluktuationen und eine ballistische Bewegungsdarstellung. Für das Bose-Gas stehen feste Teilchenzahl, Normalstatistik bei festem chemischem Potential und eine exakte kleine Fock-Zustandszählung bereit.

Die mikrokanonische Quantenzählung ist bewusst auf einen kleinen eindimensionalen Modensatz begrenzt. Sie ist kein allgemeiner Vielteilchensolver für große wechselwirkende Systeme.

### Landau-Niveaus
Das Landau-Labor erklärt die quantisierten Energien geladener Teilchen im Magnetfeld. Einstellbar sind unter anderem Magnet- und elektrisches Feld, effektive Masse, Probengröße, Quantenzahlen und g-Faktor.

- Gleichmäßig gestaffeltes Landau-Spektrum und optionale Zeeman-Aufspaltung.
- Wellenfunktionen in Landau-Eichung und ihre Führungszentren.
- Ringzustände des niedrigsten Landau-Niveaus in symmetrischer Eichung.
- Elektrische Verschiebung, Zyklotronbewegung und E-kreuz-B-Drift.
- Browseranimation mit kohärenten Wellenpaketen, stationären Dichten, Strompfeilen und Niveaubesetzungen.
- Neunstufiges Tutorial mit vorbereitetem Experiment, Beobachtungsauftrag und Erklärung.

### Ganzzahliger Quanten-Hall-Effekt
Das zugehörige Labor zeigt Füllfaktor, temperaturabhängige Besetzung, verbreiterte Zustandsdichte, Hall-Plateaus und longitudinale Übergangsmaxima gemeinsam. Spinloser, unaufgelöst spinentarteter und Zeeman-aufgelöster Modus sind unterscheidbar.

Ein Randzustandsbereich ergänzt glattes Einschlusspotential, Energiedispersion, Fermi-Kreuzungen und Gruppengeschwindigkeiten. Eine Skipping-Orbit-Animation zeigt das semiklassische Bild; Kanalzahl und Hall-Spannung bestimmen den angezeigten quantisierten Randstrom.

Plateau-Breite, Gauß-Verbreiterung und longitudinale Maxima sind in dieser Version phänomenologische Einstellungen. Sie ersetzen keine mikroskopische Unordnungs- und Lokalisierungsrechnung. Statische Abbildungen können als hochauflösende PNG beziehungsweise Vektor-PDF und Parameter als JSON exportiert werden.

Quellen: README.md; src/toposc_lab/app/landau_level_lab.py; integer_quantum_hall_lab.py; gases/; bosons/.

[[PAGE]]
# 9 | Was bereits nachweisbar funktioniert
### Prüfung am Berichtsdatum
Am 25.09.2026 wurden vier ausgewählte Testmodule auf dem aktuellen Quellstand ausgeführt: Modellregistrierung, CLI, erweiterte Forschungskonnektivität und die Phase-17.2-Suchverbesserungen. Ergebnis: 42 Tests bestanden in 9,37 Sekunden. Dies ist eine aktuelle Funktionsstichprobe, keine neue vollständige Regression und keine manuelle Prüfung sämtlicher Fenster.

### Dokumentierte größere Prüfungen
- Phase 17.1: 3.099 Tests bestanden in 949,85 Sekunden im damaligen abschließenden Gesamtprüflauf. Diese Zahl bezieht sich auf den damaligen Stand.
- Phase 17.2: 140 fokussierte Tests bestanden in 141,13 Sekunden. Dokumentiert sind außerdem Ruff und eine begrenzte Mypy-Prüfung der Forschungsmodule.
- Phase-17.2-Wiederherstellungstest: Pause, harter Prozessabbruch, Neustart und Wiederaufnahme bestanden. Wiederhergestellte und ununterbrochene wissenschaftliche Abläufe waren identisch; das erneute Öffnen eines fertigen Ablaufs verbrauchte keine zusätzlichen Exakt-Aufrufe.

### Experiment 002: echte gespeicherte Ergebnisse
Laut Audit vom 17.09.2026 wurden 254 Kandidaten vollständig ausgewertet: 252 Suchkandidaten und zwei Baselines. Alle 4.572 verbuchten Exakt-Stufen waren vollständig. Der Lauf wurde bei Zyklus 42 unterbrochen; ein älterer exportierter RUNNING-Status war kein Beleg für einen noch laufenden Prozess. Für diesen Bericht wurde der historische Audit ausgewertet, kein neuer Live-Prozessstatus ermittelt.

Der beste gefundene mittlere Qualitätswert war 0,390580 gegenüber 0,376689 für das reguläre Gitter: etwa 3,687 % höher. Zugleich sank das saubere Randgewicht von 0,997067 auf 0,819853. Es gibt damit einen Zielkonflikt, keinen pauschalen physikalischen Sieger.

148 von 252 Suchkandidaten und beide Baselines bestanden sämtliche angefragten Unordnungstests. Ein Vergleich von 100 % zu 100 % trennt deren Robustheit hier nicht weiter. Vier verwendete Seeds und die adaptive Auswahl reichen nicht für eine unabhängige Bestätigung allgemeiner Überlegenheit.

### Weitere Forschungsergebnisse und Leistungswerte
Die unabhängige Phase-16B-Prüfung wurde abgeschlossen, hat aber keinen reproduzierbaren geometrischen Mechanismus etabliert: Gate 16 lautet NOT_ESTABLISHED. Mehrere vorgegebene Vergleiche hatten zu wenige streng passende Elterngeometrien.

Der Phase-17.2-Benchmark beschleunigte acht identische Vorschläge von 17,540 auf 0,546 Sekunden; ein Wiederholungslauf ergab 26,5-fache Beschleunigung. Das betrifft ausschließlich diesen begrenzten Proposal-Test, nicht das ganze Programm. Das neue Pilotbeispiel Experiment 003 ist im Bericht vorbereitet, aber nicht gestartet.

Quellen: decisions/phase_17_2_verification.json; phase_17_2_interrupted_run_review.md; phase_17_1_implementation.md; phase_16b_validation.md in docs/.

[[PAGE]]
# 10 | Grenzen, Sicherung und Quellen
### Was noch nicht als gesichert gelten darf
- Die aktuelle Forschungswerkbank zertifiziert keine separierten oder chiralen Majorana-Moden. Ihre Majorana-Werte bleiben Diagnostik.
- Endliche Center-Probe-Localizer-Ergebnisse beweisen keine thermodynamische Phase. Eine Localizer-Lücke ist nicht gleichbedeutend mit einer Bulk-Spektrallücke.
- Eine kritische Unordnungsstärke W_c und Größenkonvergenz sind für die aktuellen Fixed-Site-Kandidaten nicht nachgewiesen.
- Die neue Suchpolitik hat noch keinen kontrolliert bestätigten Vorteil bei der wissenschaftlichen Trefferquote.
- Entfernungsabhängiges Hopping und entfernungsabhängige Paarungsstärken gehören nicht zum gegenwärtigen Phase-17-Adapter.
- Streumatrix-Invarianten wurden zurückgestellt: eine vollständige Infrastruktur für offene Systeme mit Kontakten und Transport fehlt.
- Nicht jedes Bibliotheksmodul ist als GUI-Funktion verfügbar. Die native Anwendung ist keine universelle Fernsteuerung aller Modelle oder entfernter Rechencluster.

### Was der Git-Commit sichert
Der Snapshot 37c9279 markiert den bereits sauberen Programmstand. Der fachliche Stand stammt aus eb7b3c3, der Phase 17.2 enthält. Dieser Bericht und die PDF werden anschließend separat committed. Es wurde kein Push ausgeführt.

Der Ordner results/ ist laut .gitignore von Git ausgeschlossen. Ein Quellcode-Commit sichert daher nicht die vollständigen lokalen Experimentdaten, Datenbanken, Logs oder Quellenarchive in diesem Ordner. Für eine vollständige Datensicherung muss results/ zusätzlich kopiert beziehungsweise gesichert werden. Die virtuelle Python-Umgebung ist ebenfalls nicht Bestandteil des Commits.

### Lokale Quellen für die weitere Arbeit
Die Angaben stammen aus dem lokalen Repository. Die Quellen stehen jeweils am Ende der Fachseiten. Wichtigste Einstiege sind docs/research_workbench.md, die Modellregistrierung und die Phase-17.2-Berichte unter docs/decisions/.

Der Phase-17.1-Bericht beschreibt die Vorbereitung vor dem Start von Experiment 002; für dessen späteren Stand gilt Phase 17.2. Auch die README enthält ältere Kurzbeschreibungen. Bei Abweichungen wurden aktuelle Implementierung und neuere Entscheidungsberichte herangezogen.
