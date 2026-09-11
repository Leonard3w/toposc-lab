# Konstruktives Geometrielernen: Architektur und Entscheidungsentwurf v1

Stand: 2026-09-11, Ausgangsrevision
`dadd0da60f0ce9aeb96392cc7831045029e7ebc6`.

## Entscheidung und Evidenzgrenze

Der Nutzer hat nach der lokalen Enumeration ausdrücklich einen modularen
neuronalen Graphbauer mit freien Punktpositionen, freien Verbindungen und
Liveverfolgung beauftragt. Das erweitert den zuvor gesetzten Implementierungsumfang
über Phase-10-Evolution hinaus. Es ist keine nachträgliche Änderung der abgeschlossenen
Experimente und keine Freigabe neuer Physik- oder Forschungstrainingsläufe.
Masterplan: Phasen 10–12; die [Charter](pre_phase_9_research_charter.md) bleibt maßgeblich.

Implementiert ist eine wiederverwendbare konstruktive RL-Infrastruktur und eine
explizite geometrische Engineering-Demo. Nicht implementiert ist ein wissenschaftlich
validierter Supraleitungs-Evaluator für freie Positionen. Das ist eine wesentliche
offene Forschungsaufgabe, keine umbenennbare Variante des Quadrat-Evaluators.

Die abgeschlossene lokale Enumeration bleibt unverändert unter
`results/phase_10_local_enumeration_v1_progress_fix1/full/`:
268 Slots, 256 Kandidaten, 12 Kontrollbewertungen; Screening 112/128 bzw. 128/128;
keine primären, qualifizierten oder gepaarten Treffer. Summary-SHA-256:
`16dc0a0b4e43d1ef27278fca08f3bce77b96ca48b5258d681f98ea48207594ec`.
Dieser Nullbefund betrifft den alten lokalen Raum und erlaubt keine Vorhersage
der Erfolgswahrscheinlichkeit neuronaler Suche. Frühere subjektive Prozentwerte
werden nicht als Evidenz verwendet.

## Physikalische Hypothese und ihre Grenzen

Zu prüfen wäre, ob gemeinsam variierte räumliche Anordnung und Konnektivität
einen reproduzierbaren Schutz-/Robustheitsvorteil ermöglichen, der im engen
Einzeleingriffskatalog fehlt. Das ist eine neue Arbeitshypothese, kein Befund.
Aus der Nichtlinearität einer Bewertung folgt kein positiver Kombinationseffekt.
RL ist eine mögliche Suchmethode, keine zusätzliche physikalische Begründung.

Der alte 16×16-Katalog bleibt Entwicklungsmaterial. Ein späterer neuer Kandidat
auf 20×20 ist nicht allein wegen seiner Größe ein unberührter wissenschaftlicher
Holdout. Auswahl, Größenfortsetzung und unabhängige Disorderbestätigung müssen
vor deren Auswertung separat eingefroren werden.

## Modellwahl

V1 verwendet einen lokal trainierten, gemeinsam auf alle zulässigen Aktionen
angewendeten MLP: zwölf explizite geometrische Eingaben, 16 tanh-Neuronen,
ein Logit pro Aktion, Softmax. Insgesamt 224 trainierbare Parameter. Er ist
kein Sprachmodell, kein Graphnetz mit Message Passing und kein Physiksurrogat.
NumPy ist vorhanden; keine neue ML-Abhängigkeit, Cloud oder Zugangsdaten nötig.

Freie Positionen entstehen durch Auswahl aus je 32 frisch kontinuierlich
gleichverteilt vorgeschlagenen Punkten. Es gibt kein festes Gitter. Das Netz
erzeugt aber auch keine beliebigen Koordinaten direkt: die endliche Vorschlagsmenge
beschränkt jede einzelne Entscheidung. Diese Suchverteilung ist ausdrücklich Teil
des Vertrags. Punkt- und Kantenaktionen teilen zunächst einen Actor mit Stufenmerkmal.

REINFORCE summiert die Gradienten der Log-Wahrscheinlichkeiten über alle Aktionen
einer Episode. Die abschließende Belohnung minus einer Baseline aus ausschließlich
früheren Episoden gewichtet diese Summe. Lernrate 0.01, Baseline-Update
`b <- 0.95*b + 0.05*reward`, globale Gradientennorm auf 5 begrenzt.
Das Clipping ist eine Stabilisierung, kein unverzerrter Gradientenschätzer mehr.
Fehlende Bewertung: keine Gewichts- oder Baselineänderung. Einfache Hypothese,
kleines Modell und endliche Gradientenprüfung sind hier wichtiger als viele
zusätzliche Lernkomponenten. Kein PPO-, GNN- oder Sample-Effizienz-Superlativ.

Grundlagen:

- [Williams 1992: REINFORCE](https://doi.org/10.1007/BF00992696).
- [Bello et al.: Neural Combinatorial Optimization](https://arxiv.org/abs/1611.09940).
- [Kool et al.: Attention, Learn to Solve Routing Problems](https://arxiv.org/abs/1803.08475).
- [Huang/Ontañón: Invalid Action Masking](https://arxiv.org/abs/2006.14171).

Diese Arbeiten stützen die Methodenwahl; sie beweisen keinen Vorteil für dieses
Class-D-Modell. Eine perfekte universelle Vorgehensweise existiert hier nicht.

## Bausteine und Erweiterungspunkte

| Baustein | Verantwortung | Austausch/Verwendung |
| --- | --- | --- |
| `BuildRules` | Unveränderliche Ressourcen und geometrische Grenzen | Freie Punkte oder explizite feste Punkte, kleine einfache Aufgaben |
| `BuildState` | Punkte, Kanten, Abbruchgrund | Physikfreier Zustand; `geometry()` erzeugt vorhandenes `Geometry` |
| `GraphBuilder` | Vorschläge, Masken, Übergänge, Aktionsmerkmale, Abschlussprüfung | Eigene versionierte Unterklasse über `builder=` injizieren |
| `Policy` | Aktion auswählen, terminales Lernen, Snapshot | Zufall oder neuronaler Actor; später Graphnetz |
| `Evaluator` | Rohergebnisse, Status, explizite Belohnung | Geometrische Aufgaben oder separat geprüfter Physikadapter |
| `run_training` | Endliches Budget, Seeds, Episodenarchiv, Wiederaufnahme | Keine UI- oder Modellphysik im Trainingsloop |
| `viewer` | Lesen von Livezustand und versiegeltem Replay | Separater lokaler Prozess ohne Trainingsaktionen |

Neue externe Builder/Evaluatoren müssen bei semantischer Änderung ihre Identifier
ändern. Der Codehash umfasst die drei eingebauten Kernmodule; externe Callbacks
werden nicht automatisch gehasht. Ein Identifier ist kein Beweis korrekten Verhaltens.
Eigene Policytypen benötigen zusätzlich einen kompatiblen Checkpoint-Decoder in
`restore_policy`; das aktuelle Resume unterstützt die beiden eingebauten Typen.

`GeometryGenome.from_geometry(state.geometry())` nutzt die bestehende Genomschicht.
Boundary-Sites werden dabei **nicht** aus Knotengraden erfunden. Der Endgraph enthält
zunächst keine als physikalisch validierte Randdefinition. Hopping, Pairing, Fitness
und Suchhistorie landen nicht in Geometrie-Metadaten.

Die alten Einzelkantenmutationen erhalten weder Gradfolge noch Planarität;
vorhandenes Kanten-Crossover erhält nicht automatisch Ressourcen. Der lokale
Katalogvalidator fordert genau einen Zwei-Kanten-Eingriff und wird nicht erweitert.
`validate_measurement_geometry` fordert unter anderem Quadratanzahl, Quadrat-Randsites
und neutrale Metadaten. Der vorhandene zentrale Messmasken-Builder ist für freie
Positionen kein unmittelbar verwendbarer Vertrag. Keiner dieser alten Verträge
wird stillschweigend gelockert.

## Geometrische Regeln und Machbarkeit

Standarddemo: 12 Punkte, 16 Kanten, Rechteck 4×4, Abstand mindestens 0.4,
Kantenlänge höchstens 2, Grad 1–4. Das sind technische Demoentscheidungen,
keine physikalisch kalibrierten Forschungsparameter.

Zuerst Punkte, dann Kanten; Start ohne Kanten. Mit festen Punkten entfällt die
Punktphase. Kein verstecktes Quadrat, Startbaum, Reparieren oder Nachziehen.
Keine Selbst-/Doppelkanten, geraden Kreuzungen oder Kanten durch einen dritten
Punkt. Zusammenhang und untere Gradgrenze gelten für das fertige Objekt.
Obere Gradgrenze und Kreuzungsverbot gelten bei jeder gesetzten Kante.

Notwendige Vorausschau: verbleibendes Budget muss Komponenten verbinden und
Graddefizite abdecken können; genügend zulässige Restkanten müssen existieren;
die Vereinigung aller noch möglichen Kanten muss den Graphen verbinden können.
Diese Checks sind kein vollständiger Fertigstellbarkeitsbeweis. Sackgassen und
erschöpfte Punktvorschlagsmengen bleiben gezählte fehlgeschlagene Episoden.

Längen-/Kreuzungsinventar wird pro Punktanordnung wiederverwendet. Dennoch ist
die aktuelle Vorausschau bei großen Kandidatenmengen teuer. Skalierbarkeit auf
256 freie Punkte ist nicht nachgewiesen. Ein Geometrieprofiling geht jedem großen
Training voraus; die kleine Standarddemo ist kein heimliches Forschungsscreening.

Ein festes Rechteck kontrolliert nur die erlaubte Fläche, nicht die tatsächlich
besetzte Ausdehnung oder Dichte. Insbesondere kann das Demo-Ziel kurze Kanten
räumliches Zusammenziehen begünstigen. Für Forschung fehlen unter anderem feste
Belegungs-/Ausdehnungsregeln, Rand-/Bulk-Exposition, Kopplungslängenbudget und
ressourcenangepasste Referenzen. Diese Unterschiede dürfen nicht als Geometrievorteil
umgedeutet werden. Die feste Gradobergrenze bleibt ebenfalls eine bewusste Einschränkung.

## Bewertung, Duplikate und Fehler

Die ausgelieferte `CompactWiringDemo` liefert
`reward = 1 - mean_edge_length / maximum_edge_length` für gültige Endgraphen.
Sie testet Lernen an einer billigen, nachvollziehbaren Aufgabe. Keine Hamiltonians,
Localizer, topologischen Treffer oder Disorderrechnungen. Geometrische Ablehnung
liefert −1 als ausdrücklich technische Baubewertung, nicht als L-Wert.

Ein Physikadapter muss Raw Quantities, numerische Gültigkeit, Topologiegates,
Proxy-Definition, Randdiagnostik und Belohnungsabbildung gemeinsam versionieren.
Die globale 1-%-Trefferregel des alten Experiments ist kein universelles
Reward-Rezept. Alle alten Kandidaten-/Zusatzmaskenlabels bleiben erhalten.

Evaluator-Exceptions liefern `reward=None` mit Fehlertyp/-text; sie bleiben im
Nenner und führen zu keinem RL-Update. Interrupts und Speicherfehler stoppen.
Ein wissenschaftlich verworfener Kandidat ist vom operativen Fehler zu trennen.
Ein künftiger physikalischer Evaluator darf nicht von Beobachtern abhängen.

Jede begonnene Episode gehört zum festen Budget, einschließlich Sackgassen,
Duplikaten und Fehlern. Pro abgeschlossenem Endgraph höchstens ein Evaluatoraufruf.
Cache nur für gleiche Koordinatenreihenfolge, sortierte Kanten, Evaluator-ID und
gegebenenfalls Evaluationsseed. Ausschließlich bei explizitem
`deterministic=True` wird der Seed weggelassen. Kein Caching von Fehlern,
keine behauptete Isomorphie-/Spiegelungsäquivalenz. Erneute Nutzung eines bekannten
Rewards kann das Netz aktualisieren, zählt aber nicht als unabhängiges Datum.

## Seeds, Vergleich und unabhängige Bestätigung

Pro Episode entstehen aus `(root_seed, episode_index)` getrennte PCG64-Ströme
für Punktvorschläge, Aktionsauswahl und Evaluationsseed. Netzwerkinitialisierung
ist explizit; Beobachter erhalten nur Kopien und keinen RNG. Für stochastische
Evaluatoren gehört der Evaluationsseed zum Cache-Key.

Zufall zieht gleichverteilt aus den **aktuell zulässigen Aktionen**, nicht aus
allen gültigen Endgraphen. Beide Policies verwenden denselben Builder. Dieselben
Root-Seeds definieren eine Paarung, garantieren nach divergierenden Entscheidungen
aber keine identischen späteren Geometrien. Eine Suchüberlegenheit braucht
mehrere unabhängige Trainingsläufe, gleiche Vorschlags-/Evaluationsbudgets,
separate Cache-/Fehlerzahlen sowie CPU-Zeit einschließlich Training und Abstimmung.
Eine einzelne Demo oder fallende Verbindungslänge ist keine Überlegenheitsstudie.

Vor Forschung noch einzufrieren: Modell-/Messvertrag, Referenzfamilien, konkrete
Budgets, Entwicklungs- und Holdout-Seedlisten, Trainings-/Abstimmungsbudget,
Wahl der besten Kandidaten, Abbruchregeln und Unsicherheitsauswertung. Nominierte
Kandidaten und Netzgewichte werden vor unabhängiger Bestätigung versiegelt.
Disorder-Seeds, neue Größen und Ablationen dürfen nicht ins Training zurückfließen.
Evolution erfordert einen passenden Operator im gleichen Endraum; der alte lokale
Operator ist keine faire Baseline für freie Punktplatzierung.

## Speicherung, Liveverfolgung und Wiederaufnahme

Ein neuer Lauf verlangt ein neues Outputverzeichnis. `manifest.json` bindet Regeln,
Budget, Seeds, initiale Policy, Evaluator, Kern-Codehash, Python-/NumPyversion und
Threadwerte. Jede versiegelte Episode enthält den vollständigen Netzstand,
Endgraph, Bewertung und Hash des vorigen Checkpoints und ihres Ereignisprotokolls.
Getrennte Ausführungsordner erhalten abgebrochene Versuche. Nur das abgeleitete
`live.json` wird während eines aktiven Laufs ersetzt.

Resume validiert das zusammenhängende Archivpräfix und Tracehashes und startet
die erste unversiegelte Episode mit identischen Seeds in einer neuen Ausführung.
Es erweitert das Budget nicht. Bereits komplette Archive werden nur gelesen und
geprüft. Technische Wiederholungen sind Zusatzaufwand: Evaluator-Start/Ende und
Ausführungsnummern bleiben in den Traces sichtbar. Ein Prozessabsturz zwischen
Start/Ende lässt die tatsächliche Rechenarbeit ausdrücklich unvollständig bekannt.
Es gibt noch keinen generationsübergreifenden Forschungsbudgetbericht.

Eine exklusive Writer-Lock verhindert gleichzeitige Autoren. Nach Ctrl+C wird
sie entfernt. Nach einem harten Prozessabbruch kann sie stehenbleiben; vor
manueller Entfernung muss ausgeschlossen sein, dass der Schreiber noch läuft.
Unterbrochene `.partial`-Dateien sind keine Checkpoints und bleiben erhalten.

Der Viewer bindet nur `127.0.0.1`, lädt keine externen Skripte und erlaubt nur
Livezustand oder versiegeltes Replay. Pause pausiert die **Anzeige**, nicht den
Lerner. Live wird viermal pro Sekunde gelesen; schnelle Schritte können übersprungen
werden. Replay zeigt alle gespeicherten Schritte, ohne den Lerner künstlich zu bremsen.
Die Ansicht zeigt tatsächlich gesetzte Punkte/Kanten, keine angeblich optimale
Geometrie. Evaluator- und Bewertungsstatus bleiben sichtbar.

## Abnahme und nächster Schritt

### Dreiarmiger Engineering-Pilot nach den ersten zwei Einzelläufen

Die zwei betrachteten 100-Episoden-Läufe motivieren eine Kontrolle der Initialisierung:
96 gültige neuronale gegenüber 87 gültigen Zufallsgeometrien, aber kürzere mittlere
Kanten bei gültigen Zufallsgraphen. Daraus folgt kein bewiesener Lerneffekt.
Der Nutzer beauftragte daraufhin einen Vergleich mit einer eingefrorenen Policy.

`python -m toposc_lab.design.benchmark` friert vor Ausführung zehn neue Seeds
20261001–20261010, je 100 Episoden pro Arm und identische Standardregeln ein.
3.000 Episoden insgesamt, ohne physikalische Bewertung und ohne automatischen Start.
Die lernende und eingefrorene Policy erhalten pro Seed exakt gleiche Anfangsgewichte,
Aktionswahrscheinlichkeiten und Seedpläne. Die eingefrorene Policy führt keine
Gewichts-, Baseline- oder Gradientenupdates aus; sie bleibt eine stochastisch ziehende
Policy, nicht eine deterministische Greedy-Auswahl. Zufall bildet den dritten Arm.
Die pro Seed rotierende Armreihenfolge ist vorab festgelegt.

Die primäre deskriptive Lernkontrolle ist der seedweise Unterschied der mittleren
Episodenbelohnung neural minus frozen; neural minus random wird separat berichtet.
Gewinne/Gleichstände/Verluste verwenden eine absolute Vergleichstoleranz 1e-12.
Operative fehlende Rewards machen den jeweiligen Mittelwert und das Paar unverfügbar;
der Bericht nennt verfügbare Paare, ohne einen ersetzten Nullscore. Keine p-Werte
aus korrelierten Episoden, kein automatisches Überlegenheitslabel oder Physikclaim.
Zehn Seeds sind ein operativ begrenzter Pilot, keine anhand einer Effektgröße
begründete Powerberechnung. Kein Tuning auf Pilot-Seeds und keine alten
Entwicklungsläufe als neue Replikationen. Laufende Trainingsleistung unterscheidet
sich von einer unabhängigen Evaluation der abschließend trainierten Policy.

Jeder Arm besitzt einen eigenen Rewardcache und ein vollständiges Archiv. Die
Suite speichert Protokoll-/Codebindung, verwendet eine exklusive Sperre, prüft
bei Resume auch fertige Arme und verändert vollständige Archive nicht. Für neue
Policytypen besteht nun ein expliziter Decoder für die eingefrorene MLP-Policy.
Eine neue Implementierung erhält eine neue Codebindung; alte Einzelarchive werden
für den Pilot weder fortgesetzt noch geändert. Der frühere reine Livefix bleibt
die einzige ausdrücklich zugelassene Altversionsausnahme.

Abnahme Dreiarm-Pilot: 42 fokussierte und 2.665 vollständige Tests bestanden,
Python 3.14 mit `-B`, Bytecode deaktiviert und allen vier Threadvariablen auf 1.
Strict mypy für die sieben Designmodule sauber; neues Ruff sauber,
projektweites Ruff unverändert 145 Altbefunde; `git diff --check` sauber.
Nur synthetische Testläufe ausgeführt, kein neuer Pilot-/Physiklauf gestartet.

Tests prüfen unter anderem Quadrat-Erreichbarkeit, Kreuzungen, Kanten durch Sites,
Abstände, Grad-/Zusammenhangsbedingungen, echte REINFORCE-Gradienten per finiten
Differenzen, synthetisches Lernen, Cache/Fehler, deterministisches Resume,
Archivkorruption und read-only HTTP/Replay. Keine neue Forschungsphysik erforderlich.
PowerShell-Anleitung: [Graphbauer benutzen](../constructive_design_usage_de.md).

Abnahme des Implementierungsstands: Python 3.14 mit `-B`,
`PYTHONDONTWRITEBYTECODE=1` und allen vier Threadvariablen auf 1;
33 fokussierte Tests und 2.656 Tests der vollständigen Suite bestanden.
Neues Ruff und relevantes strict mypy sauber; projektweites Ruff weiterhin
145 bekannte Altbefunde, keine neuen Befunde im Graphbauer. `git diff --check`
sauber. HTTP/Replay automatisiert geprüft; eine visuelle Browserabnahme steht
noch aus. Keine neuen Forschungsphysik- oder längeren Trainingsläufe gestartet,
nichts committed oder gepusht.

Windows-Livefix: Ein `PermissionError` beim Veröffentlichen der abgeleiteten
Liveanzeige überspringt dieses Anzeigeupdate statt den Lerner abzubrechen.
Andere I/O-Fehler und alle Archivschreibfehler bleiben strikt. Resume akzeptiert
zusätzlich ausschließlich den bekannten unveränderten Vorgänger-Kernhash
`2b4a4fdccd514989be5562534199fb4381909ac5d609946c73cb3c88df161aae`;
Originalmanifest und versiegelte Episoden werden nicht geändert. Neue Episoden
speichern ihren Ausführungscodehash. Abnahme: 2.659 Tests bestanden, darunter
36 Graphbauer-Tests; relevantes strict mypy und neues Ruff sauber,
projektweites Ruff unverändert 145 Altbefunde.

Für den nächsten Architektur-/Physikvertragsreview empfehle ich GPT-6 Astra mit
Reasoning `high`; das ist eine Arbeitsmodell-Empfehlung, kein Bestandteil des
lokalen Lernprogramms. Für die Empfehlung wurden die
[offiziellen Modellhinweise](https://developers.openai.com/api/docs/guides/latest-model)
geprüft. Forschungsläufe startet weiterhin ausschließlich der Nutzer.
