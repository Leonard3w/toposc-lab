# Größen- und Bulkregionenvergleich: Protokollentwurf v1

## Status und Frage

Protokoll-ID: `TOPOSC-P10-SIZE-001`. Entwurf vom 2026-09-09 auf Nutzerauftrag,
nach der [Auswertung der ersten Quadratkalibration](phase_10_calibration_v1_analysis.md).
Ausgangsrevision: `aee26ec825ff0b3055d8a6d8dd9a8cca76ea5106`.
Die numerischen Festlegungen unten sind neue, prüfbare Entscheidungen nach
Kenntnis der alten Daten. Ein separater Nutzer-Commit friert sie vor Umsetzung
und neuen Rechnungen ein. Protokoll- und späterer Code-Hash werden getrennt
archiviert. Ein Runner für diesen Versuch existiert noch nicht.

> Wie verändern sich die Chern-Bulkschätzungen bei größerer Systemgröße und
> verschiedenen vorab festgelegten Messregionen, unter schwacher Positionsunordnung
> und vergleichbarer Kantenaustauschintensität im Quadrat-Umfeld?

Dieser deskriptive Methodenversuch dient der
[Forschungscharter](pre_phase_9_research_charter.md). Er untersucht die
Anwendbarkeit unserer Diagnostik vor einer weiteren Geometriesuche.
Er liefert keine thermodynamische Bestätigung aus nur drei Größen, keine
Disorder-Robustheitsvalidierung und keinen Evolutionsvergleich.

## Größen und vergleichbare Eingriffe

Je Größe wird der offene `square(n_x=n, n_y=n, spacing=1.0)`-Generator verwendet.
Die physikalische Größe wächst mit `n`; Koordinaten werden nicht auf dieselbe
Box gestaucht. Das unveränderte Quadrat derselben Größe ist die Referenz.

| Größe n | Knoten | Kanten E=2n(n-1) | Randknoten | Versuchte Kantentausche K=ceil(E/28) |
| --- | ---: | ---: | ---: | ---: |
| 8 | 64 | 112 | 28 | 4 |
| 10 | 100 | 180 | 36 | 7 |
| 12 | 144 | 264 | 44 | 10 |

Die Intensität `K/E` ist etwa 3.57%, 3.89%, 3.79%. Sie ist nur näherungsweise
gleich und zählt versuchte Schritte, nicht dauerhaft verschiedene Kanten.
Berichtet werden auch `|E_final symmetric_difference E_square|/2`, dessen
Anteil am Kantenbudget, Gradfolge, Randanteil und Rand-/Innen-Endpunktklassen.

Jede Größe erhält 16 unabhängig erzeugte Trajektorien. Je Trajektorie gibt es
drei veränderte Fälle: Positionen allein `(a=0.05,k=0)`, Kanten allein
`(a=0,k=K)` und Kombination `(a=0.05,k=K)`. Das gemeinsame Quadrat wird pro Größe
einmal am Anfang und einmal am Ende ausgewertet. Es sind keine 16 unabhängigen
Nullkontrollen.

Die Stichproben werden für jede Größe neu konstruiert. Innerhalb einer Größe
sind die drei Eingriffe über dasselbe Positionsfeld und dieselbe Kantenfolge
gepaart. Zwischen Größen handelt es sich um eigenständige Realisierungen einer
festen Konstruktionsregel, nicht um denselben vergrößerten Kandidaten. An einen
alten Kandidaten Gitterringe anzuhängen könnte Randknoten mit Grad 4 auf Grad 5
oder höher bringen und würde zudem die Eingriffsdichte verdünnen.
Die neun nachträglich auffälligen alten Varianten werden nicht als Gewinnerpanel
übernommen. Damit bleibt die neue Stichprobe frei von dieser Kandidatenauswahl.

## Exakter Konstruktionsvertrag

Knotennummer `i=x*n+y`, Koordinaten `(x,y)`, Box `[0,n-1]²`. Alle ursprünglichen
Randknoten bleiben positionsfest; die `(n-2)²` inneren Knoten erhalten ein Feld
aus `Generator(PCG64(position_seed)).uniform(-1,1,size=((n-2)**2,2))`, in aufsteigender
Knotennummer. `X(a)=X0+a*U` wird einmal direkt in Float64 berechnet. Amplitude
ist pro Koordinatenkomponente in Gitterabstandseinheiten definiert.

Der Kantenpool enthält alle ursprünglichen Gitterpaare `i<j` mit maximalem
Koordinatenunterschied 1 je Achse. Beide Diagonalen jeder Elementarzelle gehören
zum Pool. Pro Schritt wird die vollständige lexikographisch geordnete Liste
zulässiger `(remove_source,remove_target,add_source,add_target)` auf `X0`
ermittelt. Auswahl: `Generator(PCG64(step_seed)).integers(n_legal)`.
Zulässigkeit: genau n² Knoten und E Kanten, Zusammenhang, Grad 2–4, einfache
ungerichtete Kanten, keine geraden Kreuzungen außerhalb gemeinsamer Endpunkte.
Keine Delaunay-Triangulation oder physikalisch motivierte Auswahl.

Nach jedem Schritt wird der ausgewählte Graph vollständig geprüft. Ist die
Liste leer, wird ein No-op mit seinem Seed festgehalten. Rücktausch und Duplikate
sind erlaubt. Die Kantenfolge wird unabhängig vom Positionsfeld gezogen und
auf beide Positionsvarianten angewandt. Jeder endgültige Fall wird vollständig
validiert: Mindestabstand 0.55, maximale Kantenlänge 1.75, Boxspannweite exakt
bis auf absolute Geometrietoleranz `1e-12`, Grad und Kreuzungsregeln wie oben.

Randdefinition: Koordinatenschale mit Dicke 0.875; sie muss exakt den `4n-4`
ursprünglichen Randknoten entsprechen. Eine äußere Randkomponente mit Index 0,
keine deklarierten Lochränder. Kanten kleiner-zu-größer orientiert und sortiert,
Displacements aus den endgültigen Koordinaten; keine Randübertritte. Einheitlicher
Typ `size_calibration_coupling`, leere Edge-Metadaten, keine Site-Typen, Faces oder
Baumstruktur, `dimension_records=()`. Geometriemetadaten ausschließlich
`generator="phase10_size_square_deformation_v1"`. Rollen und Seeds separat.

Das gesamte Geometrieinventar der angeforderten Stufe wird vor deren erster
Physikrechnung gespeichert und geprüft. Ungültige Konstruktion stoppt mit
Fehlernachweis; kein Ersatzdraw oder nachträgliches Reparieren. Ressourcenprüfung
wird für diese Größen gesondert implementiert. Die alte fest auf 64/112 ausgelegte
`CLEAN_PRIMARY`-Prüfung wird nicht für größere Geometrien umdefiniert.

## Physik, Messregionen und feste Auswertung

Modell wie zuvor: `ChiralPWaveModel`, Hopping 1, chemisches Potential 2,
Pairing 1, Chiralität +1, Achsen `(0,1)`, Referenzenergie 0; komponentenweise
Nambu-Anordnung, konstante Kantenamplituden, kein Distanzabfall. Vollständiger
Eigenwertlöser `numpy.linalg.eigh`, numerische Toleranz `1e-10`, 16 retained
Niedrigenergiezustände und bestehende PH-/Randdiagnostik. `evaluation_seed=None`.

Aus jeder Geometrie wird für denselben Hamiltonian folgendes Panel berechnet:

- Bott: Perioden `(0.95*n,0.95*n)`, `(n,n)`, `(1.05*n,1.05*n)`,
  Quantisierungstoleranz `1e-6`.
- Localizer: Zentrum `((n-1)/2,(n-1)/2)`, Energie 0, Kappa `(0.1,0.2,0.3)`.
  `L` bleibt das Minimum der drei Gaps. Konstantes Kappa bedeutet denselben
  Orts-Energie-Maßstab; dies untersucht keine umfassende Kappa-Konvergenz.
- Lokaler Chern-Marker: fünf benannte Masken, dieselben positiven Voronoi-Flächen
  aus der Clipzelle `[-0.5,n-0.5]²`, Flächensumme n² mit `rtol=0,atol=1e-10`.
  Basis-Koordinaten `tile(X,(2,1))`; Flächen und Masken in `np.unique`-Ortsordnung.

Die fünf Masken sind:

| Name | Definition in physikalischer Knotennummerierung |
| --- | --- |
| graph_depth_2 | kürzester Graphabstand zum tatsächlichen Rand mindestens 2 |
| graph_depth_3 | kürzester Graphabstand zum tatsächlichen Rand mindestens 3 |
| fixed_depth_2 | ursprünglicher Quadrat-Koordinatenabstand zum Rand mindestens 2 |
| fixed_depth_3 | ursprünglicher Quadrat-Koordinatenabstand zum Rand mindestens 3 |
| central_fraction | ursprünglicher Quadrat-Koordinatenabstand zum Rand mindestens n/4 |

Die letzten drei Masken werden einmal pro Größe als feste Knotenmengen definiert.
`central_fraction` enthält bei n=8,10,12 genau 16,16,36 Knoten; diese diskrete
Rasterung wird berichtet. Identische Masken (z. B. bei n=8) sind keine unabhängigen
Messungen. Eine leere Graphmaske bleibt nicht anwendbar und erhält keinen Ersatz.
Andere verfügbare Methoden werden trotzdem archiviert. Die technische Vorprüfung
verlangt für das geplante Panel verfügbare Masken; gegebenenfalls ist vor dem
Hauptlauf ein neuer Plan erforderlich.

Primäre deskriptive Größe ist `abs(C_central_fraction - 1)` je Variante und
Größe. Berichtet werden alle 16 Werte, Median, Minimum, Maximum und verfügbares n.
Zusätzlich die gepaarte Differenz zum Quadrat derselben Größe sowie die beiden
Unterschiede Kombination-minus-Positionen und Kombination-minus-Kanten.
Alle fünf Chern-Schätzungen, Maskengrößen, Flächen, Indexwerte, Residuen und
Warnungen bleiben einzeln sichtbar. Die Grenze `abs(C-1)<=0.005` wird unverändert
für jede Maske separat ausgewiesen, ohne nachträgliche Wahl der besten Maske.
Je Zelle werden verfügbare Fälle, operative Fehler, Anzahl erfüllter Kriterien
von 16 sowie punktweise 95%-Wilson-Intervalle berichtet. Fehler bleiben im Nenner;
zusätzlich werden bedingte Raten unter verfügbaren Fällen angegeben.

Für direkte Kontinuität wird zusätzlich eine benannte bisherige Screening-
Konjunktion ausgewiesen: beide Graphmasken quantisiert mit Betrag 1 und gleichem
Vorzeichen wie alle Bott-/Localizer-Werte, `L>=0.20`, vier PH-Paare mit Residuum
höchstens `1e-8` und mindestens vier der acht niedrigsten Absolutenergiezustände
mit Randgewicht mindestens 0.80. Ressourcenregeln gelten hier je n.
Die fünf Masken werden nicht gemeinsam als neues verschärftes Gate verlangt.
Größenabhängige Randfraktion und feste Zahl untersuchter Randzustände sind zu
berichten. Gleiche Screening-Regeln allein machen `L` nicht größenunabhängig.

## Seeds, Budget und Startfolge

Vorlaufwurzeln: `10_839_900` und `10_839_901`.
Hauptlaufwurzeln: `10_840_000` bis `10_840_015` einschließlich.
Diese neuen Rollen bleiben bis zum jeweiligen Nutzerstart unbenutzt.

Je Wurzel liefert `PCG64(root).random_raw(3)` drei Größenseeds in Reihenfolge
8,10,12. Je Größenseed liefert `PCG64(size_seed).random_raw(K+1)` zuerst einen
Positionsseed, dann K Schrittseeds. Das Positionsfeld wird unabhängig von der
Kantenfolge erzeugt. Alle Ableitungen werden gespeichert; Kollisionen zwischen
eigenständigen Rollen oder mit alten reservierten Rollen stoppen vor Physik,
ohne Neuziehung. Zukünftige Hauptlaufseeds werden beim Vorlauf nicht abgeleitet.
Tests verwenden gesonderte synthetische Seeds.

| Stufe | vollständige Physikbewertungen |
| --- | ---: |
| Vorlauf | `3 Größen * (2 Trajektorien * 3 Varianten + 2 Kontrollen) = 24` |
| Hauptlauf | `3 Größen * (16 Trajektorien * 3 Varianten + 2 Kontrollen) = 150` |
| Gesamt | 174 |

Jeder Slot enthält das gesamte Methodenpanel; Methodenaufrufe sind keine weiteren
unabhängigen Stichproben. Reihenfolge: Größen aufsteigend; je Größe Anfangskontrolle,
Positionsblock, Kantenblock, Kombinationsblock (darin Wurzeln aufsteigend),
Endkontrolle. Alle geplanten Varianten werden unabhängig vom Ergebnis berechnet.
Vorlaufdaten werden nicht mit Hauptlaufdaten zusammengelegt. Kein adaptiver
Zusatzseed, kein Physik-Caching zwischen verschiedenen Slots, kein Stopp wegen
eines besonders guten oder schlechten wissenschaftlichen Werts.

Eine direkte Wiederverwendung desselben Projektors innerhalb eines Slots ist
zulässig, sofern Tests die Gleichwertigkeit der Rohresultate mit den bestehenden
Methoden zeigen. Für jede Maske wird die vollständige positionsaufgelöste
Markerkarte gespeichert. Replays nach Abbruch zählen als zusätzliche technische
Arbeit. Aufbau-, Solver-, Diagnose-, Speicher- und Gesamtlaufzeit werden getrennt
beobachtet; ETA wird aus dem Vorlauf je Größe geschätzt.

## Kontrollen, Implementierung und Interpretation

Nullkontrollen müssen gültig und vollständig auswertbar sein. Alle Bott-,
Localizer- und fünf Chern-Indizes des Quadrats müssen +1 sein; `L>=0.20` und das
bisherige Randgate müssen gelten. Scheitert dies, wird die Ursache untersucht
und keine Schwelle während des Laufs geändert. Pro Größe werden Anfang/Ende
diskrete Resultate exakt sowie alle Eigenenergien, Chern-/Bott-Rohschätzungen
und Localizer-Gaps mit `rtol=0,atol=1e-10` verglichen. Keine Bytegleichheit
einzelner Eigenvektoren in entarteten Unterräumen verlangen.

Ein Kontrollfehler verhindert physikalische Schlüsse für die betroffene Größe;
ein Fehler der Anfangskontrolle stoppt die Rechnung dieser Stufe. Operative
Variantenfehler bleiben erkennbar und im Budget, übrige Slots laufen weiter.
Integritäts- und Speicherfehler stoppen. Der Vorlauf benötigt alle 24 verfügbaren
Bewertungen und bestandene Kontrollen; keine Mindestzahl erfolgreicher Varianten.

Architektur: separater größenabhängiger Validator und Topologie-Input-Builder,
bestehendes Modell und Methoden wiederverwenden. `Phase98TopologyInputs` kann
variable Masken und Perioden tragen; die alte primäre Gatefunktion erwartet aber
exakt zwei Chern-Resultate. Das neue Panel benötigt daher eine eigene Auswertung.
Keine Änderung des eingefrorenen alten Runner-Verhaltens und keine Phase 11.

Der neue Runner erhält einen separaten Befehl und als vorgesehenen Ergebnisstamm
`results/phase_10_size_methods_v1`, Stufen `preflight/full`. Noch nicht starten:
Umsetzung und Startbefehle folgen nach dem Protokoll-Commit. Der Nutzer startet
beide Stufen ausdrücklich. Vollständige Inputs, Eigenpaare, Markerfelder,
Geometrie-IDs, Rollen, Seeds, Protokoll-/Code-Hash und Python-/BLAS-Umgebung
werden gespeichert. Dateien werden unter neuen Namen exklusiv veröffentlicht.

Resume muss gespeicherte Pläne und versiegelte Zellinputs miteinander abgleichen,
alle Checksummen prüfen, keine abgeschlossene Physik wiederholen und aus dem
tatsächlich verfügbaren Pipeline-/Gridstatus die Verfügbarkeit ableiten.
Keine bloße Prüfung, ob ein Ergebnis-Dictionary vorhanden ist. Ein Abbruch vor
Versiegelung bleibt mit seinen bisherigen Artefakten nachvollziehbar; Ergebnisse
dürfen nicht durch günstigere Wiederholungen ersetzt werden.

Vor Nutzerstart: Python 3.14, `PYTHONDONTWRITEBYTECODE=1`, `-B`, alle vier
BLAS/OpenMP-Threadvariablen auf 1; fokussierte Tests, vollständiges pytest,
projektweites Ruff, relevantes strict mypy, `git diff --check`, `git status`.
Besonders testen: Größenbudgets, Masken-/Ortszuordnung, Schema- und
Provenienzprüfungen, unterbrochenes Resume, echte physikalische Nullkontrolle,
Fehlernenner und Vorlauf-Sperre bei nicht verfügbaren Ergebnissen.

Abnehmende zentrale Chern-Abweichungen bei größeren Systemen würden eine
Finite-Size-Erklärung stützen, ohne sie allein zu beweisen. Bleiben Unterschiede
zwischen Graph- und festen Masken, wird die Auswahlregion weiter untersucht.
Bleiben Abweichungen auch im Zentrum bestehen, kommen unter anderem lokale
physikalische Veränderungen und eine begrenzte Mittelungsregion infrage.
Keine dieser Beobachtungen rechtfertigt automatisch eine neue Toleranz,
die Bezeichnung als entdeckte Familie oder eine Aussage über Disorder-Robustheit.
Eine weitere Suche benötigt anschließend ein eigenes Protokoll.
