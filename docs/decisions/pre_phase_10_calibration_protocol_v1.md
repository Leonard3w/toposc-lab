# Phase-10-Kalibration: kontrollierte Änderungen am Quadrat, Protokoll v1

## Status und Freigabe

Protokoll-ID: `TOPOSC-P10-CAL-001`.

Am 2026-09-09 auf Nutzerauftrag als neuer, explorativer Folgeversuch ausgearbeitet.
Ausgangscommit: `674e38a614e1c41e40d9ea0b1cd435b100b91ac2` auf `main`.
Der Worktree war vor diesen Dokumentänderungen sauber. Dieses Dokument ist
zunächst ein prüfbarer Entwurf: Die neuen Zahlen, Seeds und Regeln werden erst
durch einen separaten, vom Nutzer ausgeführten Protokoll-Commit eingefroren.
Ein Runner dafür existiert noch nicht; die bisherigen `phase-10-research`-Befehle
führen diesen Versuch nicht aus.

Reihenfolge: Protokoll prüfen und separat committen; anschließend den eng
begrenzten Runner implementieren, testen und separat committen; danach startet
der Nutzer zuerst den technischen Vorlauf und später ausdrücklich den Hauptlauf.
Das Manifest muss den vollständigen Protokoll-Commit und den davon verschiedenen
Implementierungs-Commit festhalten. Die endgültigen Hashes stehen jetzt noch
nicht fest. Hier wurden keine neuen Kalibrationsgeometrien oder Physikergebnisse
berechnet und keine Ergebnisordner erzeugt.

Dies bleibt Forschung mit den vorhandenen Phase-10-Bausteinen, keine Phase 11.
Verbindlicher wissenschaftlicher Rahmen ist die
[Forschungscharter](pre_phase_9_research_charter.md). Der
[alte Suchversuch](pre_phase_10_research_protocol_v1.md), sein
[Speicher-Amendment A1](phase_10_research_storage_amendment_a1.md) und sämtliche
alten Ergebnisse bleiben unverändert. Gelöschte Phase-9.8-Ergebnisse werden
nicht rekonstruiert.

## Ausgangsbefund und Frage

Der abgeschlossene Erstversuch
[`phase_10_research_v1_storage_fix1/full`](../../results/phase_10_research_v1_storage_fix1/full/report.md)
ergab bei beiden Sucharmen 0/32 Versuchstreffer. Das feste Quadrat bestand die
sauberen Gates; sein Schutzproxy war `L = 0.2314742436299257`, die alte
Trefferschwelle `0.2546216679929183`. Keine der 2.048 Kandidatenbewertungen
bestand sämtliche sauberen Gates. Diese Bewertungen sind nicht 2.048 unabhängige
Geometrien: gemeinsame Starts und erneut bewertete Eliten sind enthalten.

In der lesenden Diagnose blieben beide lokalen Chern-Bulkschätzungen bei den
Suchkandidaten nicht gemeinsam quantisiert. Bott und Localizer lieferten oft
unterschiedliche Einstufungen. Beim Quadrat beträgt die Graphabstand-2-Schätzung
ungefähr `0.9952868`: Ihr Abstand zur ganzen Zahl liegt bereits nahe an der
festen Toleranz `0.005`. Das motiviert auch eine sehr kleine Verschiebungsstufe.

Das bedeutet weder, dass alle amorphen Geometrien trivial sind, noch dass der
gesamte bisherige Suchraum keine geeigneten Kandidaten enthält. Ebenfalls offen
bleiben endliche Größe, Rand-/Bulk-Auswahl, methodische Anwendbarkeit und mögliche
Implementierungsprobleme. Der Folgeversuch wird nach Kenntnis dieses Befunds
geplant und ist ausdrücklich keine unabhängige Bestätigung einer zuvor
registrierten Kalibrationshypothese.

> Welche der unveränderten Diagnosekriterien des erfolgreichen Quadrats bleiben
> bei kleinen Positionsänderungen, wenigen Kantenaustauschen und deren
> Kombination erfüllt, und welche Rohwerte verändern sich dabei?

Erwartetes Produkt ist eine vollständig berichtete Karte dieses kleinen
Quadrat-Umfelds, einschließlich negativer und nicht auswertbarer Fälle. Es gibt
keine Evolution, Selektion, Fitnessoptimierung oder Gewinnerauswahl. Ziel ist
nicht, nachträglich Treffer im alten Versuch zu erzeugen oder neue Grenzwerte
so lange anzupassen, bis Ergebnisse positiv werden.

## Architekturprüfung und minimaler Vertrag

Die vorhandenen Schnittstellen reichen für die wissenschaftliche Auswertung:

- `square_reference()` liefert den benannten Ausgangsgraphen einschließlich
  physikalischer Randannotation; `Geometry` / `GeometryGenome` tragen ausschließlich
  Geometrie. Seeds, Versuchszelle, Modell und Ergebnisse bleiben außerhalb davon.
- `validate_phase_9_8_geometry(..., CLEAN_PRIMARY)` prüft den unveränderten
  Ressourcen- und Einbettungsvertrag. Die neue Konstruktionsregel wird zusätzlich
  separat geprüft.
- `evaluate_phase_9_8_primary_geometry` liefert dieselbe Physik, Topologie-Grids
  und sauberen Gates. `build_phase_9_8_primary_topology_inputs` liefert die jeweils
  tatsächlich verwendeten Flächen und Bulkmasken. Kein neuer Topologiealgorithmus.
- Bestehende Geometriearchive und exakte Snapshot-IDs dienen der Nachvollziehbarkeit,
  nicht als Beweis physikalischer Verschiedenheit oder topologischer Neuheit.
  Die vorhandene geprüfte Record-Serialisierung und exklusive Veröffentlichung
  können ohne einen allgemeinen Dataset-Layer wiederverwendet werden.

Nicht blind wiederverwendbar sind `require_research_geometry` und
`legal_edge_swaps`: Sie binden den alten Suchversuch zusätzlich an einen bestimmten
Delaunay-Pool. Der neue Versuch bekommt einen eigenen, unten definierten
Quadrat-Nachbarschaftspool. Insbesondere wird keine der beiden Zell-Diagonalen
allein durch die Triangulationsentscheidung am regelmäßigen Quadrat bevorzugt.
Beide dürfen im Pool liegen, aber nie gleichzeitig kreuzend im ausgewählten Graphen.
Dies ist ein offen deklarierter neuer Konstruktionsraum, keine Änderung am alten.

Positionsänderung ist im vorhandenen Modell mehr als eine andere Zeichnung:
`build_chiral_p_wave_pairing` verwendet die normierte physikalische Kantenrichtung.
Deshalb werden nach jeder Konstruktion sämtliche Displacements aus den endgültigen
Koordinaten neu gebildet. Hopping-/Pairing-Beträge erhalten keinen Distanzabfall.
Auch Voronoi-Flächen werden aus den tatsächlichen Koordinaten neu berechnet.

Kantenaustausch verändert nicht nur den Hamiltonian, sondern möglicherweise
auch die graphabstandsbasierte Bulk-Auswahl. Die Randknoten selbst bleiben fest.
Die Masken und ihre Größen müssen deshalb je Fall sichtbar sein; dieser Versuch
isoliert nicht automatisch einen Hamiltonian-Effekt bei konstanter Bulk-Messregion.
Beim Ausgangsquadrat sind es 16 bzw. 4 physikalische Bulk-Knoten.

## Feste Physik und Geometrie

Der vollständige physikalische Vertrag des
[Erstversuchs](pre_phase_10_research_protocol_v1.md#übernommener-physikalischer-vertrag)
gilt unverändert, insbesondere:

| Größe | Festlegung |
| --- | --- |
| Modell | `ChiralPWaveModel`, Hopping 1, chemisches Potential 2, Pairing 1, Chiralität +1, Achsen `(0, 1)` |
| Ressourcen | 64 Knoten, 112 ungerichtete Kanten, Box `[0, 7]²`, Grad 2–4, zusammenhängend |
| Einbettung | Mindestabstand 0.55, maximale Kantenlänge 1.75, keine geraden Kantenkreuzungen |
| Rand | die 28 ursprünglichen Randknoten, äußere Schale 0.875, eine äußere Komponente, keine Lochränder |
| Solver | vollständiges `numpy.linalg.eigh`, Referenzenergie 0, numerische Toleranz `1e-10`, `evaluation_seed=None` |
| Bott | Perioden 7.6, 8.0, 8.4 auf beiden Achsen, Quantisierungstoleranz `1e-6` |
| Lokaler Chern-Marker | Graphabstand mindestens 2 bzw. 3 zum Rand, Voronoi-Zelle `[-0.5, 7.5]²`, Quantisierungstoleranz `5e-3` |
| Localizer | Position `(3.5, 3.5)`, Energie 0, Kappa 0.1, 0.2, 0.3 |
| Topologiegate | alle erforderlichen Gridwerte aufgelöst, Betrag 1, gleicher vorzeichenbehafteter Index und bestehende Konvergenzchecks |
| Schutzproxy | `L = min` der drei Localizer-Gaps, Gate `L >= 0.20` |
| Randgate | unter den acht niedrigsten Absolutenergien vier PH-Paare mit Residuum höchstens `1e-8`; mindestens vier der acht Randgewichte mindestens 0.80 |

Alle bisherigen Zustands-, Nambu-, Voronoi-, Validierungs- und Warnungskonventionen
werden übernommen. Insbesondere gilt komponentenweises Nambu-Tiling und kein
Ersatz leerer Bulkmasken. `clean_eligible` bleibt die bisherige Gatekonjunktion.
Es wird nicht auf die gelockerten Geometriegrenzen eines Disorder-Kanals umgestellt.
Die alte Schwelle `1.10 * R` ist hier kein Endpunkt: Wir untersuchen zunächst den
Erhalt der sauberen Kriterien, nicht eine Überlegenheit gegenüber Referenzfamilien.

## Konstruktion und faktorieller Aufbau

Ausgangspunkt sind die Koordinaten `X0` und Kanten `E0` des offenen
`square(n_x=8, n_y=8, spacing=1.0)` in der bestehenden Knotennummerierung.
Für alle Kalibrationssnapshots einschließlich der Nullkontrolle gilt dieselbe
neutrale Darstellung: Kanten lexikographisch sortiert, kleinere zu größerer
Knotennummer, Typ `calibration_coupling`, kein Randübertritt, Displacement
`X[target] - X[source]`. Es werden keine Faces, Baumstrukturen oder Site-Typen
hinzugefügt. `dimension_records=()` vermeidet eine unbelegte Übernahme der
Translationsrang-Aussage einer unendlichen regelmäßigen Familie.
Die zweidimensionale Einbettung bleibt davon unabhängig erhalten.

Geometriemetadaten enthalten nur
`generator="phase10_calibration_square_deformation_v1"`; Rollen, Seeds,
Amplituden und Operationshistorie stehen im separaten Versuchsrecord. So erhalten
identische rekonstruierte Snapshots nicht allein wegen ihrer Versuchsrolle
verschiedene IDs. Der benannte Quellgraph und sein Snapshot werden separat
dokumentiert. Der Nullfall ist physikalisch das bekannte Quadrat, nicht zwingend
byteidentisch zu dessen älterem, anders annotiertem Geometriearchiv.

### P: Positionen, unveränderte Kanten

Pro unabhängiger Trajektorie `r` wird genau ein Feld `U_r` gezogen: für die 36
inneren Knoten in aufsteigender Knotennummer je zwei unabhängige Werte aus
`Generator(PCG64(position_seed)).uniform(-1.0, 1.0, size=(36, 2))`.
Die 28 Randknoten erhalten exakt `(0, 0)`.

Die Amplituden sind `a = (0, 0.005, 0.025, 0.05, 0.10)`, in Einheiten des
ursprünglichen Gitterabstands **pro Koordinatenkomponente**, nicht als feste
euklidische Verschiebung. Es gilt `X_r(a) = X0 + a * U_r` in Float64,
immer vom Original aus gerechnet, ohne aufeinander aufsummierte Bewegungen.
Bei `a=0` werden die Originalkoordinaten direkt übernommen.
Dasselbe Feld wird für alle Amplituden und Kantenstufen dieser Trajektorie genutzt.

Die äußerste innere Koordinate liegt mindestens bei 0.9, also außerhalb der
Randschale 0.875. Spannweite und Randmitgliedschaft bleiben fest. Dies untersucht
schwache Positionsunordnung im Inneren, keine vollständig amorphe Punktwolke,
keine Randaufrauung und keine große geometrische Deformation.

### C: Konnektivität, unveränderte Positionen

Der feste potenzielle Kantenpool enthält alle Paare `i < j`, für die sich die
beiden ursprünglichen Gitterkoordinaten in jeder Komponente um höchstens 1
unterscheiden. Das sind die 112 axialen Kanten plus beide Diagonalen der 49
Elementarzellen, insgesamt 210 mögliche Kanten. Der Pool wird nur aus `X0`
bestimmt, niemals aus einem Physikergebnis oder dem Verschiebungsfeld.

Beginnend bei `E0` werden acht aufeinanderfolgende Austauschschritte versucht.
Ein Schritt entfernt genau eine vorhandene und ergänzt genau eine fehlende Kante.
Alle Vorschläge `(removed_source, removed_target, added_source, added_target)`
werden lexikographisch geordnet. Die zulässige Menge enthält genau die Vorschläge,
deren fertiger Graph auf `X0` den vollständigen `CLEAN_PRIMARY`-Vertrag erfüllt.
Ein Vorschlag wird mit `Generator(PCG64(step_seed)).integers(n_legal)` gleichverteilt
aus dieser Liste gewählt. Keine Physikbewertung beeinflusst die Auswahl.

Bei leerer Liste wird der Schritt als expliziter No-op protokolliert; es gibt
keinen Ersatzseed oder Reparaturversuch. Rücktausch und wiederkehrende Graphen
sind erlaubt. Behalten werden die Zustände nach `k = (0, 1, 2, 4, 8)` versuchten
Schritten. `k` bedeutet nicht die Zahl verschiedener, dauerhaft geänderter Kanten.
Zusätzlich wird `d = |E_k symmetric_difference E0| / 2` berichtet.

Grad 2–4 wird eingehalten, die individuelle Gradfolge aber nicht fixiert.
Die inneren Quadrat-Knoten starten bereits mit Grad 4. Daher können frühe legale
Änderungen durch die Gradgrenze stark auf randnahe Kanten beschränkt sein.
Berichtet werden bei jedem Austausch die Endpunktklassen Rand/Rand,
Rand/Innen oder Innen/Innen, alle betroffenen Knoten und die resultierende Gradfolge.
Dies ist keine gleichmäßige Stichprobe aller zulässigen Graphen und kein
isolierter Vergleich bei identischer Gradfolge oder identischen Randkopplungen.

### P × C: dieselben Eingriffe kombinieren

Jede Trajektorie enthält das volle Raster `G_r(a, k) = (X_r(a), E_r(k))` aus
5 × 5 Zellen. Die Kantenfolge wird unabhängig vom Positionsfeld auf `X0` erzeugt
und bei allen Amplituden identisch verwendet; Positionsfelder werden bei allen
Kantenstufen identisch verwendet. Jede fertige Einbettung wird trotzdem vollständig
validiert. Es gibt keine nachträgliche Neu-Triangulierung, Längenreparatur oder
ergebnisabhängige Änderung der Kombinationen.

Vor der ersten Physikbewertung einer Stufe wird ihr gesamter Konstruktionsplan
mit allen Seeds, Feldern, Kantenfolgen, No-ops, IDs und Validierungen versiegelt.
Eine geometrisch ungültige Kombination stoppt die Konstruktion vor der Physik
mit Fehlernachweis; sie wird weder verworfen noch durch einen anderen Draw ersetzt.
Das ist ein zu untersuchender Konstruktionsfehler, kein negatives Physikergebnis.

## Seeds, Budget und Reihenfolge

| Rolle | Wurzelseeds | Umfang |
| --- | --- | ---: |
| Technischer Vorlauf | `10_829_900` bis `10_829_901` einschließlich | 2 Trajektorien |
| Kalibrationshauptlauf | `10_830_000` bis `10_830_015` einschließlich | 16 Trajektorien |

Aus jeder Wurzel werden durch `PCG64(root).random_raw(9)` genau neun UInt64-Werte
in fester Reihenfolge gezogen: zuerst der Positionsseed, danach die acht
Schrittseeds. Werte werden als Python-Integer gespeichert; der vollständige
abgeleitete Plan ist prüfbar. Rollen werden nicht durch Uhrzeit, Python-`hash`
oder globale RNG-Zustände bestimmt. Hauptlaufwurzeln werden vor dessen Freigabe
nicht ausgeführt, auch nicht in Implementierungstests. Tests verwenden eigene
synthetische Seeds außerhalb dieser und der alten reservierten Rollen.

Die alten Such-, Referenz-, Validierungs- und Bestätigungsseeds bleiben ihren
ursprünglichen Rollen zugeordnet und werden hier nicht verwendet. Die neuen
Trajektorien sind explorative Kalibrationsdaten, keine zukünftigen Holdouts.
Ein späterer Such- oder Bestätigungsversuch braucht eine eigene neue Freigabe.

Je Trajektorie gibt es 4 reine Positionsfälle, 4 reine Konnektivitätsfälle und
16 Kombinationen. Der gemeinsame Nullfall `(0, 0)` wird nicht als 16 unabhängige
Replikationen gezählt: Er wird einmal vor und einmal nach dem jeweiligen Lauf
unter demselben Code und derselben Umgebung wirklich ausgewertet.

| Stufe | Geplante vollständige Physikbewertungen |
| --- | ---: |
| Technischer Vorlauf | `2 × (4 + 4 + 16) + 2 = 50` |
| Hauptlauf | `16 × (4 + 4 + 16) + 2 = 386` |
| Beide Stufen zusammen, ohne Wiederholungsarbeit | 436 |

Feste Auswertungsreihenfolge: Nullkontrolle am Anfang; reine Positionsfälle
(Trajektorien aufsteigend, darin `a` aufsteigend); reine Konnektivitätsfälle
(Trajektorien aufsteigend, darin `k` aufsteigend); Kombinationen (Trajektorien,
darin `a`, darin `k` aufsteigend); Nullkontrolle am Ende. Alle vorgesehenen
Varianten werden unabhängig von Zwischenergebnissen ausgewertet.

No-ops, Duplikate und operative Bewertungsfehler verbrauchen ihren Slot. Es gibt
kein Physik-Caching und keine Zusatzseeds für schlechte Ergebnisse. Wiederholte
Arbeit nach einem Abbruch wird getrennt gezählt, nicht als neue Beobachtung.
Geometriekonstruktion und Physik erhalten getrennte Zeitmessungen; eine feste
Laufzeit wird vor dem Vorlauf nicht zugesichert.

## Kontroll-, Fehler- und Auswertungsregeln

Beide Nullkontrollen müssen `clean_eligible=True` liefern. Zwischen Anfang und
Ende müssen diskrete Gate- und Indexwerte übereinstimmen; finite Eigenenergien,
Bott-Rohschätzungen, beide Chern-Bulkschätzungen und alle drei Localizer-Gaps werden
mit `rtol=0`, `atol=1e-10` verglichen. Einzelne Eigenvektoren werden wegen möglicher
Basisrotationen entarteter Unterräume nicht auf Bytegleichheit geprüft.
Der alte Quadratwert ist bekannte Vorinformation, kein auf neue Code-/BLAS-Versionen
übertragbarer Bitgleichheitstest. Neue Rohwerte und Provenienz bleiben vollständig
sichtbar; eine unerklärte Änderung wird vor physikalischen Schlussfolgerungen geprüft.

Eine fehlgeschlagene Anfangskontrolle verhindert die Variantenrechnung. Eine
fehlgeschlagene Endkontrolle macht den Lauf für physikalische Schlussfolgerungen
kontrollungültig, ohne die bereits erzeugten Daten zu löschen. Speicher-,
Integritäts- und Provenienzfehler stoppen den Lauf; sie werden nicht als
wissenschaftlich gescheiterte Kandidaten verschleiert.

Operative Fehler einer Variantenbewertung bleiben mit Fehlertyp erhalten; sie
erlauben keine Aussage über die Topologie dieser Variante. Die übrigen geplanten
Slots laufen weiter, soweit Speicherung und Laufkontrolle intakt sind. Der
technische Vorlauf besteht nur bei vollständiger Speicherung, verfügbaren
Bewertungen, gültigen Nullkontrollen und bestandener Resume-/Integritätsprüfung.
Er verlangt ausdrücklich keine Mindestzahl erfolgreicher veränderter Geometrien.

Je Variante werden archiviert: vollständige Geometrie und Pipelineauswertung,
alle drei Bott-, zwei Chern- und drei Localizer-Gridresultate einschließlich
Rohwerten, Quantisierungsfehlern und Warnungen; tatsächliche Flächen, Bulkmasken
und Knotenzuordnung; `L`, sämtliche Gategründe, die acht geordneten Randgewichte
mit Anzahl über 0.80 und PH-Paarresiduen sowie die vorhandenen 16
Niedrigenergie-Zustandsdiagnosen. `minimum_boundary_weight_first_four` ist nicht
mit dem tatsächlichen Vier-aus-acht-Randgate zu verwechseln.

Es werden zusätzlich die einzelnen Gatebedingungen ausgewiesen, nicht nur der
erste zurückgegebene Sammelgrund: etwa können unaufgelöste Chern-Werte eine
gleichzeitig triviale Bott-Einstufung im Sammelgrund verdecken. Fehlend oder
unaufgelöst bedeutet niemals numerisch null. Lokale und globale Diagnosen bleiben
auch bei Widerspruch nebeneinander sichtbar.

Die primäre Berichtseinheit ist die unabhängige Trajektorie, nicht jede ihrer
24 korrelierten Varianten. Für jede feste Nichtnullzelle werden Anzahl
`clean_eligible` von 16, 95%-Wilson-Intervall, einzelne Gateanteile und Anzahl
operativer Fehler berichtet. Operative Fehler bleiben im Nenner dieser
Gesamterfolgsrate; zusätzlich wird die Rate unter verfügbaren Auswertungen mit
ihrem eigenen Nenner ausgewiesen. Die 24 Intervalle sind punktweise und nicht
simultan abgesichert. Vorlaufdaten werden nicht mit dem Hauptlauf zusammengelegt.

Rohwertdarstellungen zeigen alle Trajektorien sowie Median, Minimum und Maximum
der verfügbaren Werte mit Anzahl. Gepaarte Änderungen gegenüber dem gemeinsamen
Quadrat und den jeweils passenden reinen Eingriffen sind deskriptiv. Keine
p-Wert-Suche, kein neuer zusammengesetzter Score, kein adaptiver Abbruch und keine
Behauptung einer monotonen kritischen Amplitude: Verlust und Wiederkehr eines
Gates entlang derselben Trajektorie werden beide berichtet.

## Interpretation und vorab festgelegte nächste Entscheidung

| Beobachtung | Zulässige Konsequenz |
| --- | --- |
| Nullkontrolle, Integrität oder technische Auswertung scheitert | Ursache prüfen; kein physikalisches Negativresultat behaupten, keine Schwellen lockern |
| Positionseingriffe bestehen häufiger, Kanteneingriffe verlieren Gates | Konnektivität einschließlich Grad-, Randkopplungs- und Bulk-Maskenänderungen gezielt untersuchen; noch kein isolierter Mechanismusnachweis |
| Reine Eingriffe bestehen, Kombinationen verlieren Gates | Hinweis auf einen gemeinsamen Effekt in diesem Design; unabhängigen Folgeversuch planen |
| Schon die kleinsten Eingriffe verlieren Gates | Rohwerte und Methodenanwendbarkeit prüfen; nicht automatisch eine topologische Phasengrenze behaupten |
| Alle geprüften Eingriffe bestehen | Stabilität nur im untersuchten kleinen Quadrat-Umfeld; weder Vorteil noch Erreichbarkeit des alten amorphen Suchraums belegt |

Die jeweils beteiligten Trajektorien und Gatearten müssen genannt werden;
gemischte Resultate dürfen nicht auf ein bequemes Tabellenetikett reduziert werden.
Bott ist grundsätzlich global, der Localizer grundsätzlich lokal; Unterschiede
sind deshalb nicht automatisch ein Softwarefehler. Siehe
[Loring, A Guide to the Bott Index and Localizer Index](https://arxiv.org/abs/1907.11791).
Der lokale Chern-Marker ist räumlich aufgelöst und auch bei offenen Rändern
definiert; die gewählte finite Bulk-Auswertung bleibt daher sichtbar. Siehe
[Bianco und Resta, Mapping topological order in coordinate space](https://arxiv.org/abs/1111.5697).
Diese Literaturhinweise ersetzen keine Prüfung der konkreten Class-D-Implementierung.

Es gibt hier keine Größenextrapolation, Hamiltonian-Disorder-Validierung,
Majorana-Nullmodenbehauptung, neue Familie, Materialvorhersage oder Aussage über
Evolutionsüberlegenheit. Das Protokoll kalibriert keine Toleranzen nach Ergebnissen,
sondern untersucht das Verhalten der bisherigen festen Diagnosepipeline.
Eine Änderung von Modell, Gate, Amplituden, Operationen oder Stichprobe braucht
ein neues versioniertes Protokoll und frische dafür reservierte Datenrollen.

## Anforderungen an die spätere Umsetzung

Ein eigener, schmaler Runner trennt Konstruktion, unveränderten Physikadapter,
Kontrolle, Speicherung und deskriptiven Bericht. Keine Änderung des eingefrorenen
`phase-10-research`-Verhaltens und keine Erweiterung zu einem allgemeinen Dataset.
Vorgesehener neuer Ergebnisstamm: `results/phase_10_calibration_v1`, mit getrennten
Stufen `preflight` und `full`. Ein belegter Stamm wird ohne explizites, validiertes
Resume nicht überschrieben; im jetzigen Dokumentationsschritt wird er nicht angelegt.

Live-Fortschritt nennt Stufe, Block `positions/connectivity/combined/control`,
Trajektorie, `a`, `k`, erledigte/geplante Slots, Gateergebnisse, Laufzeit, ETA,
CPU/RAM und letzten versiegelten Stand. Nicht das alte missverständliche
Suchfeld `hits` als alleinige Statusinformation übernehmen. Ein lesbares
`report.md` erklärt die Gatekarte; ein maschinenlesbarer Bericht erhält Rohwerte
und Nenner. Dafür muss der Nutzer den Lauf und das Mitlesen selbst starten können.

Provenienz: Protokoll- und Code-Commit, Statusprüfung, Python-/Paketversionen,
BLAS-/Threadumgebung, alle Rollen und abgeleiteten Seeds, geplanter Umfang sowie
Hashes der versiegelten Artefakte. Unter Windows ausschließlich neue, unveränderliche
Records/Checkpoint-Namen veröffentlichen; bereits offene Zielarchive nicht ersetzen.
Resume verwendet gespeicherte Eingaben und dieselbe Revision/Umgebung, verlängert
weder Raster noch Budget und prüft die versiegelten Inhalte vor Wiederaufnahme.

Vor dem Nutzerstart: Python 3.14 mit `PYTHONDONTWRITEBYTECODE=1` und `-B`,
fokussierte Tests, vollständiges pytest, projektweites Ruff, relevantes strict
mypy, `git diff --check` und `git status`. Bestehende Ruff-Befunde sind von neuen
zu trennen. Für numerische Tests und Lauf gelten `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `BLIS_NUM_THREADS` jeweils `1`.
Tests prüfen insbesondere unabhängige Rollen, identische gepaarte Eingriffe,
Displacements, Rand-/Grad-/Kreuzungsregeln, Nullfall, 50/386-Slotbudgets,
Rohwert-/Maskenzuordnung, No-ops, Fehlernenner, unveränderte alte CLI und Resume.

Keine geschützten `.pyc`-Artefakte, `geometry_demo.npz` oder alten Ergebnisse
verändern, löschen oder committen. Der Nutzer führt sämtliche Commits, Pushes
und wissenschaftlichen Starts selbst aus.
