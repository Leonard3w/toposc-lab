# Phase-10-Forschungslauf: Evolution gegen Zufallssuche, Protokoll v1

## Status und Freigabe

Protokoll-ID: `TOPOSC-P10-EVO-RS-001`.

Am 2026-09-08 auf Nutzerauftrag ausgearbeitet, vor Implementierung und
Ausführung dieses Forschungsversuchs. Ausgangspunkt ist der abgeschlossene
Phase-10-Stand `71d376af8ebbadec9ccc254ffddbe64f527b903e` auf `main`.
HEAD, lokales `origin/main` und die lesend geprüfte GitHub-Referenz stimmen
überein; der Worktree war vor dieser Dokumentänderung sauber.

Dieses Dokument konkretisiert den besprochenen Versuch zur
[Forschungscharter](pre_phase_9_research_charter.md) und zum
[Phase-10.20-Benchmark](phase_10_20_search_benchmark.md). Es ist ein
Forschungsprotokoll auf Basis von Phase 10, kein Beginn von Phase 11.
Die allgemeine Zustimmung zum Versuch ist bereits erfolgt. Die hier neu
ausformulierten Zahlen, Seeds und Entscheidungsregeln sind zur Prüfung sichtbar;
ihre endgültige Fassung wird durch einen separaten Protokoll-Commit festgelegt.

Die Reihenfolge ist:

1. Dieses Dokument und seinen Verweis in der Bedienungsanleitung separat
   committen. Der Nutzer führt die Git-Befehle aus.
2. Die Umsetzung referenziert den vollständigen Hash dieses Protokoll-Commits
   und wird nach ihren Tests in einem weiteren Commit festgehalten.
3. Der Nutzer startet den technischen Vorlauf mit reservierten Vorlauf-Seeds.
4. Nach bestandenem Vorlauf startet der Nutzer den vollständigen Versuch selbst.

Der Protokoll-Hash ist nicht der oben genannte Ausgangscommit. Er steht erst
nach Schritt 1 fest. Ein Manifest muss beide Revisionen eindeutig unterscheiden.
In diesem Vorbereitungsschritt wurden keine neuen Geometrien, Hamiltonians,
Suchresultate oder Disorder-Realisierungen berechnet. Ältere Phase-9.8-Ergebnisse
wurden weder geladen noch rekonstruiert. Die frühere Kalibrierung im
Phase-9.8-Protokoll ist bekannte Vorinformation; dieser Versuch ist deshalb keine
unabhängige Neubegründung jener physikalischen Parameter und Schwellen.

## Frage und zulässiges Ergebnis

> Findet die festgelegte evolutionäre Konnektivitätssuche bei gleichem
> Bewertungsbudget häufiger mindestens einen starken Screening-Kandidaten
> als unabhängige Zufallssuche?

Die Replikationseinheit ist ein gepaartes Suchversuchspaar. Die primäre
Zielgröße ist die Wahrscheinlichkeit, innerhalb von 32 Bewertungsversuchen
mindestens einen Kandidaten über der eingefrorenen Referenzschwelle zu finden.
Trefferzeitpunkt und beste beobachtete Rohwerte sind sekundäre Ergebnisse.

Der Versuch untersucht saubere, endliche 64-Knoten-Systeme im bestehenden
spinlosen chiralen Class-D-Modell. Er liefert Suchleistungsdaten und konkrete
Geometrien für eine spätere Validierung. Eine höhere Trefferrate allein belegt
weder größere Disorder-Robustheit noch einen kausalen geometrischen Mechanismus.
Thermodynamische Aussagen, isolierte Majorana-Nullmoden, Materialvorhersagen,
Literaturpriorität und ein Vorteil gegenüber ganzen Referenzfamilien liegen
außerhalb dieser Auswertung.

Die konkrete Nullhypothese dieses Suchvergleichs lautet: Beide Suchstrategien
haben dieselbe marginale Wahrscheinlichkeit eines Versuchstreffers, bedingt auf
das eine festgelegte Referenzpanel und dieses Protokoll. Das ist eine andere
Hypothese als die physikalische Robustheitshypothese der Forschungscharter.

## Übernommener physikalischer Vertrag

Verbindliche Grundlage für Generatoren, Modell, numerische Konventionen und
wissenschaftliche Gates ist das
[Phase-9.8-Protokoll](pre_phase_9_8_random_search_protocol_v1.md), eingefroren
unter `dc967ec2876f221d7b4f362f6224d7d3716f395e`. Übernommen werden dessen
Abschnitte zum physikalischen Modell, primären Kandidatenstratum,
stochastischen Generator, Referenzkonstruktionen, Topologieinputs und sauberen
Zulässigkeitsgates. Suchalgorithmus, Seeds, Budget, Statistik, Persistenz und
Validierungsablauf werden für diesen neuen Versuch unten eigenständig definiert.

| Festlegung | Wert |
| --- | --- |
| Modell | `ChiralPWaveModel`, spinlos, chirales `p_x + i p_y` |
| Parameter | Hopping `1.0`, chemisches Potential `2.0`, Pairing `1.0` |
| Orientierung | Chiralität `+1`, physikalische Achsen `(0, 1)` |
| Energie und Toleranz | Referenzenergie `0.0`, numerisch `1e-10` |
| Zustände | 16 niedrigenergetische Zustände, Nullmodentoleranz `1e-10` |
| Ressourcen | genau 64 Knoten und 112 ungerichtete Kopplungen |
| Koordinaten | explizit `(64, 2)`, Spannweite genau `[0, 7] × [0, 7]` |
| Abstände | Knotenabstand mindestens `0.55`, Kantenlänge höchstens `1.75` |
| Graph | zusammenhängend, Grad 2–4, keine Schleifen oder Doppelkanten |
| Einbettung | gerade Kanten ohne Kreuzungen außerhalb gemeinsamer Endpunkte |
| Physikalischer Rand | äußere Koordinatenschale der Dicke `0.875`, 24–32 Knoten |
| Randkomponente | genau eine äußere Komponente, keine deklarierten Lochränder |
| Solver | bestehende vollständige Diagonalisierung mit `numpy.linalg.eigh` |
| Zufälligkeit der sauberen Physik | keine, `evaluation_seed=None` |

Die Orientierung jeder Kante ist kleinere zu größerer Knotennummer; ihr
Displacement ist Zielkoordinate minus Quellkoordinate. Die Kantenfolge ist
lexikographisch nach Endpunkten geordnet. Hopping und Pairing bleiben pro Kante
gleich stark, ohne Distanzabfall oder Optimierung von Modellparametern.
Die vorhandene Geometrieprüfung verwendet für ihre toleranzbehafteten Tests
`1e-12`; direkte Abstands- und Gradgrenzen bleiben direkt geprüft.

Die vollständigen Phase-9.8-Konvergenzprüfungen gelten unverändert:

- Bott-Perioden `(7.6, 7.6)`, `(8.0, 8.0)`, `(8.4, 8.4)`;
- lokale Chern-Marker auf Graphabstand-2- und Graphabstand-3-Bulkmasken;
- positive, auf die Zelle `[-0.5, 7.5]²` beschnittene Voronoi-Flächen mit
  Flächensumme 64 innerhalb `1e-10`;
- Localizer-Probe `(3.5, 3.5)`, Energie 0, Kappa `0.1`, `0.2`, `0.3`;
- numerische Methodentoleranz `1e-10`, Quantisierungstoleranzen `1e-6` für
  Bott und `5e-3` für den lokalen Chern-Marker;
- alle erforderlichen Gridwerte aufgelöst, Betrag des Index jeweils 1 und
  gleicher vorzeichenbehafteter Index über alle Methoden und Gridwerte.

Komponentenweise Nambu-Anordnung verlangt `np.tile(coordinates, (2, 1))`.
Leere Bulkmasken werden nicht gelockert. Methodenspezifische Konvergenzflags
müssen aus den behaltenen Grids stammen; die drei pipelineweiten
Topologiepflicht-/Konvergenzflags bleiben wie in Phase 9.8 auf `False`.

Der Schutzproxy `L` ist das Minimum der drei Localizer-Gaps und muss mindestens
`0.20` betragen. Die ersten acht Zustände in `(abs(energy), state_index)`-Ordnung
müssen vier deterministisch zugeordnete Teilchen-Loch-Energiepaare mit maximalem
Paarresiduum `1e-8` bilden. Mindestens vier dieser acht Zustände müssen Randgewicht
mindestens `0.80` haben. Die Phase-9.8-Minimumkostenpaarung einschließlich ihres
Tie-Breaks gilt unverändert. Energien, IPR, Ortswahrscheinlichkeiten, Randgewichte
und individuelle Majorana-Diagnostik bleiben erhalten.

`clean_eligible` ist genau die Konjunktion der bestehenden Geometrie-, Topologie-,
Schutzproxy- und Randgates. Es gibt keine zusätzliche Forderung nach Nullmoden
oder individueller Majorana-Selbstkonjugation. Der endliche spektrale `gap`
behält seine bisherige Bedeutung und wird nicht als Bulk- oder topologische
Lücke umbenannt.

## Referenzen und Trefferschwelle

Vor der Suche wird ein gemeinsames Referenzpanel unter derselben Code-Revision
ausgewertet und versiegelt:

1. Ein offenes Quadratgitter `square(n_x=8, n_y=8, spacing=1.0,
   boundary_x="open", boundary_y="open")`. Seine vorhandenen 28 Randknoten
   werden wie in Phase 9.8 explizit als äußere Randkomponente annotiert.
2. 32 Realisierungen von `hard_core_planar_reference`, Version 1, aus den
   unten reservierten neuen Referenz-Seeds. Zufälliger Spannbaum und
   längenpriorisierte Ergänzung folgen dem bestehenden Generatorvertrag.
3. Als deskriptive Kontrollen einmal `ammann_beenker_patch(radius=4.0,
   spacing=1.0)` und einmal `sierpinski_carpet(order=2, spacing=1.0)` mit ihren
   nativen Phase-9.8-Topologieinputs und unveränderter Familiengeometrie.

Die letzten beiden Kontrollen sind nicht ressourcengleich und gehen weder in
die Schwelle noch in den primären Vergleich ein. Ihr vorhandener
Anwendbarkeitsstatus wird erhalten, insbesondere die fehlende lokale
Chern-Bulkmaske beim kleinen Sierpiński-Teppich.

`R = max(L)` über alle sauberen, zulässigen primären Referenzrealisierungen.
Die daraus bestimmte Schwelle ist vor dem ersten Suchkandidaten unveränderlich.
Ein verfügbares Kandidatenergebnis zählt genau dann als Treffer, wenn
`clean_eligible is True` und `L >= 1.10 * R`.

Wissenschaftlich unzulässige Referenzen bleiben mit Gründen im Panel und gehen
nicht in `R` ein. Ein operativer Auswertungsfehler einer primären Referenz
verhindert dagegen den Suchstart: Ein möglicherweise stärkerer Vergleich darf
nicht durch einen Softwarefehler aus der Schwelle verschwinden. Gibt es keine
zulässige primäre Referenz, endet der Lauf ohne Suchvergleich.

Die Schwelle gilt gegenüber dem stärksten zulässigen Mitglied dieses festen
Panels, nicht gegenüber einem bekannten Maximum über sämtliche Referenzfamilien.
Randanteil und Gradprofil werden innerhalb ihrer erlaubten Bereiche berichtet;
diese Bereichskontrolle ist kein exaktes Matching jeder lokalen Struktur.

## Sucharme, Budget und Reihenfolge

| Größe | Festlegung |
| --- | ---: |
| Unabhängige gepaarte Versuche | 32 |
| Populationsgröße `N` | 8 |
| Übergänge nach Generation 0 | 3 |
| Bewertungen je Arm und Versuch | `8 × (3 + 1) = 32` |
| Bewertungen Evolution insgesamt | 1.024 |
| Bewertungen Zufallssuche insgesamt | 1.024 |
| Gemeinsames Referenzpanel | 33 primäre + 2 deskriptive Bewertungen |
| Vollständiger Lauf ohne Vorlauf/Wiederholungsarbeit | 2.083 Bewertungsversuche |

Je Versuch erzeugt `hard_core_planar_graph`, Version 1, acht Startgeometrien.
Die Registry erhält die abgeleiteten Sampler-Seeds direkt; es gibt keinen
zusätzlich vorgeschalteten Phase-9.1-Sampler. Beide Arme erhalten dieselben
geordneten Genome. Jeder Arm wertet sie eigenständig aus; ein gemeinsam
vorhandener Treffer ist entsprechend ein Treffer in beiden Armen.

Danach erzeugt Evolution drei Folgegenerationen. Zufallssuche erzeugt 24 neue
unabhängige Geometrien mit demselben Generator, ohne Fitness-Rückmeldung.
Die vollständige Evolution wird zuerst ausgeführt, danach der vollständige
Zufallsarm; die Versuchswurzeln werden aufsteigend abgearbeitet. Zufallsarm und
Referenzen bekommen weder Eltern noch Suchhistorie als Eingabe.

Generation 0, erneut bewertete Eliten, Duplikate und fehlgeschlagene
Kandidatenauswertungen verbrauchen jeweils einen Bewertungsslot. Es gibt kein
Fitness-Caching, keine Deduplizierung vor der Bewertung und keinen vorzeitigen
Stopp nach einem Treffer. Ein Bewertungsslot enthält die gesamte festgelegte
Physikdiagnostik, nicht nur eine einzelne Diagonalisierung.

Generatorinterne Konstruktion und Aufzählung zulässiger Mutationen werden
getrennt zeitlich erfasst. Gleiche Bewertungsslots bedeuten nicht automatisch
gleiche Laufzeit. Die einmalige Erzeugung der gemeinsamen Startgeometrien wird
als gemeinsamer Aufwand ausgewiesen; sie wird für einen Armvergleich nicht
einseitig der Evolution zugerechnet. Laufzeiten sind bei dieser festen
Armreihenfolge deskriptiv und erlauben keinen unabhängigen Geschwindigkeitsnachweis.

Evolution behält die Koordinaten ihrer acht anfänglichen Abstammungslinien.
Zufallssuche kann zusätzlich neue Koordinatensätze ziehen. Beide Strategien
unterliegen demselben Ressourcenvertrag, erkunden aber unterschiedlich: Dieser
Versuch vergleicht ihre Gesamtleistung, isoliert nicht den Effekt der Selektion
auf einem identischen festen Koordinatensatz.

## Fitness, Auswahl und Mutation

Die Fitness verwendet drei getrennte Rohwerte in absteigender lexikographischer
Reihenfolge:

1. `clean_eligible` als explizites 0/1-Auswahlziel;
2. `localizer_protection_proxy`;
3. `minimum_boundary_weight_first_four` aus der bestehenden Zustandsordnung.

Die boolesche Gateentscheidung bleibt zusätzlich boolesch im wissenschaftlichen
Datensatz. Ein numerisch abgeschlossenes, wissenschaftlich unzulässiges Ergebnis
kann weiterhin verfügbare Fitness mit erstem Wert 0 besitzen. Operative Fehler
haben keine Fitness. Fehlende Werte werden nicht durch Null, Unendlich oder
erfundene Strafterme ersetzt. Es gibt keine Gewichtung, Normierung oder Rundung
für Fitnessvergleiche. Gleichstand bedeutet Gleichheit aller drei gespeicherten
Werte; die numerischen Physiktoleranzen werden nicht zu neuen Fitness-Toleranzen.

Turniergröße ist 2, `selection_count=8`. Wettbewerber werden innerhalb eines
Turniers ohne Zurücklegen aus verfügbaren Mitgliedern gezogen, Gewinner über
Turniere mit Zurücklegen. Gewonnen hat das lexikographisch beste Mitglied;
vollständige Gleichstände löst der vorhandene explizite PCG64-Turniermechanismus.
Mindestens eine Elite wird erhalten, einschließlich ihrer vollständigen besten
Gleichstandsgruppe in Quellreihenfolge. Eine reine Elitegeneration ist zulässig
und wird vollständig neu bewertet. Es gibt keine Diversitätsquote und keinen
Novelty-Bonus in dieser festgelegten Strategie.

Der Operator `phase10_research_delaunay_edge_swap_v1` verwendet ausschließlich
Kantenaustausch bei festem Koordinatensatz:

1. Rekonstruiere den Delaunay-Kantenpool der Elternkoordinaten mit denselben
   Optionen `Qbb Qc Qz Q12`, ohne Jitter, degenerierte Dreiecke bis zur doppelten
   Fläche `1e-10` ablehnend, und der Längengrenze `1.75` wie der Generator.
   Auch Nachkommen müssen Teilgraphen dieses Pools bleiben.
2. Nach Eliten verbleiben `k` Plätze. Kind `j`, `0 <= j < k`, verwendet genau
   den Gewinner von Auswahlplatz `j` als einzigen Elternteil und
   Validierungsursprung. Andere gezogene Auswahlplätze bleiben im Ledger.
3. Zähle alle Paare aus einer vorhandenen zu entfernenden Kante und einer im
   Elternteil fehlenden Poolkante auf. Sortiere sie nach
   `(removed_source, removed_target, added_source, added_target)`.
4. Behalte genau Austausche, deren fertiger Graph den ganzen Geometrievertrag
   erfüllt. Der temporäre Graph nach bloßem Entfernen muss nicht zulässig sein.
   Diese Prüfung wertet keine Physik oder Fitness aus.
5. Ziehe für jedes Kind einen Operator-Seed als nächstes PCG64-Rohwort aus dem
   Reproduktionsstream. Ein eigener `Generator(PCG64(operator_seed))` wählt mit
   `integers(number_of_legal_swaps)` einen Austausch gleichverteilt aus der
   geordneten vollständigen Liste.
6. Ist die Liste leer, entsteht eine ausdrücklich protokollierte unveränderte
   Kopie (`no_legal_swap`). Sie verbraucht ihren Bewertungsslot. Der Operator-Seed
   wird auch in diesem Fall erzeugt, aber kein Listenindex gezogen.

Die Umsetzung verwendet die vorhandene Rewire-Primitive und stellt danach
explizit die oben definierte sortierte Kantenfolge wieder her. Unveränderte
Kanten behalten Attribute und Orientierung; die neue Kante hat das exakt aus
den Koordinaten bestimmte Displacement und dieselben neutralen Attribute wie
die Generatorkanten. Die Quellgeometrie wird nie verändert. Koordinaten,
Knotenzahl, Randzuordnung und Einbettungsdimension bleiben erhalten.
Generatorprovenienz beschreibt den Vorfahren; Operation, Elternbezug und neuer
exakter Snapshot werden zusätzlich aufgezeichnet.

Dies ist ein Austausch einer Kante, kein gradsequenzenerhaltender
Zwei-Kanten-Swap. Grad 2–4 wird geprüft, die individuelle Gradfolge darf sich
ändern. Knotenedits, Koordinatenmutation, Crossover und Modellparametersuche
gehören nicht zu diesem Operator. Leere Mutationsräume, wiederholte Eltern und
vollständige Elitegleichstände dürfen die Suche stagnieren lassen; ein Neustart
oder erzwungener zusätzlicher Mutant würde das Protokoll ändern.

## Seeds und reproduzierbare Ausführung

Alle Intervalle sind inklusiv. Die neuen Rollenbereiche waren vor Erstellung
dieser Datei in `src`, `tests`, `docs` und README nicht referenziert. Das ist eine
Repositoryprüfung, keine Aussage über unbekannte Rechnungen außerhalb des Repos.

| Rolle | Seeds | Verwendung |
| --- | --- | --- |
| Technischer Vorlauf | `10_799_900..10_799_909` | ausschließlich Entwicklungs-/Vorlaufrollen |
| Gepaarte Hauptversuche | `10_800_000..10_800_031` | 32 Versuchswurzeln |
| Amorphe Referenzen | `10_801_000..10_801_031` | direkte Registry-Generator-Seeds |
| Reservierte spätere Validierung | `10_810_000..10_810_063` | in diesem Lauf unbenutzt |
| Reservierte spätere Bestätigung | `10_820_000..10_820_127` | in diesem Lauf unbenutzt |

Für jede Versuchswurzel erzeugt der deterministische Benchmarkmodus mit einem
lokalen `PCG64(root)` zuerst ein Rohwort als Evolutionswurzel, dann 32 Rohwörter
als Sampler-Seeds in absoluter Slotreihenfolge. Die ersten acht erzeugen die
gemeinsame Startpopulation; die restlichen 24 sind ausschließlich Zufallsarm.
Es werden in diesem Modus keine Evaluations-Seeds gezogen oder in der
Physikprovenienz behauptet. Die bestehende stochastische Benchmarkvariante
behält bei der späteren API-Erweiterung ihre bisherige Seed-Semantik.

Die Evolutionswurzel liefert unverändert zwei Rohwörter pro Übergang:
Selektions-Seed, dann Reproduktions-Seed. Beide werden auch bei voller Elite
gezogen. Der Reproduktionsstream liefert die oben definierten Operator-Seeds.
Jeder abgeleitete Seed wird mit Rolle und Indizes protokolliert. Zufällige
Kollisionen mit reservierten Rollen oder anderen unabhängigen Streams sind
Vertragsfehler; es wird weder nachgezogen noch ein anderer Seed gewählt.
Bewusste gemeinsame Starts und Wiederaufnahme derselben Arbeit sind keine
unabhängigen Streams. Held-out-Seeds werden niemals zu Testdaten umgewidmet.

Die Budgets des bestehenden Generators bleiben unverändert: höchstens
`1_000_000` Punktvorschläge insgesamt und `10_000` vollständige
Konstruktionsversuche pro angefordertem Generator-Seed. Interne Ablehnungen
verwenden denselben fortlaufenden PCG64-Stream und behalten ihre Gründe.
Es gibt keine zusätzliche äußere Wiederholung bei Generatorfehlern.

Python 3.14 und `PYTHONDONTWRITEBYTECODE=1` sind verbindlich. Ausführung erfolgt
sequenziell mit der vorhandenen `.venv`, ohne `uv run`, globalen RNG oder
zeitabhängige Seeds. Vor dem Pythonstart werden `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1` und `BLIS_NUM_THREADS=1` gesetzt.
Die effektive numerische Backendkonfiguration, Paketversionen, Pythonversion,
Betriebssystem, CPU und Threadkonfiguration werden dokumentiert. Diese
Reproduzierbarkeitswahl wird im Vorlauf zeitlich gemessen und nicht anhand
günstiger Suchergebnisse geändert. Keine Vorab-Laufzeitzusage ohne Messung.

## Statistische Auswertung

Die 32 Versuchspaare sind die Einheiten der primären Auswertung. Die 2.048
Bewertungsversuche sind keine unabhängigen Replikationen. Alle folgenden
Auswertungen verwenden das vollständige, versiegelte Ergebnis; eine laufende
Fortschrittsanzeige führt weder Signifikanztests noch Stoppentscheidungen aus.

Für jedes Paar werden die beiden Entscheidungen „mindestens ein Treffer in
32 Slots“ gespeichert. Die vollständige 2×2-Tabelle enthält beide erfolgreich,
nur Evolution erfolgreich (`b`), nur Zufallssuche erfolgreich (`c`) und beide
erfolglos. Der geschätzte Unterschied der Trefferraten ist `(b - c) / 32`.

Der einzige primäre Test ist ein zweiseitiger exakter McNemar-Test mit
`alpha=0.05`. Für `m=b+c > 0` ist der p-Wert
`min(1, 2 * sum(comb(m, j) for j in range(min(b, c) + 1)) / 2**m)`;
für `m=0` wird er als 1 berichtet. Dies ist der binomiale exakte Fall des
[McNemar-Tests](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html)
und kann gegen `scipy.stats.binomtest(b, m, 0.5, alternative="two-sided")`
geprüft werden; siehe die
[SciPy-API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).
Eine zusätzliche statsmodels-Abhängigkeit ist dafür nicht erforderlich.

Nur wenn alle 32 Paare vollständig sind, `b > c` und `p <= 0.05`, darf die
Auswertung einen statistisch gestützten Suchvorteil der Evolution unter diesem
Protokoll melden. Bei `c > b` und demselben Kriterium lautet das Ergebnis
Suchnachteil. Ansonsten lautet es „kein statistisch aufgelöster Unterschied“;
das ist kein Äquivalenznachweis. 32 Paare sind ein begrenztes festes Erstbudget,
keine zugesicherte Teststärke. Die Stichprobe wird nicht anhand des p-Werts
vergrößert und nicht mit früheren Phase-9.8-Läufen gepoolt.

Pro Arm werden Trefferrate und zweiseitiges 95%-Wilson-Intervall über 32 Versuche
mit `z=1.959963984540054` berichtet. Die Differenz wird nicht durch Subtraktion
dieser beiden marginalen Intervalle mit einem vermeintlich gepaarten Intervall
versehen. Die Unsicherheit und der Test gelten bedingt auf das feste Panel;
Referenzpanelvariabilität wird hier nicht unabhängig repliziert.

Sekundär, ohne weitere Signifikanzentscheidung, werden berichtet:

- erster Treffer als einbasierter Slot 1–32 oder `None` und kumulative
  Versuchstrefferkurven nach jedem Slot;
- bestes bisher verfügbares lexikographisches Tripel und bester Schutzproxy
  unter sauberen zulässigen Kandidaten, bei Fehlen ausdrücklich `None`;
- Paarvergleiche dieser Endwerte, getrennt nach Verfügbarkeit und Gleichstand;
- alle Gategründe, operative Fehler, verfügbare Bewertungen und exakten
  Snapshot-Duplikate; Duplikate sind keine zusätzlichen Entdeckungen;
- Zeitaufwand für Generierung, Mutation, Physik und Speicherung, CPU/RAM,
  Eliteanzahl und `no_legal_swap`-Häufigkeit.

Ein Mittelwert der Trefferzeit nur über erfolgreiche Läufe darf nicht als
allgemeiner Suchgeschwindigkeitsvorteil verwendet werden, weil er Fehlläufe
ausblenden würde.

Zur vorab festgelegten Sensitivitätsanalyse werden auf denselben gespeicherten
Ergebnissen Trefferkurven für Faktoren `1.00`, `1.05`, `1.10`, `1.20` berechnet.
Außerdem werden die Rankings einmal mit vertauschter Priorität von Schutzproxy
und Randgewicht sowie als Pareto-Auswahl der beiden Rohwerte innerhalb der
sauberen zulässigen Kandidaten dargestellt. Dies sind deskriptive
Neuauswertungen bereits beobachteter Kandidaten, keine simulierten alternativen
Suchverläufe. Sie dürfen die primäre Entscheidung oder die unten definierte
Kandidatenliste nicht ersetzen. Eine Untersuchung alternativer Suchfitness
benötigt einen neuen eingefrorenen Versuch mit neuen Such-Seeds.

## Fehler, Versiegelung und Wiederaufnahme

Gewöhnliche numerische Invalidität, Evaluatorausnahmen und Fehler beim Aufbau
der Fitness bleiben als getrennte Fehlerkategorien im vollständigen Slot-Ledger;
der Slot zählt als erfolglos. Wissenschaftliche Gatefehler eines numerisch
abgeschlossenen Ergebnisses werden getrennt davon gespeichert.

Generatorfehler nach dem vorhandenen internen Budget, ungültige Nachkommen,
Provenienzwidersprüche, fehlende erforderliche Seed-/Vertragsdaten oder ein
Selektionsabbruch stoppen die Kampagne als unvollständig. Dazu zählt eine
Generation mit zu wenigen verfügbaren Mitgliedern für das Zweierturnier, wenn
noch Nachkommen benötigt werden. Das Turnier wird nicht heimlich verkleinert.
Ein abgebrochener Versuch darf nicht entfernt, ersetzt oder als vollständig
erfolgloses 32-Slot-Paar erfunden werden. Ein solcher Präfix erhält keinen
primären Überlegenheitsbefund.

Kandidateneingaben werden vor ihrer Bewertung erfasst, Ergebnisse unmittelbar
danach. Referenzpanel und vollständige Versuchspaare werden atomar versiegelt.
Eine neue, explizit versionierte Kampagnenhülle um die bestehenden Phase-10-
Ledgers und Checkpoints enthält mindestens:

- Protokoll-ID/-Hash, Code-Commit, Dateihashes, komplette Einstellungen und
  Umgebung sowie tatsächlichen Vorlauf-Status;
- alle Seedrollen, Slot-/Generationsindizes, Eltern, Operationen, Legal-Swap-
  Anzahl, gewählten Swap und neue exakte Geometrie-IDs;
- verlustfreie Geometrien, vorhandene Evaluationsdaten, vollständige
  Topologiegrids, Warnungen, Randzustandsdaten und Fehler;
- Referenzpanel, `R`, jede Trefferentscheidung, Rankings, Statistik und
  deskriptive Sensitivitätsauswertung;
- Fortschrittsereignisse, letzter versiegelter Zustand und Unterbrechungen.

Die Mindestgarantie für `--resume` ist Wiederaufnahme nach dem letzten komplett
versiegelten Versuchspaar. Ein bereits versiegeltes Panel oder Paar wird nicht
neu ausgewertet. Für ein unterbrochenes Paar wird der vorhandene Präfix unter
seiner Ausführungsnummer erhalten; dasselbe gesamte Paar darf mit denselben
Seeds wiederholt werden. Diese Wiederholungsarbeit zählt getrennt zur realen
Rechenzeit, nicht als neues unabhängiges Versuchspaar. Eine bereits gespeicherte
operative Fehlerentscheidung darf durch Resume nicht nachträglich zum Treffer
werden. Solche Ergebniswidersprüche verhindern die Versiegelung.

Resume verändert weder Zielbudget noch Versuchsliste. Protokoll, Code, Pakete
und numerische Umgebung müssen zur gespeicherten Ausführung passen. Defekte
Dateien, abweichende deterministische Resultate oder ein wissenschaftlich
motivierter Wechsel der Umgebung erlauben keinen stillen Neustart. Änderungen
nach sichtbaren Ergebnissen werden als Amendment mit Begründung und betroffenen
frischen Seeds festgehalten. Ein Stromausfall allein erzeugt keine neue
wissenschaftliche Version, sofern unveränderte Arbeit auditierbar fortgesetzt wird.

## Vorlauf, Bedienung und Ergebnisdateien

Der geplante neue Einstieg heißt `toposc phase-10-research`. Er existiert am
Ausgangscommit noch nicht; die folgenden Aufrufe beschreiben den später zu
implementierenden Bedienvertrag, keine aktuell startfähigen Befehle.

`--preflight --output results/phase_10_research_v1` verwendet ausschließlich die
Vorlaufrollen. Die Seeds `10_799_900` und `10_799_901` führen jeweils ein Paar
mit denselben `N=8`, drei Übergängen und Physikgates aus. Das deterministische
Quadrat dient nur diesem Vorlauf als Schwellenreferenz mit Faktor `1.10`.
Die übrigen acht Seeds
`10_799_902..10_799_909` prüfen Generierung und Mutationsraum ohne Physikauswertung.
Der Vorlauf umfasst somit 129 Physikbewertungsversuche plus acht Geometriechecks;
er ist kein Bestandteil der 2.083 Hauptlaufslots.

Bestehen verlangt konsistente Eingaben, vollständige technische Ledgers,
Zulässigkeit des Quadrat-Referenzchecks, korrekte deterministische Auswertung,
funktionierende Speicherung und mit Tests belegte Unterbrechungs-/Resume-
Semantik. Eine Mindesttrefferzahl oder ein Evolutionsvorteil ist ausdrücklich
kein Bestehenskriterium. Aufwand und ETA werden aus diesem Vorlauf geschätzt.
Vorlaufergebnisse dürfen nicht zum Tuning des Hauptversuchs verwendet werden.

`--full --output results/phase_10_research_v1` erfordert den passenden bestandenen
Vorlauf und startet ausschließlich den hier definierten sauberen Suchvergleich.
`--resume --output results/phase_10_research_v1` setzt den eigenen unterbrochenen
Lauf nach obigen Regeln fort. Der Hauptlauf verwendet einen neuen `full`-Bereich
im Outputverzeichnis und überschreibt keinen vorhandenen Lauf. Der Vorlauf
bleibt unter `preflight` eindeutig getrennt.

Die Konsole zeigt Phase, Versuch `i/32`, Arm, Generation `0..3`, Slot `1..32`,
abgeschlossene und verbleibende Arbeit, bisherige Treffer, Zeit/ETA, CPU/RAM und
letzten versiegelten Stand. Auch während längerer Generierung oder Auswertung
gibt es Heartbeat-Ereignisse, ohne einen fertigen Slot vorzutäuschen. Die
maschinenlesbare Ereignisdatei heißt `events.jsonl` im Outputverzeichnis und
kann später in einer zweiten PowerShell verfolgt werden:

```powershell
Get-Content results\phase_10_research_v1\events.jsonl -Wait
```

Der fertige Lauf liefert eine lesbare `report.md`, `summary.json`, das
unveränderliche Panel, vollständige Paar-/Kandidatenartefakte und Graphiken zu
Trefferkurven und vorab ausgewählten Kandidaten. Graphiken entstehen aus
gespeicherten Ergebnissen ohne erneute Physikauswertung.

Nach Abschluss wird pro erfolgreichem Arm und Versuch dessen bester starker
Kandidat nach der festgelegten lexikographischen Ordnung nominiert. Diese
Nominierungen werden pro Arm gleichartig geordnet, bei vollständigem Gleichstand
nach exakter Geometrie-ID und dann `(trial_index, attempt_index)`. Pro Arm
werden höchstens acht unterschiedliche exakte Geometrie-IDs als Kandidatenliste
versiegelt. Eine armübergreifend identische Geometrie behält beide Fundrollen.
Die Graphfingerprints gelten nicht als physikalische Deduplizierungsidentität.
Bei null Treffern ist eine leere Liste ein gültiges Ergebnis.

Diese Liste ist nur eine nachvollziehbare Vorauswahl für spätere Forschung.
Es laufen hier keine Disorder-Ensembles, keine Bestätigung und kein Größen-
Scaling. Die reservierten Seeds bleiben unberührt. Ein separates Protokoll
muss vor deren Verwendung Kandidatenauswahl, Kanäle, Stärke, Vergleich und
Entscheidungsregeln festlegen; Phase-9.8-Validierung wird nicht automatisch
mitgestartet.

Alle neuen Artefakte liegen im neuen Phase-10-Verzeichnis. Die gelöschten
Phase-9.8-Ergebnisordner werden weder benötigt noch wieder angelegt. Die
geschützten Observables-Bytecode-Dateien und `geometry_demo.npz` bleiben
unverändert. Commit und Push bleiben Nutzeraktionen.

## Architekturprüfung und begrenzter Umsetzungsbedarf

Die Prüfung des Ausgangsstands ergibt folgende Wiederverwendung und Lücken:

| Bereich | Vorhanden und erforderliche Ergänzung |
| --- | --- |
| Geometrie/Physik | Registry, Genome, Rewire und `validate_phase_9_8_geometry` sowie `evaluate_phase_9_8_primary_geometry` wiederverwenden; keine neue Physikformel |
| Benchmark-Provenienz | `BenchmarkEvaluator` und Audit verlangen aktuell `int`-Seeds; einen expliziten deterministischen Modus mit `None` und passender Versionierung ergänzen |
| Fitness/Selektion | Bisher skalar oder Pareto; lexikographische Definition, Turniere, vollständige Elitegleichstände und Codec-Unterstützung kohärent ergänzen |
| Wissenschaftliche Rohwerte | Phase-9.8-Bundle vollständig behalten; drei explizit benannte abgeleitete Ziele ohne Überschreiben vorhandener Observablen an die Fitness anbinden |
| Mutationsraum | Deterministischen Delaunay-Pool und legalen Austausch als versuchsspezifischen Operator anbinden; gemeinsame Generatorlogik bei Bedarf gezielt öffentlich machen |
| Laufhülle | Kampagnenpersistenz, Ereignisse, Bericht und Resume fehlen beim Benchmark; vorhandene Generationen-Checkpoints und CLI-Monitoring gezielt wiederverwenden |

Der neutrale Genomvertrag bleibt erhalten: Geometrie beschreibt Struktur,
Koordinaten, Orientierung und Randmetadaten; Physik, Fitness und Trefferstatus
gehören in Evaluations-/Suchobjekte. Eine Anbindung über explizite abgeleitete
Evaluationsfelder muss ihre physikalische Herkunft bewahren; wissenschaftliche
Werte dürfen nicht als erfundene reine Graphdeskriptoren ausgegeben werden.

Die allgemeine skalar/Pareto-Semantik und der bisherige stochastische
Benchmarkpfad bleiben kompatibel. Neue Fitnessdefinitionen müssen durch alle
betroffenen Konstruktoren, Auditprüfungen und Checkpoint-/Resume-Codecs getragen
werden. Versionsänderungen brauchen Regressionstests und klare Ablehnung
inkompatibler gespeicherter Zustände. Allgemeine Dataset-, ML- oder autonome
Discovery-Infrastruktur ist für diese Laufhülle nicht erforderlich.

Vor dem startbaren Lauf sind fokussierte Vertragstests, vollständiges `pytest`,
projektweites Ruff, relevantes striktes mypy mit `--python-version 3.14`,
`git diff --check` und `git status` erforderlich. Tests laufen mit Python 3.14,
`PYTHONDONTWRITEBYTECODE=1`, lokalem explizitem pytest-Temporärverzeichnis und
`-p no:cacheprovider`. Wissenschaftliche Haupt-/Referenz-/Held-out-Seeds werden
in Tests nicht physikalisch ausgewertet. Dokumentation allein benötigt keine
erneute Physiktestsuite.

Die Umsetzung muss insbesondere paarweise identische Starts, exakte Budgets
trotz Eliten/Duplikaten/Fehlern, Seedrollen, richtige lexikographische Ties,
gültige Austausche, leere Mutationsräume, bekannte exakte Statistikfälle,
vollständige Rohwertpersistenz und Wiederaufnahme ohne selektive Wiederholung
prüfen. Technischer Vorlauf und vollständiger Forschungslauf bleiben explizite
Nutzerstarts nach dieser Umsetzung.
