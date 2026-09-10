# Lokale Suche nach der Messvalidierung: Entscheidungsentwurf v1

## Status und Herkunft

Entwurf vom 2026-09-10 auf Basis von
`74cc5e7e2e7d2b2887721a628bd38c3649e0c12d`.
Dies ist die Architekturprüfung für ein separates Suchprotokoll, noch kein
eingefrorener Versuchsplan und keine Startfreigabe. Keine neuen Physikrechnungen,
keine Änderung bestehender Gates, keine Phase 11, kein ML.

Grundlagen: [Forschungscharter](pre_phase_9_research_charter.md),
[Messvalidierungsprotokoll](pre_phase_10_measurement_validation_protocol_v1.md),
[Messvertragsentwurf](phase_10_measurement_contract_draft_v1.md) und
[wiederhergestellter Bericht](../../results/phase_10_measurement_validation_v1_storage_fix1_report_fix1/report.md).
Die zugehörige `summary.json` hat SHA-256
`59945c710b52de47c4fb694cb7a9abaf54ab6459948f2849c807334df06ded47`.
Die ursprünglichen Archive und der separat versiegelte Wiederherstellungsbericht
bleiben unverändert. Der Masterplan bleibt ein Entwicklungsfahrplan; dieser
Schritt setzt das Forschungsprogramm innerhalb Phase 10 fort.

## Was die vorhandenen Daten tatsächlich erlauben

MEAS-VAL-001 akzeptiert den begrenzten Messvertrag: Kontrollen bestehen,
zwölf mu=12-Varianten sind korrekt trivial, zwölf mu=2-Varianten bestehen das
positive Screening und alle sechs zentralen Größenunterschiede bleiben unter
0.005. Für die mu=2-Eingriffe existiert kein separat bewiesenes Soll-Label;
das Screening ist nicht selbst eine unabhängige Bestätigung ihrer Topologie.

Eine zusätzliche, nachträgliche Gegenüberstellung der bereits gespeicherten
L-Werte mit der jeweiligen topologischen Anfangskontrolle ergibt:

| Quadratgröße | L des unveränderten Quadrats | L der sechs Varianten, Minimum bis Maximum | Varianten über Quadrat |
| --- | ---: | ---: | ---: |
| 16×16 | 0.738933984 | 0.668380691 bis 0.737406372 | 0/6 |
| 20×20 | 0.948221779 | 0.706669001 bis 0.943012950 | 0/6 |

L ist jeweils das Minimum der drei Localizer-Gaps bei kappa=0.1,0.2,0.3.
Die Referenzwerte stammen aus `controls[].start.methods.localizer[].gap`, die
Variantenwerte aus `variants[].localizer_protection_proxy` der genannten Summary.
Dies ist eine deskriptive Nachauswertung, kein neuer vorregistrierter Test.
Verglichen wird innerhalb derselben Größe, nicht L von 16×16 mit L von 20×20.
Das positive Messurteil bleibt bestehen; es bescheinigt keinen Schutzvorteil.

## Vorgeschlagene nächste Frage

> Gibt es unter vorab begrenzten lokalen, grad- und ressourcenkontrollierten
> Zwei-Kanten-Umverdrahtungen des offenen Quadrats einen Kandidaten, der das
> festgelegte topologische Screening erhält und den Localizer-Schutzproxy L
> gegenüber dem unveränderten Quadrat relevant erhöht?

Nullhypothese für diesen engen Versuch: Kein zulässiger Kandidat überschreitet
die noch vor neuen Physikdaten festzulegende Mindestverbesserung gegenüber dem
Quadrat. Auch ein leerer Trefferbestand ist ein gültiges Ergebnis. Die Daten
liefern bisher keinen Anlass, einen positiven Ausgang zu versprechen.

Das ist eine lokale Machbarkeitsfrage, nicht die vollständige Charterfrage nach
Disorder-Robustheit gegenüber den stärksten benannten Referenzfamilien.
Evolution gegen Zufall ist eine zweite, algorithmische Frage. Sie wird erst
sinnvoll, wenn der definierte Suchraum ausreichend viele unterschiedliche
Kandidaten enthält und ein begrenztes Suchbudget begründet ist.

## Minimaler Kandidatenraum für die nächste geometrische Prüfung

Vorgeschlagener Dry-Run, ausdrücklich ohne Hamiltonian oder Solver:

- Offene Quadrate n=16,20; dieselben sechs 16-Site-Patches je Größe wie in
  MEAS-VAL-001 (boundary/interior und Offsets -2,0,2).
- Jeder Kandidat startet vom unveränderten Quadrat und ersetzt genau zwei
  vorhandene Einheitskanten mit vier verschiedenen Endpunkten innerhalb eines
  solchen Patches durch eine andere Paarung derselben vier Endpunkte.
- Die zwei neuen Kanten müssen vorher fehlen und beide Länge sqrt(2) haben.
  Damit haben alle Varianten denselben geänderten Kantenlängenbestand.
- Koordinaten, Site-IDs, Rand, Knoten-/Kantenanzahl und jeder einzelne Knotengrad
  bleiben unverändert. Zusammenhang, Kreuzungsfreiheit, maximale Kantenlänge
  1.75 und die übrigen Ressourcenregeln werden vollständig geprüft.
- Genau ein Eingriff; keine aufeinanderfolgenden Mutationen, keine kombinierte
  Defektpopulation, keine bewegten Knoten oder Modellparameteroptimierung.
- Exakte Duplikate durch überlappende Patches zusammenführen, dabei sämtliche
  Entstehungswege erhalten. Symmetrien gesondert kennzeichnen; ein Graphhash
  oder eine Rotation allein beweist keine physische Gleichwertigkeit.

Andere Paarungen im selben Patch sind eine explizite lokale Erweiterung des
bisher geprüften Musters. Sie sind nicht schon durch MEAS-VAL-001 validiert.
Ein gültiger Geometriecheck garantiert weder Screening noch Messanwendbarkeit.
Wenn der Katalog nur bekannte oder symmetrieverwandte Fälle enthält, endet
dieser Vorschlag ohne neuen Physiklauf. Er wird nicht automatisch erweitert.

Ergebnis des Dry-Runs sollen Kandidatenzahl, exakte IDs, Ablehnungsgründe,
Patchzugehörigkeit, bekannte Fälle und mögliche Symmetriebeziehungen sein.
Der kleine Katalog ist ein Experimentinventar, keine Phase-11-Datasetfunktion.
Bei überschaubarem Inventar ist vollständiges Durchprüfen die vorgeschlagene
Referenzoption; es erlaubt keine Aussage über einen evolutionären Suchvorteil.

## Messung, Ranking und faire Vergleiche

1. Modell für die Leistungsfrage bleibt hopping=1, mu=2, pairing=1 mit denselben
   Orts-, Kanten- und Nambu-Konventionen. mu=12 und H=0 dienen nur Kontrollen;
   ihre L-Werte gehen nie ins Leistungsranking ein.
2. Primäre zentrale Region und physischer Rand werden am Ausgangsquadrat fixiert.
   Bott-/Localizer-Panel, PH-/Randdiagnostik, fünf Chern-Masken und lokale
   Markeränderungen bleiben einzeln sichtbar. Kandidaten bestimmen ihre
   Messregion nicht nach erreichten Werten. Alte Screeninglabels bleiben erhalten.
3. Nur Kandidaten mit positivem Primärscreening sind rankingfähig. Danach ist
   `delta_L = L(candidate) - L(square gleicher Größe)` die vorgeschlagene
   Rankinggröße. Chern-Nähe an 1 wird nicht maximiert. Mindestverbesserung,
   Gleichstandstoleranz und numerische Sensitivitätsprüfung sind noch einzufrieren.
4. Die Quadratkontrolle ist die Baseline dieses lokalen Versuchs, nicht die
   stärkste Referenz der gesamten Charter. Zwei längere Kanten statt zweier
   Einheitskanten bleiben ein offengelegter Ressourcenunterschied zum Quadrat;
   gleiche Kantenanzahl allein beweist keine universelle physische Fairness.
   Alle Eingriffsvarianten teilen dieselbe Längenänderung und Kopplungspolitik.
5. Bei einem später begründeten Evolution-Zufall-Vergleich: identische zulässige
   Menge, Startpopulationen, physikalische Einstellungen und Versuchsbudgets.
   Startpopulationen, Eliten, Wiederholungen, ungültige Vorschläge und Fehler
   werden nach einer vorab fixierten Zählregel behandelt; kein kostenloses
   Nachziehen günstiger Ersatzkandidaten. Physikaufrufe und Vorschläge getrennt
   berichten; Cache-Regel, Seeds, Paarung und statistische Auswertung einfrieren.
6. Ohne rankingfähige Kandidaten bleibt das Ergebnis ausdrücklich leer; kein
   ausgedachter L-Wert für fehlgeschlagene Methoden. Ein hoher L-Wert ohne
   Screening ist kein Treffer. Operative Fehler bleiben in ihren geplanten Nennern.

## Architekturgrenzen und noch offene Freeze-Punkte

Vorhandene Geometry-/Genome- und Validierungs-APIs wiederverwenden. Ein separater
Katalog-Builder beschreibt die neue Menge; `build_measurement_plan` und der
eingefrorene 36-Slot-Runner bleiben unangetastet. Der vorhandene
`validate_measurement_geometry` prüft nicht allein, ob exakt die geforderte
Zwei-Kanten-Operation, identische Koordinaten und die standortweise Gradfolge
vorliegen; diese Bedingungen muss der Katalog zusätzlich belegen.

Ein späterer Evaluator kann bestehende Methodenaufrufe und Speicherbausteine
verwenden, darf aber keine alte Protokoll-ID für neue Inputs benutzen. Für ein
neues Experiment sind Code-/Protokollrevision, Ausgabeordner, Vorlauf, sichere
Wiederaufnahme, Bericht und Nutzerstart separat festzulegen.

Vor einer Physikfreigabe fehlen: geprüfte Kataloggröße und Symmetriepolitik,
Entscheidung Enumeration oder Suchbenchmark, feste Budgets und gegebenenfalls
Seedlisten, Schwelle für relevanten L-Gewinn, Sensitivitätsregeln, Kontrollbudget,
Abbruchregeln und eine von Auswahl getrennte Bestätigungspolitik.
Diese Liste ist kein bereits ausführbares Protokoll.

Die heute bekannten n=16/20-Fälle sind für diese neue Hypothesenbildung
Entwicklungsdaten. Eine erneute Auswertung macht sie nicht zu unabhängigen
Holdouts. Neue Platzierungen derselben Muster sind ebenfalls keine unabhängigen
Familien. Disorder-Seeds oder größere Bestätigungssysteme werden erst in einem
gesonderten Plan reserviert, ohne sie bei der Auswahl anzusehen. L bleibt ein
Proxy, bis unabhängige Robustheitsprüfungen eine weitergehende Aussage tragen.

Nächster begrenzter Arbeitsschritt: den beschriebenen Katalog geometrisch
prüfen und zählen. Keine Physikrechnung und kein automatischer Übergang zu
einem größeren Suchraum. Danach wird über das numerische Protokoll entschieden.
