# Lokale Geometrie-Enumeration: Protokollentwurf v1

## Status, Herkunft und Frage

Protokoll-ID: `TOPOSC-P10-LOCAL-ENUM-001`, Entwurf vom 2026-09-10.
Katalogrevision: `32e4f1ef33b63be0b9b617a9d43880d8b8be24f5`.
Grundlagen sind die [Forschungscharter](pre_phase_9_research_charter.md), der
[Suchentwurf](phase_10_local_search_design_draft_v1.md) und das
[Messvalidierungsprotokoll](pre_phase_10_measurement_validation_protocol_v1.md).

> Enthält der vollständig festgelegte lokale Geometriekatalog Kandidaten,
> die das primäre Screening erhalten und den Localizer-Schutzproxy gegenüber
> dem gleich großen unveränderten Quadrat um mindestens 1 % verbessern?

Dieser Text wird nach Kenntnis der bisherigen Kampagnen festgelegt. Die zwölf
bereits gemessenen Katalogfälle sowie die Quadratkontrollen sind bekannt.
Alle n=16/20-Ergebnisse dieses Versuchs gehören zur Kandidatensuche. Ein
größenübergreifender Befund aus dieser Suche ist keine unabhängige Bestätigung.

Dieses Dokument legt den vorgeschlagenen numerischen Vertrag vollständig fest.
Sein separater Commit durch den Nutzer friert ihn ein. Danach folgen ein eigener
Runner, dessen Tests und ein Implementierungscommit. Erst anschließend startet
der Nutzer Vorlauf und Hauptlauf. Aktuell existiert nur der geometrische
Katalogbefehl; dieser Text führt keine neuen Physikrechnungen aus.

## Kandidateninventar und Ressourcen

Verwendet wird genau die Ausgabe von `build_local_rewire_catalog()` an der
genannten Revision: 128 Kandidaten auf 16×16 und 128 auf 20×20. Die Reihenfolge
ist n aufsteigend, darin lexikographisch nach `(removed_edges, added_edges)`
mit sortierten, aufsteigend orientierten Site-Paaren. Der SHA-256 der durch
Zeilenumbrüche verbundenen exakten Geometrie-IDs in dieser Reihenfolge ist
`13c9e2c750af250f20fd3d245d48b6abc9b2936176e5187dd76ba245ff1c6be0`.
Dies ist die Katalogreihenfolge, keine lexikographische Sortierung der Hashtexte.

Jede Geometrie ersetzt am Ausgangsquadrat genau zwei Einheitskanten mit vier
verschiedenen Endpunkten durch zwei zuvor fehlende Diagonalen der Länge sqrt(2).
Alle Endpunkte liegen in mindestens einem der sechs vorab definierten
16-Site-Patches. Pro Größe bleiben Knotenanzahl n², Kantenanzahl 2n(n−1),
Koordinaten `(x,y)`, Site-ID `x*n+y`, jeder Knotengrad, Gitterabstand 1 und
der physische Rand mit 4n−4 Sites unverändert. Die Patches und ihre Herkünfte
werden vollständig übernommen. Kataloggültigkeit garantiert noch kein Screening.

Vor der ersten Physikbewertung werden alle Genome, IDs, Kantenänderungen,
Herkünfte und Messregionen gespeichert und geprüft: einfacher zusammenhängender
Graph, unveränderte Koordinaten und Gradfolge, Kreuzungsfreiheit, Mindestabstand
0.55, maximale Kantenlänge 1.75, Geometrietoleranz 1e-12 und die neutralen
Darstellungsdaten aus MEAS-VAL-001. Der neue Runner prüft sowohl den
Referenz-Delta-Vertrag als auch den vollständigen Messgeometrie-Validator.
Kein zusätzlicher Kandidat, Ersatz oder Reparaturschritt ist vorgesehen.

Gleiche Geometrien aus mehreren Patches werden einmal physikalisch bewertet;
alle lokalen Patchdiagnosen bleiben erhalten. C4-/D4-Signaturen werden nur
angezeigt. Keine Symmetriereduktion der 256 Bewertungen, keine daraus
abgeleiteten unabhängigen Stichprobenzahlen. Bei Patchherkunft `boundary` kann
die tatsächliche Eingriffstiefe größer als null sein: tatsächliche Koordinaten,
Randtiefe und Überlappung mit der zentralen Region werden separat berichtet.

Der Modellcode setzt Hopping und Pairingbetrag pro Kante unabhängig von deren
Länge fest. Damit teilen die Varianten das Kopplungsbudget des Quadrats; ihre
gesamte Kantenlänge ist um 2(sqrt(2)−1) größer. Dieser Unterschied bleibt
sichtbar. Die Quadratbaseline dient der lokalen Proxyfrage. Ein Vorteil
gegenüber den stärksten benannten Referenzfamilien der Charter wird hier nicht geprüft.

## Modell, Methoden und numerische Prüfung

Alle Kandidaten verwenden `ChiralPWaveModel` mit hopping=1, mu=2, pairing=1,
Chiralität +1, Ortsachsen (0,1), komponentenweiser Nambu-Basis und Energie 0.
Kontrollrollen am unveränderten Quadrat sind `topological` mit diesen Werten,
`trivial` mit hopping=1, mu=12, pairing=1 und `undefined` mit allen drei Werten 0.
Die Kontrollbegründungen und Kopplungskonventionen aus MEAS-VAL-001 gelten weiter.
Es gibt keine Disorderrealisierungen und keine Zufallsseeds; Seedfelder sind null.

Pro Slot wird ein vollständiger Eigenpaarsatz mit `numpy.linalg.eigh` bestimmt.
Hamiltonian, Eigenwerte, Eigenvektoren und 16 Niedrigenergiezustände für die
gapped Rollen werden archiviert. Allgemeine numerische Toleranz: 1e-10.
Zusätzliche numerische Abnahme, mit D als Hamiltoniandimension:

- alle Matrizen, Eigenwerte und Methodenrohwerte endlich;
- `max(abs(H-H†)) <= 1e-10`;
- `||H V - V diag(E)||_F / max(1, ||H||_F) <= 1e-10`;
- `||V† V-I||_F / sqrt(D) <= 1e-10`.

Diese Residuen prüfen den Solver, sie beweisen keine Fehlergrenze der
topologischen Diagnose. Ein Verstoß ist `numerical_failure`, erhält keinen
Trefferstatus und bleibt mit seinen Rohwerten im Bericht.

Das primäre Methodenpanel bleibt exakt:

- drei Bott-Berechnungen mit Perioden (0.95n,0.95n), (n,n), (1.05n,1.05n),
  Quantisierungstoleranz 1e-6;
- drei Localizer bei Probe `((n-1)/2,(n-1)/2)`, Energie 0 und
  `K_primary=(0.1,0.2,0.3)`; `L=min(gap(k) für k in K_primary)`;
- fünf Chern-Masken in Reihenfolge graph_depth_2, graph_depth_3, fixed_depth_2,
  fixed_depth_3, central_fraction; Quantisierungstoleranz 0.005;
- positive Voronoi-Flächen in `[-0.5,n-0.5]²`, pro Site 1, Summe n² innerhalb
  absoluter Toleranz 1e-10 und eindeutige Orts-/Site-/Nambu-Zuordnung.

Die primäre zentrale Region hat ursprüngliche Koordinaten-Randtiefe >=n/4;
sie enthält 36, 64 bzw. 100 Sites für n=12,16,20. Die festen Tiefenmasken
verwenden ursprüngliche Tiefe >=2/3; die Graphmasken Graphdistanz >=2/3 zum
festen physischen Rand. Kandidaten verändern die primäre Site-Menge nicht.
Die vier zusätzlichen Masken bleiben sichtbare Sensitivitätsdiagnosen.

Zusätzlich werden für jeden Slot, unabhängig vom primären Ergebnis, sechs
Localizer berechnet: `K_extra=(0.09,0.11,0.18,0.22,0.27,0.33)` am selben Probeort.
Das sind ±10 % um die drei primären kappa-Werte. `L_extra` ist ihr Minimum.
Die Zusatzwerte verändern weder L noch das bisherige Primärscreening.
Sie prüfen eine endliche lokale Einstellungssensitivität; sie sind keine
Disorderprüfung und kein Test eines kontinuierlichen kappa-Intervalls.

Alle 17 Methodenaufrufe (3 Bott, 5 Chern, 9 Localizer) werden einzeln mit
Ergebnis oder konkretem Fehler erfasst. Nach einem fehlgeschlagenen H-Aufbau
oder Solver werden abhängige Methoden als nicht ausführbar markiert. Ein
einzelner Methodenfehler verhindert nicht die übrigen möglichen Aufrufe.
Die Anzahl interner Diagonalisierungen wird separat gemessen: 17 Methodenaufrufe
bedeuten nicht 17 oder nur einen tatsächlichen Solveraufruf.

## Klassifikation und Treffer

`min(abs(E)) <= 1e-10` ergibt `undefined_fermi_projector`. Sonst müssen die drei
Bott-Indizes, die drei invertiblen primären Localizer-Indizes und der quantisierte
zentrale Chern-Index gemeinsam +1 oder gemeinsam 0 sein, um
`primary_topological` bzw. `primary_trivial` zu erhalten. Andere Fälle sind
`mixed_or_unresolved`; unerwartete gemeinsame Integer werden explizit benannt.
Fehlende Werte sind weder Index 0 noch L=0. Abweichende Zusatzmasken setzen
`regional_sensitivity` und behalten ihre Rohwerte.

Positives Primärscreening erfordert `primary_topological`, L>=0.20,
PH-Paarresiduum <=1e-8 und mindestens vier der acht Zustände niedrigster
Absolutenergie mit physischem Randgewicht >=0.80, wie in MEAS-VAL-001.
Numerische Abnahme ist zusätzlich Voraussetzung aller neuen Trefferlabels.

Vergleichsreferenz ist immer die topologische Anfangskontrolle derselben Größe.
Festgelegt werden `r_min=0.01` und eine absolute Entscheidungszone `eps=1e-8`
in den bestehenden Energieeinheiten. Die 1-%-Schwelle ist eine bewusst gewählte
operative Mindestverbesserung, keine universelle physikalische Grenze. Sie wird
nach den bekannten negativen L-Vergleichen, vor neuen Kandidatenphysikdaten
festgelegt und im laufenden Versuch nicht angepasst.

Für jede numerisch gültige, primär screeningfähige Geometrie:

```text
delta_L = L_candidate - L_reference
relative_gain = delta_L / L_reference
margin = delta_L - 0.01 * L_reference
```

Die Referenz muss positiv screeningfähig sein, deshalb ist L_reference>0.
Ein `primary_proxy_hit` erfordert margin>eps. Bei abs(margin)<=eps lautet
das Schwellenurteil `threshold_borderline`; bei margin<−eps ist die Schwelle
nicht erreicht. Keine Rundung vor der Entscheidung.

Ein `sensitivity_qualified_hit` erfordert darüber hinaus:

1. alle sechs zusätzlichen Localizer verfügbar, invertibel und Index +1;
2. für alle neun kappa-Werte `gap_candidate(k)-gap_reference(k) >= -eps`;
3. `L_extra_candidate - L_extra_reference - 0.01*L_extra_reference > eps`.

Jeder nicht erfüllte Punkt wird einzeln berichtet; Werte in der zusätzlichen
Schwellenzone ±eps gelten ebenfalls als grenznah und nicht als qualifiziert.
So bleibt sichtbar, ob ein primärer Vorteil nur bei bestimmten Einstellungen
auftritt oder auf eine Verschlechterung anderer Panelwerte trifft.

Ranking ausschließlich innerhalb derselben Größe unter den numerisch gültigen,
primär screeningfähigen Kandidaten: delta_L absteigend. Die oberste
Gleichstandsgruppe enthält alle mit Abstand <=eps zum maximalen delta_L.
IDs sortieren nur die Anzeige innerhalb der Gruppe. Trefferflags bleiben
separat; bei leerer Menge ist das Ranking leer. Keine Auswahl nach Chern-Nähe.

## Größenpaarung und lokale Diagnose

Für jede 16×16-Geometrie ist eine feste 20×20-Fortsetzung definiert: Alle
Endpunkte der entfernten und hinzugefügten Kanten erhalten y'=y+2. Für
boundary-Herkünfte gilt x'=x, für interior-Herkünfte x'=x+1. Danach Site-ID
`20*x'+y'`, Paare kanonisch aufsteigend. Die Katalogprüfung am 2026-09-10 hat
eine Bijektion der 128 Paare bestätigt; alle Patch-Herkünfte eines Kandidaten
haben denselben Arm und erhalten Offset und Arm unter dieser Fortsetzung.
Der Runner prüft die vollständige Zuordnung vor Physikbeginn erneut.

Ein `paired_proxy_candidate` erfordert `sensitivity_qualified_hit` an beiden
Größen und `abs(C_central_20-C_central_16)<=0.005`. Berichtet werden alle
128 Paare einschließlich fehlender, gemischter und negativer Befunde, nicht
nur die Paarungen der jeweiligen Rangbesten. L selbst wird nicht über Größen
gleichgesetzt. Die größere Mittelungsregion und sinkende Defektdichte bleiben
Teil der Interpretation; der Größenwert 0.005 ist eine operative Übernahme
für diese beschränkte Fragestellung und kein Konvergenzbeweis.

Je Kandidat: vollständiges Markerfeld, Chern-Werte aller fünf Masken, alle neun
Localizerwerte, Bott-/Rand-/PH-Ergebnisse und Änderungen gegen das gleich große
Anfangsquadrat. Für jede gespeicherte 16-Site-Patchherkunft signierter und
absoluter Mittelwert der Markeränderung; zusätzlich nach jeder ursprünglichen
Randtiefe. Kein Zusammenwerfen überlappender Patches als unabhängige Beobachtungen.
Graphmaskenänderungen behalten hinzugekommene/entfernte Sites und Beiträge.

## Festes Budget, Kontrollen und Reihenfolge

| Stufe | Quadratische Kontrollen | Eingriffsslots | Gesamt |
| --- | ---: | ---: | ---: |
| preflight, n=12 | 3 Rollen jeweils Anfang/Ende = 6 | 2 bekannte mu=2-Eingriffe | 8 |
| full, n=16 | 3 Rollen jeweils Anfang/Ende = 6 | 128 Katalogkandidaten, mu=2 | 134 |
| full, n=20 | 3 Rollen jeweils Anfang/Ende = 6 | 128 Katalogkandidaten, mu=2 | 134 |
| Gesamt | 18 | 258 | 276 |

Vorlauf: Anfangskontrollen topological, trivial, undefined; bekannte
MEAS-VAL-Eingriffe mit offset=0 in Reihenfolge boundary, interior; Endkontrollen
in derselben Rollenreihenfolge. Diese zwei n=12-Fälle testen auch lokale
Markerberichte und Eingriffsrouting. Der Vorlauf verlangt keinen L-Gewinn.

Hauptlauf: n=16 vor n=20, je Größe Anfangskontrollen in derselben Rollenfolge,
128 Kandidaten in Katalogreihenfolge, Endkontrollen. Bekannte Kandidaten werden
als bekannt markiert und regulär neu berechnet. Die 256 Kandidatenslots verwenden
ausschließlich mu=2; es gibt keine zusätzlichen mu=12-Katalogbewertungen.
Die negativen Kontrollen zertifizieren daher nicht jede neue Eingriffsgeometrie
einzeln. Es gibt weder Cacheübernahme alter Physik noch physikalische
Symmetriereduktion. Feste Reihenfolge liefert keine randomisierte Zeitdriftkontrolle.

Kontrollen müssen die Vorlauf-Kontrollregeln von MEAS-VAL-001 erfüllen:
topological primär positiv und alle fünf Chern-Indizes +1; trivial Index 0 in
allen primären und zusätzlichen Chern-Regionen und positives Screening abgelehnt;
undefined exakt H=0, erwartete Fermi-Ablehnung bei Bott/Chern, Gesamtklasse
undefined. Alle neun Localizer müssen bei den gapped Quadratkontrollen den
erwarteten Index +1 bzw. 0 liefern und invertibel sein; bei undefined Index 0
und Gap kappa/sqrt(2) innerhalb absoluter Toleranz 1e-10. Die H=0-Kontrolle
prüft exakt undefinierte Projektoren, nicht alle numerisch fastkritischen Fälle.

Anfang/Ende je Rolle und Größe: identische diskrete Zustände und Fehlercodes;
Eigenwerte, Bott-/Chern-Rohwerte, alle neun Localizer-Gaps, PH-Paarresiduum und
verfügbare Randgewichte stimmen bei rtol=0, atol=1e-10 überein. Verglichen werden
die im eingefrorenen Boundary-Builder ausgewählten Werte; bei undefined werden
keine Randgewichte degenerierter Nullzustände als Kontrollkriterium verwendet.
Eigenvektoren selbst müssen nicht gleich sein. Die beiden Vorlaufeingriffe
müssen numerisch gültig sein, das Primärscreening bestehen und alle sechs
Zusatzlocalizer mit Index +1 liefern; Zusatz-Chernabweichungen bleiben Diagnose.

Eine falsche Anfangskontrolle stoppt die Stufe. Falsche Endkontrollen verhindern
ein positives Gesamturteil. Operative Kandidatenfehler bleiben im Nenner, übrige
Slots laufen weiter. Integritäts-/Speicherfehler stoppen sofort. Pro Slot
genau ein regulärer Versuch; Wiederaufnahme nach einem technischen Abbruch
darf einen unversiegelten Slot erneut versuchen, archiviert aber jeden Versuch
und weist Zusatzaufwand separat aus. Keine Wiederholung zur Verbesserung von Scores.

Bei vollständig ausführbaren Slots werden im Vorlauf 136 und im Hauptlauf
4.556 Methodenaufrufe angefordert. Die undefined-Kontrollen liefern planmäßig
16 bzw. 32 Fermi-Ablehnungen (acht je undefined-Slot); sie sind keine
unerwarteten operativen Fehler. Falls weitere Kandidaten auf der Fermi-Energie
liegen, werden deren begründete Projektor-Ablehnungen zusätzlich gezählt.
Ein korrekt erfasster undefinierter Projektor oder ein nichtinvertibler
Localizer ist ein wissenschaftlich nicht qualifizierender Befund und kein
fehlender Slot. Davon getrennt bleiben unerwartete Exceptions, unvollständige
Methodenresultate und gescheiterte numerische Prüfungen.
Nach 8 bzw. 268 versiegelten Slots endet die angeforderte Stufe ohne Budgeterweiterung.

## Bericht, Entscheidung und weitere Bestätigung

Der Live-Bericht zeigt Stufe, Größe, Slot, Rolle, ID, Methode, Zeit, CPU/RAM,
letzte Versiegelung und getrennte Zähler `screening_passes`, `primary_hits`,
`qualified_hits`. Trefferzähler zählen nur Katalogkandidaten; im Vorlauf
werden sie nicht als wissenschaftlicher Erfolgszähler interpretiert.
Gepaarte Treffer stehen im Schlussbericht, sobald beide Seiten verfügbar sind.
ETA nutzt gemessene Zeiten je Größe und weist neue Größen anfangs als offen aus.

Der Schlussbericht enthält alle Einzelwerte, verfügbare Nenner, Fehler,
Grenzfälle, Zusatzmasken, die zwei getrennten Rankings und sämtliche 128 Paare.
Gesamturteil in dieser Priorität:

1. `control_validation_failed`: mindestens eine erforderliche Kontrolle ungültig.
2. `incomplete_or_unresolved`: bei gültigen Kontrollen mindestens ein
   Kandidat durch operative, numerische oder erforderliche Methodenfehler
   nicht vollständig entscheidbar. Positive Teilbefunde bleiben sichtbar,
   vollständige Ausschöpfung wird nicht behauptet.
3. `paired_proxy_candidates_found`: vollständiges Inventar, gültige Kontrollen
   und mindestens ein gepaarter Proxykandidat.
4. `single_size_or_setting_dependent_hits`: keine gepaarten Kandidaten,
   aber mindestens ein primärer Proxytreffer.
5. `threshold_borderline_only`: kein primärer Treffer, mindestens ein sonst
   screeningfähiger Fall in der primären Entscheidungszone.
6. `no_relevant_proxy_gain_in_catalog`: vollständiges Inventar und keiner der
   vorstehenden Befunde. Gemischte Topologieevidenz und wissenschaftliches
   Screeningversagen werden ausgewiesen; das Urteil behauptet nur fehlende
   qualifizierende Evidenz unter dem eingefrorenen Vertrag.

Die Schwellenzone des Zusatzpanels bleibt bei primären Treffern als
Einstellungsabhängigkeit sichtbar (Urteil 4). Die 0.005-Grenze für Größenpaare
verwendet inklusive Gleichheit wie angegeben; ihre Rohdifferenz wird gespeichert.

Endliche deterministische Enumeration: keine p-Werte oder binomialen
Konfidenzintervalle aus Kandidaten- oder Symmetriezahlen. Selbst ein gepaarter
Treffer ist eine zu bestätigende Entdeckung im ausgewählten Katalog.
Der Vorteil des Suchalgorithmus, Disorder-Robustheit, neue Graphfamilien und
isolierte Majorana-Nullmoden sind keine Schlussfolgerungen dieses Versuchs.

Für eine spätere unabhängige Bestätigung wird zuerst die gesamte gepaarte
Treffermenge mit IDs und Größenfortsetzung versiegelt. Neue Disorder-Seeds,
zusätzliche Größen, stärkste ressourcenangepasste Familien und die Auswahlregel
für ein etwaiges begrenztes Bestätigungsbudget werden in einem separaten
Protokoll vor deren Berechnung festgelegt. Es ist hier kein Bestätigungsbudget
freigegeben. Bei ausbleibendem Treffer endet dieser Versuch mit dem Nullbefund;
eine neue Hypothese erhält eine neue Version und verwendet bekannte Daten
ausdrücklich als Entwicklungsmaterial.

## Umsetzung und Abnahme

Vorgesehener Befehl nach Implementierung: `phase-10-local-enumeration` mit
explizit einem von `--preflight`, `--full`, `--resume`; Ergebnisstamm
`results/phase_10_local_enumeration_v1`. Die Stufen startet der Nutzer getrennt.
Der Hauptlauf verlangt gültigen Vorlauf bei gleicher Code-Revision und Umgebung.
Protokoll-ID, eingefrorener Text und Commit, Code-Revision, Kataloghash,
Inputinventar, Libraryversionen und Threadwerte werden vor Physikbeginn gebunden.
Resume prüft Originalinventare und Hashes und überspringt versiegelte Slots.
Bestehende Archive werden nicht überschrieben; abgeleitete Berichtswiederherstellung
über eine andere Revision bleibt ausdrücklich ein separater Vorgang.

Der Katalog und die bisherigen Runner behalten ihre Verträge. Neue Eingabebuilder
und Zusammenfassung sind erforderlich: Ein Katalogkandidat kann mehrere Patches
haben, zusätzliche Localizer dürfen die drei Primärwerte nicht ersetzen und die
MEAS-VAL-Schlussregel für 36 Slots passt nicht zu diesem Experiment. Bestehende
Geometry-, Modell-, Topologie-, Boundary- und Speicherfunktionen werden genutzt.
Äquivalente Wiederverwendung innerhalb eines Slots erfordert Tests; die
Zählung tatsächlicher Aufbau-/Solver-/Methodenaufrufe bleibt transparent.

Umgebung: Python 3.14, `PYTHONDONTWRITEBYTECODE=1`, `-B`, OMP_NUM_THREADS,
OPENBLAS_NUM_THREADS, MKL_NUM_THREADS und BLIS_NUM_THREADS jeweils 1.
Vor Nutzerstart: fokussierte Tests, vollständiges pytest, projektweites Ruff,
relevantes strict mypy, `git diff --check` und Git-Status. Kontrollphysik auf
n=12 ist für Tests erlaubt; neue Kandidatenphysik auf n=16/20 erst beim
Nutzerstart. Geschützte Worktree-Artefakte und gelöschte Phase-9.8-Ordner bleiben
unangetastet.

Erforderliche Regressionen: 8/268-Slotplan; Kataloghash und Bijektion; alle
Herkünfte; Primär-/Zusatzpaneltrennung; analytische H=0-Kontrolle; endliche
Solverresiduen; strikte 1-%-Grenze und ±eps; alle Treffer-/Schlusszustände;
unvollständige Ergebnisse ohne erfundene Werte; gespeicherte Dictionaries;
unterbrochene Slots und manipulierter Resume; Ende nach festem Budget.
