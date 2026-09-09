# Begrenzte Validierung des Messvertrags: Protokollentwurf v1

## Status und Entscheidungsfrage

Protokoll-ID: `TOPOSC-P10-MEAS-VAL-001`, Entwurf vom 2026-09-09.
Ausgangsrevision: `c0467aaf20bdc5b5eaabfe33a60744cf74b8b374`.
Grundlage sind der [Messvertragsentwurf](phase_10_measurement_contract_draft_v1.md),
der [Rand-/Innenbefund](phase_10_edge_location_v1_analysis.md) und die
[Forschungscharter](pre_phase_9_research_charter.md).

Dieser Plan wird nach Kenntnis der bisherigen Ergebnisse festgelegt. Die
12×12-Daten sind Entwicklungsdaten. Die hier vorgesehenen 16×16- und
20×20-Physikbewertungen wurden während der Protokollerstellung nicht berechnet.
Der Nutzer friert diesen Text durch einen separaten Commit ein; erst danach
folgen Implementierung, Tests und deren eigener Commit. Ein Runner für dieses
Protokoll existiert noch nicht.

> Unterscheidet eine vorab räumlich festgelegte Chern-Messregion zusammen mit
> Bott und Localizer unsere Kontrollklassen korrekt, und ergibt sie bei
> derselben lokalen Kantenänderung auf 16×16 und 20×20 ein konsistentes,
> ausreichend größenstabiles Screening?

Der Versuch umfasst genau 6 Vorlauf- und 36 Hauptlaufbewertungen. Er endet mit
einem der drei unten definierten Urteile. Es gibt keinen automatischen
Zusatzlauf, keine nachträgliche Maskenwahl und keine Schwellenanpassung innerhalb
dieses Versuchs. Seine Aussage gilt für den untersuchten Quadratbereich;
allgemeine Graphfamilien und Disorder-Robustheit bleiben weitere Forschungsfragen.

## Kontrollklassen und unabhängige Begründung

Die drei Modellrollen verwenden `ChiralPWaveModel` mit explizit gespeicherten
`ChiralPWaveParameters`. Achsen `(0,1)`, Chiralität +1, Referenzenergie 0,
komponentenweise Nambu-Basis und die vorhandenen Kopplungskonventionen gelten
für alle Rollen. Parameteränderungen dienen nur der Methodenprüfung; daraus
wird kein Geometrievorteil gegenüber anders parametrierten Referenzen abgeleitet.

| Modellrolle | hopping | chemical_potential | pairing | Erwartung auf unverändertem Quadrat |
| --- | ---: | ---: | ---: | --- |
| topological | 1 | 2 | 1 | Index +1 in der bisherigen Projektkonvention |
| trivial | 1 | 12 | 1 | Index 0, gapped und zum atomaren Grenzfall deformierbar |
| undefined | 0 | 0 | 0 | H=0; Fermi-Projektor für die Methoden nicht definiert |

Die Erwartung für das topologische Referenzmodell wird nicht durch eine
nachträgliche Wahl der Chern-Maske definiert. Aus den lokalen Hopping- und
Pairing-Konventionen folgt für das unendliche Quadrat
`xi(k)=-mu-2*t*(cos(kx)+cos(ky))` und
`E(k)^2=xi(k)^2+4*pairing^2*(sin(kx)^2+sin(ky)^2)`.
Bei nichtverschwindendem Pairing schließen die Bulkenergien bei
mu=-4t, 0 und 4t. mu=2 liegt im gapped Bereich mit Chern-Betrag 1. Das
Vorzeichen +1 ist in der Projektkonvention aus den bisherigen Quadratkontrollen
bekannt und wird vor der Umsetzung anhand der Fourier-/Ortsachsenkonvention
geprüft, nicht nach neuen Kontrollergebnissen umgedreht.

Die offizielle [p+ip-Beispieldokumentation von TightBindingApproximation.jl](https://quantum-many-body.github.io/TightBindingApproximation.jl/dev/examples/Superconductor/)
zeigt ebenfalls triviale Bereiche außerhalb der Bandgrenzen; ihre Vorzeichen
und Parameterdarstellung werden nicht ungeprüft auf unseren Code übertragen.
Die obige Dispersionsformel ist die Ableitung für die lokalen Projektkonventionen.

Die triviale Erwartung lässt sich stärker und auch für die Eingriffsvarianten
begründen: Bei Knotengrad höchstens 4 besitzt jede BdG-Zeile abseits der
Diagonale höchstens vier Hopping- und vier Pairing-Beiträge vom Betrag 1.
Für `H=diag(-12 I,+12 I)+V` gilt deshalb `||V||_2<=8`. Entlang
`H(s)=diag(-12 I,+12 I)+s V`, 0<=s<=1, bleibt der Abstand von 0 mindestens 4.
Die Deformation endet ohne Gap-Schließung beim atomaren Projektor. Diese
Normabschätzung ist unabhängig von der getesteten Chern-Maske. Sie macht auch
alle vorgesehenen mu=12-Eingriffsvarianten zu negativen Kontrollfällen.
Die endlichen numerischen Indikatoren müssen diese Erwartung erst erfüllen.

Beim Modell `undefined` ist H exakt null. Bott und lokaler Chern-Marker müssen
wegen Zuständen exakt auf der Fermi-Energie die Auswertung ablehnen. Ein
invertibler Localizer allein darf diesen Fall nicht als gapped trivial oder
topologisch klassifizieren: Bei den geraden n und dem mittigen, zwischen Sites
liegenden Probeort kann er trotzdem Index 0 liefern. Für H=0 ergibt sich sein
minimaler Gap analytisch als `kappa/sqrt(2)`. Beides wird separat geprüft.
Dieser Fall testet die Behandlung eines exakt undefinierten Projektors; er
repräsentiert nicht sämtliche fastkritischen oder numerisch schwierigen Fälle.

Dass lokale Chern-Marker innerhalb eines endlichen Samples räumlich variieren
können, ist Teil ihrer ursprünglichen Definition; siehe
[Bianco und Resta, Mapping topological order in coordinate space](https://arxiv.org/abs/1111.5697).
Die unten gewählten Regionen und Toleranzen sind Festlegungen dieses Versuchs,
keine aus dieser Quelle abgeleiteten universellen Genauigkeitsgarantien.

## Geometrien und Größenfortsetzung

Offene Quadrate mit Gitterabstand 1, Knotennummer `i=x*n+y`, Koordinaten
`(x,y)` in `[0,n-1]²`. Keine Positionsunordnung, Zufallszahlen oder Seedableitung.

| Stufe | n | Knoten | Kanten | Randknoten | zentrale Orte |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vorlauf | 12 | 144 | 264 | 44 | 36 |
| Hauptlauf | 16 | 256 | 480 | 60 | 64 |
| Hauptlauf | 20 | 400 | 760 | 76 | 100 |

Im Vorlauf werden nur unveränderte Quadratkontrollen gerechnet. Der Hauptlauf
enthält zusätzlich je Größe sechs Eingriffsgeometrien: drei tangentiale
Positionen mal zwei Arme. Jede Geometrie startet vom unveränderten Quadrat.

Anker: `y=n//2-1+offset` mit `offset in (-2,0,2)`.
Arm `boundary`: x=0. Arm `interior`: x=n//4. Keine Rotation oder Spiegelung.
Eingriff wie in EDGE-LOCATION-001:

```text
A=(x,y), B=(x+1,y), C=(x+1,y+1), D=(x+2,y+1)
entfernen: {A,B}, {C,D}
hinzufügen: {A,C}, {B,D}
```

Dieselben zwei Einheitskanten werden durch zwei parallele Diagonalen der Länge
sqrt(2) ersetzt. Jeder betroffene Knotengrad bleibt gleich. Die gesamte
Gradfolge, Knoten-/Kantenanzahl, Box und Koordinaten bleiben referenzgleich.
Der innere Eingriff liegt vollständig in der zentralen Region, der Randeingriff
außerhalb. Die innere Randdistanz wächst als n/4; die lokale Mustergröße bleibt
gleich. Es ist ein einzelner lokaler Defekt mit sinkender Dichte `2/E`, keine
Prüfung bei konstanter Unordnungsdichte. Die tangentialen Offsets sind feste
Abstände von der mittigen Lage. Die Rand-/Innenumgebung bleibt unterschiedlich.

Lokale Messregion: die 16 Sites `(x+u,y+v)` mit u=0,1,2,3 und v=-1,0,1,2.
Sie wird vollständig gespeichert, weder beschnitten noch aufgefüllt. Referenz
ist immer das Anfangsquadrat derselben Größe und derselben Modellrolle.
Jede Eingriffsgeometrie wird einmal mit topological und einmal mit trivial
gerechnet. Die undefined-Rolle wird nur an den unveränderten Quadraten benutzt.

Ressourcenvalidator: Zusammenhang, einfache ungerichtete Kanten, sortierte
Orientierung source<target, Displacements aus Koordinaten, keine geraden
Kreuzungen außer gemeinsamen Endpunkten, Knotengrade 2–4, Mindestabstand 0.55,
Maximallänge 1.75, Geometrietoleranz `1e-12`. Genau die ursprünglichen `4n-4`
Randknoten in der Koordinatenschale 0.875, eine äußere Randkomponente Index 0.
Keine Site-Typen, Faces oder Baumstruktur, `dimension_records=()`.
Neutrale Darstellungsdaten wie in SIZE-001:
`edge_type="size_calibration_coupling"`, leere Edge-Metadaten und ausschließlich
`generator="phase10_size_square_deformation_v1"` als Geometriemetadatum.
Rolle, Größe, Offset, Arm und Modellparameter stehen separat im Inputplan.

Die 18 theoretischen Eingriffsmuster für n=12,16,20 wurden bei der Planung nur
mit Ganzzahlkoordinaten auf Budget, Gradfolge, Zusammenhang, Kreuzungen,
Messregion und zentrale Überlappung geprüft. Keine zukünftige Physik wurde
ausgewertet. Die produktive Prüfung muss jeden geplanten Input erneut vollständig
validieren. Keine Reparatur, Ersatzgeometrie oder Änderung an den alten Validatoren.

## Methodenpanel und Klassifikation

Vollständiges `numpy.linalg.eigh`, numerische Toleranz `1e-10`, 16 retained
Niedrigenergiezustände für die gapped Rollen, keine Zufallsseeds.
Jeder Slot archiviert Hamiltonian bzw. reproduzierbare Modellinputs,
vollständige Eigenpaare, Solverstatus und Methodenresultate. Modellparameter
müssen in der tatsächlichen Modellprovenienz stehen, nicht nur im Rollennamen.

Panel pro Hamiltonian:

- Bott mit Perioden (0.95n,0.95n), (n,n), (1.05n,1.05n), Toleranz `1e-6`.
- Localizer bei `((n-1)/2,(n-1)/2)`, Energie 0, kappa=0.1,0.2,0.3.
  L ist das Minimum der drei Gaps. Kappa bleibt im gleichen Orts-/Energiemaßstab;
  ein größenunabhängiger physikalischer Schutz wird daraus nicht vorausgesetzt.
- Lokaler Chern-Marker mit fünf Masken, Quantisierungstoleranz 0.005:
  `graph_depth_2`, `graph_depth_3`, `fixed_depth_2`, `fixed_depth_3`,
  `central_fraction`. Definitionen wie SIZE-001: Graphdistanz >=2/3;
  ursprüngliche Koordinaten-Randtiefe >=2/3 bzw. >=n/4.
- Positive Voronoi-Flächen in `[-0.5,n-0.5]²`, jede Fläche 1 und Summe n²
  (`rtol=0,atol=1e-10`); explizite physische Site-/Unique-Orts-/Nambu-Zuordnung.

Primäre Chern-Region ist ausschließlich `central_fraction`, vor der ersten
Bewertung jeder Größe als Site-Menge festgelegt. Alle anderen Regionen sind
Sensitivitätsdiagnosen. Keine Entfernung der Eingriffssites aus einer Maske.
Feste Tiefen-2/Tiefen-3-Regionen enthalten bei n=16 jeweils 144/100 Sites,
bei n=20 jeweils 256/196 Sites; die zentralen Regionen 64/100 Sites.

Die primäre Topologieklassifikation verwendet ausschließlich tatsächlich
berechnete Ergebnisse und folgende Reihenfolge:

1. Ungültige Geometrie, Modell- oder Speicherintegrität: Abbruch mit Fehlerbeleg.
2. Fehlgeschlagener Solver: `operational_failure`, keine positive Klassifikation.
3. `min(abs(E))<=1e-10`: `undefined_fermi_projector`, unabhängig von eventuell
   verfügbaren Localizerresultaten. Erwartete Methoden-Ablehnungen separat erfassen.
4. Bei verfügbarem gapped Solver: alle drei Bott-Indizes, alle drei invertiblen
   Localizer-Indizes und der quantisierte zentrale Chern-Index müssen gleich
   sein. Gemeinsam +1 ergibt `primary_topological`; gemeinsam 0 ergibt
   `primary_trivial`. Jeder andere Fall ergibt `mixed_or_unresolved`, mit den
   Einzelresultaten und Gründen. Ein gemeinsam anderer Integer wird separat
   als unerwarteter Index dokumentiert und nicht still in die zwei Klassen umgedeutet.

Diese Statusnamen beschreiben das neue endliche Primärpanel. Widerspruch einer
zusätzlichen Region wird als `regional_sensitivity` zusammen mit allen fünf
Werten ausgegeben, nicht verschwiegen und nicht als zusätzliche unabhängige
Stimme gezählt. Das alte SIZE-Screening wird daneben unverändert beschreibend
berechnet, soweit seine Inputs verfügbar sind. Alte Ergebnislabels bleiben gültig.

Ein separat benanntes positives Screening für diesen Test erfordert
`primary_topological`, L>=0.20, PH-Paarresiduum <=1e-8 und mindestens vier der
acht niedrigsten Absolutenergiezustände mit Randgewicht >=0.80. Die triviale
Kontrolle muss dieses positive Screening ablehnen; sie muss kein positives
Randzustandsgate erfüllen. L allein klassifiziert keine Topologie.

Bei undefined werden Bott und alle Chern-Masken separat auf die erwartete
Fermi-Ablehnung geprüft. Alle drei Localizer werden trotzdem ausgewertet:
erwartet Index 0 und Gap `kappa/sqrt(2)` innerhalb `rtol=0,atol=1e-10`.
Das Gesamtergebnis bleibt `undefined_fermi_projector`. Degenerierte
Nullenergie-Eigenvektoren begründen keine Randzustandsaussage.

## Budget, Reihenfolge und Vorlauf-Freigabe

| Stufe | Beginn-/Endkontrollen | Eingriffsbewertungen | Gesamt |
| --- | ---: | ---: | ---: |
| preflight, n=12 | 3 Rollen jeweils Anfang und Ende | 0 | 6 |
| full, n=16 | 3 Rollen jeweils Anfang und Ende | 3 Offsets × 2 Arme × 2 Rollen = 12 | 18 |
| full, n=20 | 3 Rollen jeweils Anfang und Ende | 3 Offsets × 2 Arme × 2 Rollen = 12 | 18 |
| Gesamt | 18 | 24 | 42 |

Je Größe: Anfangskontrollen in Reihenfolge topological, trivial, undefined;
dann im Hauptlauf Modellrollen topological, trivial, darin Offsets -2,0,2,
jeweils boundary vor interior; zuletzt Endkontrollen topological, trivial,
undefined. Größen aufsteigend. Vorlauf nur die beiden Kontrollblöcke.
Diese feste Reihenfolge beansprucht keine randomisierte Entkopplung von Zeitdrift;
die Kontrollwiederholungen prüfen technische Reproduzierbarkeit.

Vorlauf bestanden heißt exakt: alle sechs Slots technisch bearbeitet und
versiegelt; beide topological-Kontrollen `primary_topological` und positives
Screening; beide trivial-Kontrollen `primary_trivial` und positives Screening
abgelehnt; beide undefined-Kontrollen mit genau den erwarteten Zuständen der
Methoden. Für topological müssen zusätzlich alle fünf Chern-Indizes +1 sein,
für trivial alle fünf 0. Keine unerwartete Nichtverfügbarkeit.

Erwartete Fermi-Ablehnungen im undefined-Fall sind korrekt bearbeitete
Kontrollergebnisse. Eine pauschale Forderung nach sechs vollständig verfügbaren
Chern-Panels wäre falsch. Der Bericht trennt `slots_processed`,
`unexpected_failures`, `expected_method_rejections` und `controls_passed`.

Bei jeder Rolle müssen Anfang/Ende in Status und diskreten Ergebnissen exakt
übereinstimmen. Verfügbare Eigenenergien, Bott-/Chern-Rohschätzungen und
Localizer-Gaps werden mit `rtol=0,atol=1e-10` verglichen; bei undefined zusätzlich
die erwarteten Fehlercodes. Keine Gleichheit einzelner Eigenvektoren verlangen.

Vorlauf und Hauptlauf werden getrennt vom Nutzer gestartet. Der Hauptlauf ist
nur nach bestandenem Vorlauf bei identischer Revision und Umgebung zulässig.
Eine ungültige Anfangskontrolle stoppt die angeforderte Stufe. Fehler in den
Endkontrollen werden erhalten und verhindern ein positives Schlussurteil. Unerwartete operative
Variantenfehler bleiben im Nenner; die übrigen Slots werden weiter bearbeitet.
Integritäts- und Speicherfehler stoppen sofort. Keine Auswahl nach Vorlaufeffekt.

## Festgelegte Auswertung und endgültiger Entscheidungspunkt

Primär ausgewertet wird die Klassifikation gegen die unabhängig begründeten
Kontrollrollen: je Größe Zahlen korrekt, falsch, gemischt/unaufgelöst,
unerwartet fehlgeschlagen und erwartbar abgelehnt. Die mu=12-Eingriffsvarianten
werden zusätzlich als analytisch begründete triviale Kontrollen erfasst.
Die mu=2-Eingriffsvarianten sind Untersuchungsfälle ohne separat bewiesenes
Soll-Label; ihre Abweichungen werden nicht als bekannte Fehlklassifikationen gezählt.

Je Größe/Rolle/Arm/Offset werden alle fünf Chern-Werte, L, Bott-/Localizer-Panel,
Warnungen, Rand- und PH-Diagnostik sowie beide Screeninglabels berichtet.
Gepaarte Unterschiede beziehen sich auf denselben Offset und dieselbe Rolle:
`abs(C_interior-q)-abs(C_boundary-q)`, q=1 für topological, q=0 für trivial.
Alle Einzelwerte und Median/Minimum/Maximum mit verfügbarem n bleiben sichtbar.

Lokale Auswertung: vollständige Markeränderung gegenüber der passenden
Anfangsreferenz, signierter und absoluter Mittelwert in der 16-Site-Region;
zusätzlich jede ursprüngliche Randtiefe d=0,...,n//2-1. Graphmaskenänderungen
gegenüber den entsprechenden festen Masken erhalten ihre Site-IDs und
Markerbeiträge. Keine Schwelle auf einzelne Markeränderungen.

Größenvergleich: Für jede der sechs topological-Eingriffslagen wird
`abs(C_central(n=20)-C_central(n=16))` berichtet. Der vorab gewählte Grenzwert
für ausreichende Stabilität in diesem engen Panel ist 0.005; das ist ein neuer
operativer Akzeptanzwert, kein Konvergenztheorem. Die abnehmende Defektdichte und
größere Mittelungsregion gehören zur Interpretation.

Nach dem festen Hauptbudget gibt es genau drei Urteile, in dieser Reihenfolge:

1. **Kontrollvalidierung fehlgeschlagen:** irgendein Referenzkontrollblock
   verletzt die oben definierten Vorlauf-Kontrollregeln oder eine der zwölf
   mu=12-Eingriffsvarianten liefert nicht `primary_trivial`. Keine Freigabe
   des neuen Messvertrags; Ursachenbericht mit konkreter Fehlerliste.
2. **Kontrollen bestanden, Eingriffspanel noch nicht stabil aufgelöst:**
   Kontrollen bestehen, aber mindestens eine der zwölf mu=2-Eingriffsvarianten
   verfehlt das neue positive Screening, eine davon ist unerwartet nicht
   auswertbar oder einer der sechs Größenunterschiede überschreitet 0.005.
   Keine Freigabe für eine leistungsbezogene neue Suche auf dieser Basis.
   Das Experiment endet mit diesem Befund, ohne automatische Verlängerung.
3. **Begrenzter Messvertrag akzeptiert:** sämtliche vorstehenden Prüfungen
   bestehen. Dies erlaubt die Ausarbeitung eines separaten Suchprotokolls
   innerhalb des untersuchten lokalen Quadratbereichs mit festgelegten
   Messregionen. Es bescheinigt weder Disorder-Robustheit noch Anwendbarkeit
   auf beliebige Graphfamilien. Abweichende Zusatzmasken bleiben im Bericht.

Die Daten sind deterministisch. Es gibt keine p-Werte, binomialen
Konfidenzintervalle oder Anspruch auf unabhängige Zufallsbestätigung. Vorlauf-
und Entwicklungsdaten werden nicht mit dem Hauptlauf gepoolt. Die zwei neuen
Größen sind eine vorab festgelegte Übertragungsprüfung auf bislang nicht
ausgewertete Konfigurationen, keine umfassende thermodynamische Bestätigung.
Jeder spätere Änderungsbedarf wird als neue Entscheidung dokumentiert.

## Architektur und Reproduzierbarkeit

Die Inspektion des aktuellen Codes zeigt drei Grenzen der direkten Wiederverwendung:

- `phase_10_size_methods` akzeptiert nur n=8,10,12. Ein neuer Validator und
  Input-Builder für dieses Protokoll darf diese eingefrorene Größenliste nicht ändern.
- `evaluate_phase_9_8_descriptive_geometry` bindet fest die alten Modellparameter.
  Der neue Runner benötigt einen eigenen `GeometryModelAdapter` mit den tatsächlich
  angeforderten `ChiralPWaveParameters`. Kein Umetikettieren des alten Adapters.
- `evaluate_phase_9_8_topology` liefert ein gemeinsames Bundle und bricht bei
  einem Methodenfehler ab. Hier werden Bott, Chern und Localizer mit ihren
  bestehenden öffentlichen Funktionen getrennt aufgerufen und Ergebnisse/Fehler
  einzeln archiviert. Ein Bott-Fehler darf die Localizerdiagnose nicht verschwinden lassen.

Bestehende Geometry-, Modell-, Solver-, Nambu-, Marker- und Speicher-APIs sollen
wiederverwendet werden. Gleichwertige Wiederverwendung eines Eigenpaarsatzes
innerhalb eines Slots ist nur mit überprüfter Äquivalenz zulässig; keine
Physikwiederverwendung zwischen Slots oder aus früheren Kampagnen.

Vorgesehener Ergebnisstamm `results/phase_10_measurement_validation_v1`, Stufen
`preflight/full`. Das gesamte angeforderte Inventar mit Geometrie-IDs, Rollen,
Modellparametern, Masken und Modell-/Solverkonfiguration wird vor der ersten
Physikbewertung versiegelt. Keine alten Ergebnisse überschreiben. Jeder Versuch
bleibt mit exklusiv veröffentlichten Inputs/Outcomes nachvollziehbar. Resume
prüft alle Inventare und Hashes, bindet sie an den eingefrorenen Plan sowie die
identische Revision/Umgebung und überspringt abgeschlossene Slots.

Fortschritt: Stufe, Größe, Kontroll-/Variantenrolle, Modellrolle, Offset, Arm,
aktuelle Methode, bearbeitete Slots, CPU/RAM, letzte Versiegelung. Heartbeat
während Aufbau und Solver. Aufbau-, Hamiltonian-, Solver-, Methoden-, Speicher-
und Gesamtzeit getrennt. ETA bei unbekannter größerer Systemgröße ausdrücklich
vorläufig; keine unkommentierte lineare Hochrechnung aus n=12.

Vor Nutzerstart: Python 3.14, `PYTHONDONTWRITEBYTECODE=1`, `-B`; alle vier
BLAS/OpenMP-Threadvariablen auf 1. Fokussierte Tests, vollständiges pytest,
projektweites Ruff, relevantes strict mypy, `git diff --check` und `git status`.
Tests dürfen Kontrollphysik auf n=12 verwenden; keine Hauptlaufphysik auf n=16
oder 20 vor dem Nutzerstart. Geometrische und synthetische Tests sind erlaubt.
Besonders prüfen: korrektes Modellrouting, H=0-Behandlung je Methode,
triviale Normschranke, Primär-/Zusatzmasken, sämtliche drei Schlussurteile,
unvollständige Ergebnisse, erwartete Ablehnungen gegenüber operativen Fehlern,
unterbrochenes Resume und manipulierte Provenienz. Geschützte Artefakte aus
früheren Arbeitsschritten bleiben unangetastet.
