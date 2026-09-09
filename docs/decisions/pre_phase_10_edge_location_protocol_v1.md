# Rand- und Inneneingriffe: Protokollentwurf v1

## Status, Frage und Geltungsbereich

Protokoll-ID: `TOPOSC-P10-EDGE-LOCATION-001`. Entwurf vom 2026-09-09,
Ausgangsrevision `aa6800a5e40f26342eabeac912ef24132dafaa04`.
Dieser Plan folgt auf die bekannte
[räumliche SIZE-001-Auswertung](../analysis/phase_10_size_methods_spatial_v1/report_de.md).
Er ist noch nicht eingefroren und noch nicht als Runner implementiert.
Ein separater Nutzer-Commit fixiert die Festlegungen vor Implementierung und
neuen Physikrechnungen. Alte Protokolle und Ergebnisarchive bleiben unverändert.

> Wie unterscheiden sich lokale Markeränderung und zentrale Chern-Schätzung,
> wenn derselbe graderhaltende Zwei-Kanten-Eingriff am Rand oder im Inneren
> desselben endlichen Quadrats angebracht wird?

Dies ist ein deterministischer Methodenversuch zur
[Forschungscharter](pre_phase_9_research_charter.md), keine Suche nach Gewinnern,
keine unabhängige Disorder-Validierung und kein Nachweis einer robusteren Familie.
Nur n=12 wird untersucht; Größenkonvergenz ist damit nicht prüfbar.

## Architekturprüfung: warum ein neuer Operator nötig ist

`legal_size_swaps` aus `src/toposc_lab/search/phase_10_size_methods.py` entfernt
bisher eine vorhandene Kante und fügt eine andere, zuvor abwesende Kante hinzu.
Im unveränderten offenen Quadrat haben alle inneren Knoten Grad 4. Unter der
Obergrenze 4 muss jeder innere Endpunkt einer neuen Kante zugleich Endpunkt der
entfernten Kante sein. Wären beide neuen Endpunkte innen, müsste dieselbe Kante
wieder hinzugefügt werden; sie ist jedoch nicht im Pool zuvor abwesender Kanten.
Jeder zulässige erste Schritt hat daher mindestens einen Randendpunkt an der
hinzugefügten Kante. Spätere Schritte können durch veränderte Grade anders aussehen.

Read-only-Prüfung der bestehenden API auf den drei unveränderten Quadraten:

| n | Zulässige erste Einzeltausche | Davon mit neuer Kante ausschließlich innen |
| --- | ---: | ---: |
| 8 | 592 | 0 |
| 10 | 928 | 0 |
| 12 | 1328 | 0 |

Die bisherige Konstruktionsregel begünstigt damit anfänglich Randbeteiligung.
Das ist kein Beweis, dass sie alle beobachteten räumlichen Effekte erklärt.
Ein Rand-/Innenvergleich darf nicht durch Lockerung der Gradgrenze oder durch
Umbenennen alter Einzeltausche entstehen. Hier wird ausdrücklich ein neuer,
graderhaltender Zwei-Kanten-Eingriff eingeführt. Seine Wirkung ist nicht direkt
mit den alten Trajektorien aus zehn versuchten Einzeltauschen gleichzusetzen.

## Exakter Geometrievertrag

Referenz: offenes Quadrat n=12, `i=x*12+y`, `(x,y)` ganzzahlig in `[0,11]²`.
144 Knoten, 264 Kanten, 44 Randknoten; keine Positionsunordnung, keine Zufallswahl.
Jede Variante startet erneut von derselben Referenz, nicht vom vorherigen Fall.

Das untransformierte Eingriffsmuster besteht aus vier Orten:

```text
A=(x,y)       B=(x+1,y)
C=(x+1,y+1)   D=(x+2,y+1)
entfernen: {A,B}, {C,D}
hinzufügen: {A,C}, {B,D}
```

Die beiden entfernten horizontalen Kanten der Länge 1 werden durch zwei parallele
Diagonalen der Länge sqrt(2) in benachbarten Elementarzellen ersetzt. Keine zwei
kreuzenden Diagonalen derselben Zelle. Alle vier Knoten verlieren und gewinnen
je eine Kante. Der Tausch ist atomar; die Zwischenstufe mit nur einer entfernten
Kante ist keine Physikvariante.

Ein Paar hat denselben tangentialen Anker `y` und dieselbe Transformation,
aber `x=0` für `boundary` bzw. `x=3` für `interior`. Für alle vier betroffenen
Orte ist der minimale ursprüngliche Koordinaten-Randabstand damit 0 bzw. 3.
Tangentiale Anker: `y in (3,4,5,6,7)`.

Transformation `T=R^r S^m`, erst S, dann R:
`S(x,y)=(x,11-y)`, `R(x,y)=(11-y,x)`, `m in (0,1)`, `r in (0,1,2,3)`.
Sie wird auf die Eingriffsorte und die unten definierte Messregion angewandt.
Das Koordinatenarray und die Knotennummerierung des Referenzquadrats bleiben
unverändert; nur die entsprechenden Kantenpaare werden ersetzt.

Die 40 Paar-IDs sind `p=(4*m+r)*5+(y-3)`. Es ergeben sich 80 verschiedene
Kantenmengen. Kein Rücktausch, keine Wiederholung innerhalb einer Variante,
keine Auswahl anhand von Physikwerten und kein Ersatz für ungültige Geometrien.

Es gelten alle Ressourcen- und Darstellungsregeln von
[SIZE-001](pre_phase_10_size_methods_protocol_v1.md), geprüft mit
`validate_size_geometry(...,12)`: Zusammenhang, einfache planare gerade Kanten,
Abstand mindestens 0.55, Kantenlänge höchstens 1.75, Grad 2–4, feste äußere
Koordinatenschale 0.875 und orientierte, sortierte Kanten mit neu berechneten
Displacements. Der neutrale Builder `build_size_geometry` bleibt unverändert;
Typ `size_calibration_coupling`, Metadaten ausschließlich
`generator="phase10_size_square_deformation_v1"`. Rollen und Transformationen
stehen separat im neuen Eingriffsplan, nicht in der Geometry.

Zusätzliche Prüfungen: Koordinaten exakt gleich der Referenz; Grad an jedem
einzelnen Knoten exakt gleich der Referenz; genau zwei entfernte und zwei neue
Kanten; alle neuen Kanten aus dem bestehenden lokalen Kantenpool; erwartete
Randtiefe; gleiche entfernte/hinzugefügte Längen und Richtungen innerhalb des
Paars; identische Geometrie-IDs nur bei ausdrücklich deklarierten Kontroll- oder
Vorlaufwiederholungen. Es findet keine geometrische Reparatur statt.

Rand und Inneres besitzen trotz gleicher globaler Gradfolge verschiedene lokale
Umgebungen: Am Rand hat ein Eingriffsendpunkt Grad 3, innen haben alle vier
Grad 4. Auch die Nähe zu einer offenen Systemgrenze ist nicht isoliert von dieser
Umgebung veränderbar. Gemessen wird der Ortseffekt dieses konkreten Transplants,
nicht ein von sämtlichen lokalen Eigenschaften unabhängiger reiner Abstandseffekt.
Gegenüber dem Quadrat ändert sich das Kantenlängenspektrum; zwischen den beiden
Paarhälften ist diese Änderung identisch. Konstante Kopplungsamplituden bleiben
wie bisher ohne Distanzabfall.

## Physik und vorab definierte Messgrößen

Das vollständige Physik-/Methodenpanel wird unverändert aus SIZE-001 übernommen:
`ChiralPWaveModel(t=1, mu=2, pairing=1, chirality=+1)`, Achsen `(0,1)`,
Referenzenergie 0, vollständiges `numpy.linalg.eigh`, Toleranz `1e-10`, 16
Niedrigenergiezustände und komponentenweise Nambu-Anordnung; `evaluation_seed=None`.
Bott-Perioden pro Achse 11.4, 12, 12.6 mit Quantisierungstoleranz `1e-6`;
Localizer-Zentrum `(5.5,5.5)`, Kappa 0.1, 0.2, 0.3; L ist das Minimum der Gaps.
Die fünf Chern-Masken sind `graph_depth_2`, `graph_depth_3`, `fixed_depth_2`,
`fixed_depth_3`, `central_fraction`, definiert genau wie in SIZE-001.
Clippingzelle `[-0.5,11.5]²`, positive Voronoi-Flächen mit Summe 144
(`rtol=0,atol=1e-10`); Orts-, Masken- und Nambu-Zuordnung explizit archivieren.
Ohne Ortsverschiebung müssen alle 144 Flächen 1 sein (dieselbe Toleranz).

Primäre deskriptive Größe je Paar:

`D_p = abs(C_central_fraction(interior)-1) - abs(C_central_fraction(boundary)-1)`.

Positives D bedeutet eine größere zentrale Chern-Abweichung beim inneren
Eingriff, negatives D das Gegenteil. Alle 40 Werte, verfügbare Paarzahl,
Median, Minimum und Maximum werden berichtet; zusätzlich alle zehn Gruppen
`(m,y)` mit ihren vier r-Werten. Keine nachträgliche Auswahl günstiger Orte.
Die zentrale Maske enthält stets die 36 Orte mit ursprünglicher Tiefe >=3;
sie ist identisch mit `fixed_depth_3`, also keine unabhängige weitere Messung.
Der innere Eingriff liegt in dieser Maske, der Randeingriff außerhalb. Der
primäre Kontrast enthält deshalb absichtlich die unterschiedliche Nähe zur
Messregion; er allein besagt nicht, dass ein Eingriff die Phase stärker zerstört.

Festgelegte sekundäre Auswertung:

- Vollständige ortsaufgelöste Markerkarte in physikalischer Site-Reihenfolge;
  `delta_i = marker_variant_i - marker_start_reference_i` am selben Ort.
- Für jedes Eingriffsmuster die 16 Orte
  `P=T({(x+u,y+v): u=0,1,2,3; v=-1,0,1,2})`. Diese einseitige, zum Inneren
  gerichtete Messregion liegt für alle Fälle vollständig im Quadrat; kein
  Clipping und kein Auffüllen. Sie enthält exakt dieselben relativen Orte in
  beiden Paarhälften. Berichtet werden Mittelwert von `abs(delta_i)`, signierter
  Mittelwert von delta und jeweils `interior-minus-boundary`.
- Für jede ursprüngliche Randtiefe d=0,...,5: Zahl der Orte, Mittelwert von
  `abs(delta_i)` und signierter Mittelwert. Keine Anwendung der Chern-
  Quantisierungstoleranz auf einzelne Marker oder diese Differenzgrößen.
- Alle fünf Chern-Schätzungen, Residuen, Warnungen und Maskenmengen; pro Maske
  separat `abs(C-1)<=0.005`. Für Graphmasken zusätzlich die Zerlegung
  `C_variant(M_variant)-C_ref(M_ref)` in
  `[C_variant(M_variant)-C_ref(M_variant)] + [C_ref(M_variant)-C_ref(M_ref)]`.
  Der zweite Term isoliert Maskenwahl am Referenzfeld, nicht jede Wechselwirkung.
- Bott-/Localizer-Werte und Rohgrößen, L, PH- und Randdiagnostik sowie das alte
  benannte SIZE-Screening. Keine zusätzliche Konjunktion aller fünf Masken und
  keine gelockerte Schwelle. L bleibt ein Schutz-Proxy, kein nachgewiesener Bulkgap.

Verfügbarkeit und Fehler bleiben je Methode sichtbar. Ein operativer Fehler
ist kein physikalischer Misserfolg. Eine nicht verfügbare Paarhälfte macht D
nicht verfügbar, nicht null. Kriterienzahlen erhalten den festen Nenner 40
pro Arm, zusätzlich verfügbare n und bedingte Anteile. Diese deterministisch
gewählten, teilweise symmetrisch verwandten Orte sind keine 40 unabhängigen
Stichproben: keine Wilson-/Bootstrap-Intervalle, p-Werte oder Populationsraten.
Die vier Rotationen werden zur Orientierungsprüfung einzeln archiviert; ihre
Streuung ist kein statistischer Standardfehler. Spiegelungen werden nicht
stillschweigend als zusätzliche unabhängige Physik oder als Vorzeichenwechsel
des festgehaltenen Modellparameters behandelt.

## Budget, Kontrollen und Startfolge

Keine Generations-, Disorder- oder Validierungsseeds: Das Inventar wird vollständig
durch die obigen Ganzzahlen bestimmt, der volle Solver ist deterministisch
konfiguriert. Alte und reservierte künftige Seedrollen werden nicht benutzt.
Dieser Versuch ersetzt keine spätere Bestätigung auf frischen Zufallsdaten.

| Stufe | Varianten | Quadratkontrollen | Physikbewertungen |
| --- | ---: | ---: | ---: |
| Technischer Vorlauf | Paare `(m,r,y)=(0,0,5),(0,1,5)`, beide Arme | Anfang + Ende | 6 |
| Hauptlauf | alle 40 Paare, beide Arme | Anfang + Ende | 82 |

Der Vorlauf enthält bewusst vier Geometrien aus dem eingefrorenen Hauptinventar.
Er ist nur eine technische Probe, kein unabhängiger Datensatz. Seine Werte werden
nicht zur Auswahl oder Veränderung des Hauptinventars verwendet und nicht mit
dem Hauptlauf gepoolt. Jeder Hauptslot wird frisch berechnet, ohne Physik-Caching
aus dem Vorlauf oder alten Kampagnen. Kontrollwiederholungen sind technische
Replikate, keine unabhängigen Nullkontrollen.

Reihenfolge je Stufe: Anfangsquadrat, Paare aufsteigend nach p, Endquadrat.
Gerade p: boundary vor interior; ungerade p: interior vor boundary. Fehlende p
im Vorlauf werden übersprungen, nicht neu nummeriert. Vollständiges Inventar und
seine Prüfungen werden vor der ersten Physikrechnung der jeweiligen Stufe
versiegelt. Keine Rechnung mit noch nicht vollständig validiertem Inventar.

Kontrollen müssen vollständig verfügbar sein: Bott, alle drei Localizer und
alle fünf Chern-Indizes +1, L>=0.20, mindestens vier der acht niedrigsten
Absolutenergiezustände mit Randgewicht >=0.80 und vier PH-Paare mit Residuum
<=1e-8. Anfang/Ende: diskrete Resultate exakt gleich; alle Eigenenergien,
Chern-/Bott-Rohschätzungen und Localizer-Gaps gleich mit `rtol=0,atol=1e-10`.
Keine Bytegleichheit entarteter Eigenvektoren verlangen.

Fehler der Anfangskontrolle, Geometrie-, Integritäts- oder Speicherfehler stoppen.
Operative Variantenfehler werden archiviert; übrige Slots laufen weiter.
Fehler der Endkontrolle verhindert physikalische Schlüsse für die Stufe.
Vorlauf-Freigabe verlangt 6/6 verfügbare Bewertungen des gesamten Panels und
bestandene Kontrollen, aber keine Mindestzahl bestandener Varianten-Screenings.
Bei Methoden-Nichtverfügbarkeit keine Ersatzmaske und kein Schwellenwechsel.
Wissenschaftlich gute oder schlechte Ergebnisse lösen keinen Zusatzlauf aus.

## Umsetzung nach Nutzer-Freeze

Separater Eingriffsplan, streng geprüfte Paardefinition und separater Runner;
bestehende Modell-, Geometrie-, Topologie- und Archiv-APIs wiederverwenden.
Die eingefrorenen alten Runner und ihre Gates werden nicht verändert. Kein
allgemeiner Evolutionsoperator, keine Population und keine Phase-11-Funktionalität.

Vorgesehener neuer Stamm: `results/phase_10_edge_location_v1`, getrennte Stufen
`preflight/full`. Noch keine Startbefehle: Erst Protokoll separat committen,
dann implementieren, prüfen und Implementierung separat committen. Beide
Physikstufen startet der Nutzer ausdrücklich. Code-, Protokoll- und Umgebungs-
Hashes werden getrennt gespeichert; Resume verlangt unveränderte Revision und
Umgebung, bindet Inputs an den vollständigen Plan, prüft alle Versiegelungen und
rechnet abgeschlossene Slots nicht neu. Ausschließlich exklusive Veröffentlichung
neuer Artefakte; keine Überschreibung historischer Ergebnisse.

Fortschritt zeigt Stufe, p, m, r, y, Arm, aktuelle Pipelinephase, erledigte Slots,
Zeit, CPU, RAM und letzten versiegelten Pfad. Heartbeat auch bei Geometrieaufbau
und langen Methodenaufrufen. ETA erst aus tatsächlich beobachteten Laufzeiten;
Aufbau-, Solver-, Diagnose-, Speicher- und Gesamtzeit getrennt berichten.

Vor Nutzerstart: Python 3.14, `PYTHONDONTWRITEBYTECODE=1`, `-B`;
OMP/OpenBLAS/MKL/BLIS jeweils ein Thread; fokussierte Tests, vollständiges pytest,
projektweites Ruff, relevantes strict mypy, `git diff --check`, `git status`.
Neue Tests insbesondere für alle 80 Geometrien, Paartransformationen und
16-Orte-Messregionen, konstante Gradfolge, präzise Masken-/Site-Zuordnung,
Fehlernenner, Kontrollsperren, Plan-/Provenienzmanipulation und unterbrochenes Resume.

## Bisherige Prüfung dieses Entwurfs

Vor dem Schreiben dieses Plans wurden ausschließlich Geometrien im Speicher
konstruiert und mit den bestehenden APIs geprüft: alle 80 Kantenmengen
verschieden, alle gültig, Gradfolge unverändert, Eingriffstiefen 0/3 und alle
16-Orte-Messregionen vollständig. Außerdem wurde die obige Erstschritt-Tabelle
mit `legal_size_swaps` reproduziert. Python 3.14 mit `-B` und deaktiviertem
Bytecode; keine Hamiltonians, Eigenwerte oder Marker berechnet, keine neuen
Ergebnisordner angelegt. Diese Machbarkeitsprüfung ist kein Vorlauf und keine
Ergebnisprüfung des noch nicht implementierten Experiments.

Ein stärkerer lokaler Effekt bei ähnlichem zentralem Mittel würde räumliche
Mittelung als mögliche Erklärung stützen. Ein größerer zentraler Kontrast bei
ähnlichem lokalen Effekt würde die Nähe zur festen Messregion hervorheben.
Beides kann gemeinsam auftreten; abweichende Topologieindikatoren müssen einzeln
untersucht werden. Auch ein Nullkontrast ist ein gültiger Befund. Keine dieser
Beobachtungen allein belegt eine neue Phase, ein vorteilhaftes Motiv oder den
Robustheitsvorteil aus der übergeordneten Forschungsfrage.
