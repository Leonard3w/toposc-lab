# Phase 18 v1: unabhängige endliche Validierung

Protokoll vor neuen physikalischen Ergebnissen, 25.09.2026.
Die Konfiguration und Kohorte werden mit SHA-256 gesichert und zusammen mit dem
Quellarchiv in einer neuen Studien-Datenbank abgelegt. Historische Daten bleiben read-only.

## Auswahl, Hypothesen und Grenzen

Zehn Kandidaten: reguläres 10x10-Gitter; drei historische Kandidaten aus der
vollständig ausgewerteten Experiment-002-Kohorte; sechs Strukturkontrollen.
Historische Regel: bestes Q-Mittel; danach höchstes sauberes Randgewicht unter
Q > Referenz und Randgewicht >= 0.97; dritter Vertreter größtmöglicher
symmetriereduzierter Kantendistanz zu beiden unter Q >= 0.95 * bestes Q.
Gleichstände lexikographisch nach ID. Falls unzulässig/fehlend: expliziter
struktureller Ersatz, ohne historische Ergebnisbehauptung.

Kontrollen: drei Paare aus einem gesäten, begrenzten geometrischen Vorschlagspool.
Min/Max von (a) ausgewogener vertikaler/horizontaler Schnittanbindung,
(b) innerer Gradvarianz und (c) Langbindungsanteil. Bereits benutzte symmetriegleiche
Graphen ausgeschlossen. Auswahl ausschließlich nach Geometrie, niemals neuer Physik.
Alle Deskriptoren und Paarunterschiede speichern. Die Paare verändern mehrere
Eigenschaften und prüfen daher zunächst Assoziationen, keine isolierten Ursachen.
Gezielte Einbau-/Rückbau-Interventionen werden als konkrete Kanten-Edits dokumentiert;
kausale Aussagen erfordern anschließend mehrere unabhängige Ausgangsgraphen.

Hypothesen: stärkere Engstellenanbindung; gleichmäßigere lokale Anbindung und
geringere Anisotropie; Nutzen längerer Bindungen bei der aktuellen festen
Kopplungskonvention. Ein Vergleich verschiedener Kopplungskonventionen ist vertagt.

## Endpunkte und numerische Gültigkeit

Primär: gemittelter Qualitätsunterschied zur regulären Referenz über das nach
Pilot eingefrorene positive W-Raster, gleiches Gewicht je Breite. Ganze Seeds
sind unabhängige Resampling-Einheiten; Breiten desselben Seeds sind abhängig.
Drei historische Kontraste: simultane zweiseitige Intervalle über Bonferroni
(je 98.3333%), zusätzlich deskriptive punktweise 95%-Intervalle. Keine Behauptung
eines Mechanismus anhand dieser ausgewählten Einzelgeometrien.

Sekundär: Q-Verteilungen, Wilson-95%-Intervalle der Erfolgsanteile, gepaarte
Erfolgsdifferenzen und Randwerte. Erfolg bleibt das Phase-17-Zentrumskriterium.
Singuläre Localizer, nicht endliche Werte, Symmetrie-/Solverfehler sind ungültig,
nicht trivial. Nicht-singuläre Skaleninkonsistenz ist ein gültig berechnetes
Nichtbestehen des bestehenden Kriteriums; die einzelnen Indizes bleiben sichtbar.
Nenner: geplant, vorhanden, numerisch gültig, ungültig, fehlend und erfolgreiche
Realisierungen separat. Bedingte Erfolgsquote nur über gültige Ergebnisse;
zusätzlich Worst/Best-Grenzen einschließlich fehlender/ungültiger Ergebnisse.
Paarvergleiche nur über beidseitig gültige gemeinsame Seeds, Anzahl ausweisen.
Kein bestätigender Überlegenheitsanspruch bei fehlenden/ungültigen primären Paaren.

## Pilot und Einfrieren

Pilot-Seeds 180101,180102. W=1.2,3,6,9,12; W=0 genau einmal als saubere Referenz,
keine Scheinvermehrung unabhängiger Beobachtungen. Gleicher Seed erzeugt dasselbe
normierte Onsite-Feld für alle Kandidaten und Breiten.
110 geplante Stufen, maximal 120 Versuche, 3600 s kooperativ, ein BLAS-Thread.
Test-/Engineering-Seeds sind getrennt. Kein unbegrenztes Nachrechnen.

Rasterregel ausschließlich anhand regulärer Pilot-Erfolgsanteile: erste Breite
mit Anteil <= 0.5 und vorhergehende Rasterbreite einklammern, dann fünf linear
verteilte positive Breiten im Intervall einfrieren. Liegt der Übergang unter
der ersten positiven Breite, fünf positive Punkte bis dorthin verwenden.
Wird er nicht eingeklammert oder ist die Referenz ungültig: ursprüngliches Raster
beibehalten und die fehlende Eingrenzung als offene Frage melden. Keine monotone
Glättung und kein W50-Fit. Nichtmonotonie ausdrücklich berichten.

Bestätigung: Seeds 181001-181050, sauberer Referenzlauf separat. 50 Realisierungen
je positiver Breite/Kandidat; keine Pilot- oder historischen Daten poolen.
Nach 50 Seeds keine automatische Erweiterung. Liegt die simultane CI-Halbbreite
des primären Qualitätskontrasts über 0.01, ist die Präzision unzureichend;
eine neue separat budgetierte Studie ist vorzuregistrieren. Kein Stoppen bei
erreichter Signifikanz und kein optionales Nachrechnen bis zum gewünschten Ergebnis.

## Räumliche und Randdiagnostik

Einheiten: Gitterabstand 1, Ursprung (0,0), Fermi-Energie 0, unveränderte kappas.
3x3 Innenproben bei Koordinaten 2,4.5,7; horizontale und vertikale Mittelschnitte
bei 0,1,2,4.5,7,8,9 (vereinigte Proben, Zentrum einmal). Für jeden Ort und jede
Skala Index, Lücke und Invertierbarkeit speichern. Randproben sind keine
zusätzlichen Bestehenskriterien. Karten sind diskrete Proben, keine lückenlose Fläche.

Lokaler Chern-Marker: vorhandene Klasse-D-Implementierung, Nambu-Komponenten
pro Ort aufsummiert, Ortsfläche 1, expliziter innerer Bereich Randabstand >=2.
Volle offene Probe hat kompensierende Randbeiträge. Bulk-Mittel nur deskriptiv,
keine vorausgesetzte Quantisierung oder Mobilitätslücke; Fermi-Nullzustände melden.

Randbereich vorab |E| <=0.5 (Hopping-Einheiten), beide BdG-Vorzeichen. Gleiches
Fenster unter Unordnung, Zahl der Zustände angeben; leeres Fenster ist missing,
nicht Randgewicht 0. Nahe Entartungen innerhalb 1e-8 werden an der Fenstergrenze
vollständig aufgenommen und als Gruppenprojektor ausgewertet. Primär ist das
gemittelte Projektorgewicht des ganzen Fensters, invariant unter Basisrotationen.
Randstreifen Abstand <1,<2,<3, Profil in ganzzahligen Abstandsschalen, mittlerer
Abstand als Eindringtiefen-Deskriptor (kein ungeprüfter Exponentialfit).
Energien, Einzelgewichte, Gruppen und Fensterprojektor speichern.
Localizer-Qualität und Randlokalisierung bleiben getrennt.

## Literatur und Reichweite

- Cerjan/Loring, Spectral-Localizer-Tutorial: https://arxiv.org/abs/2411.03515
- Bianco/Resta, lokaler Chern-Marker: https://arxiv.org/abs/1111.5697

Localizer-Lücke, Bulk-Spektrallücke und Mobilitätslücke sind verschiedene Größen.
Ein hohes Randgewicht ist mit Randzuständen vereinbar, zertifiziert aber weder
Chiralität noch geschützten Transport oder isolierte Majorana-Nullmoden.
