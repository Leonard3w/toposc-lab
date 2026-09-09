# Auswertung der Quadratkalibration TOPOSC-P10-CAL-001

Lesende Prüfung am 2026-09-09 nach Abschluss des Nutzerlaufs. Code und lokales
`origin/main`: `aee26ec825ff0b3055d8a6d8dd9a8cca76ea5106`, Worktree zuvor sauber.
Grundlage: [eingefrorenes Protokoll](pre_phase_10_calibration_protocol_v1.md),
[Hauptlaufbericht](../../results/phase_10_calibration_v1/full/report.md) und
[Zusammenfassung](../../results/phase_10_calibration_v1/full/summary.json).

## Integrität und Reproduktion

Die vollständigen Artefaktinventare des Vorlaufs (155 Dateien) und Hauptlaufs
(1.163 Dateien) stimmen mit den gespeicherten SHA-256-Prüfsummen überein.
Alle 386 versiegelten Hauptlaufzellen wurden mit ihren Eingaben geladen.
Geometrie-IDs, Ergebnisrollen und die Zuordnung zum gespeicherten Konstruktionsplan
stimmen überein. Die Physikläufe tragen `evaluation_seed=None`, sind gültig und
haben keine operative Fehlermeldung. Sämtliche Eigenenergien der beiden
Nullkontrollen stimmen exakt überein; beide Kontrollen bestehen die alten Gates.

Die Chern-Bulkschätzungen wurden aus den bereits gespeicherten vollständigen
Eigenvektoren und Eigenenergien reproduziert. Es gab keine neue Diagonalisierung,
keine neue Geometrie und keine Nutzung zukünftiger Seeds. Die maximale Abweichung
zu den archivierten Chern-Werten beträgt auf diesem Rechner `0.0`.
Dies prüft Archivierung, Zuordnung und Formelreproduktion; es ist keine von der
vorhandenen Formel unabhängige mathematische Validierung des Verfahrens.

SHA-256 der gelesenen Dateien (Hash der gesamten Datei):

| Datei relativ zum Ergebnisstamm | SHA-256 |
| --- | --- |
| `preflight/complete.json` | `d44e6167e1296a7efdbb70854573ea46b2ed1d17b91797b216a6f433683f50c9` |
| `full/complete.json` | `3e0e03cfdfb4d58c2cdbeaa3fba2eddfcee3e0e3b044c526034c807dc95f37bc` |
| `full/construction_plan.json` | `6b0c34fdd6b7d7951928bbeab63e02e8e4240406efe311e4ba48424a14bc13f7` |
| `full/summary.json` | `fdb0a28b44a49d0d05acd4ac3f9942926eedca5815fbe1b629b76a7b15bccbd3` |

## Befund

Von 384 Varianten bestehen 99 alle sauberen Gates. Die zwei Nullkontrollen zählen
separat. Varianten derselben der 16 Trajektorien sind korreliert; aus `99/384`
wird keine Erfolgswahrscheinlichkeit für unabhängige Geometrien geschätzt.

| Bedingung | Varianten, die sie erfüllen |
| --- | ---: |
| Bott: alle drei Gridwerte +1 | 384/384 |
| Localizer: alle drei Gridwerte +1 | 384/384 |
| Schutzproxy `L >= 0.20` | 384/384 |
| Randgate einschließlich PH-Paarung | 369/384 |
| Beide Chern-Bulkwerte innerhalb 0.005 von +1 | 99/384 |

Alle 285 abgelehnten Varianten scheitern am Chern-Quantisierungskriterium.
15 davon scheitern zusätzlich am Vier-aus-acht-Randgewichtskriterium. Sie liegen
bei acht Kantenschritten, jeweils drei Trajektorien pro Amplitude. Das ist keine
Mehrheitsabstimmung zwischen den Methoden: Die eingefrorene Gatekonjunktion gilt.

Die bestehenden Flags `topology_unresolved` und `topology_not_converged` bedeuten
hier, dass mindestens eine finite Chern-Schätzung nicht innerhalb der Toleranz
einer ganzen Zahl liegt. Das Konvergenzflag stammt aus der Gleichheit aufgelöster
Gridindizes. Es belegt keinen eigenständigen Größen-Konvergenztest und meldet
hier auch kein Scheitern des Eigenwertlösers.

Bei reinen Positionsänderungen bestehen für `a = 0.005, 0.025, 0.05, 0.10`
jeweils `16, 11, 8, 5` von 16 Trajektorien. Bei unveränderten Positionen bestehen
nach `k = 1, 2, 4, 8` versuchten Kantentauschen `7, 6, 2, 0` von 16.
Alle 128 protokollierten Austauschschritte wurden ausgeführt; es gab keine No-ops.

Die Änderungen sind nicht überall nur eine knapp verfehlte Toleranz:
33 Varianten haben in mindestens einer Bulkregion `abs(C - 1) > 0.05`.
Bei `a=0, k=8` reicht die Graphabstand-2-Schätzung von etwa 0.745 bis 1.009.
Die Ursache lässt sich deshalb nicht pauschal als Rundungsproblem bezeichnen.

Neun Varianten bestehen alle Gates und überschreiten deskriptiv die alte
Suchschwelle `0.2546216679929183`. Sie stammen aus nur zwei Trajektorien:
`10830004` bei `k=2` und `a=0, 0.005, 0.025, 0.05`, sowie `10830012` bei `k=4`
und allen fünf Amplituden. Der höchste zulässige Proxy ist `0.26193788611284047`.
Diese Fälle wurden in einer anderen, nach dem Suchergebnis geplanten Kalibration
gefunden. Sie ändern den abgeschlossenen Evolution/Zufall-Vergleich nicht und
sind keine neun unabhängigen Entdeckungen oder validierten Robustheitsvorteile.

## Explorative Diagnose der Auswertungsregion

Nach Kenntnis der Resultate wurden zwei zusätzliche Auswertungen ausschließlich
aus den gespeicherten Eigenpaaren vorgenommen. Sie ersetzen keine alten Gates.

Rezept zur Reproduktion für jede Zelle:

1. `load_sealed(cell_directory)` und zugehöriges versiegeltes `input.json` laden.
2. Aus `run.simulation_result` alle Eigenvektoren mit Eigenenergie kleiner null
   zum besetzten Projektor `P` zusammensetzen; `Q = I - P`.
3. `basis_coordinates = tile(geometry.coordinates, (2, 1))` benutzen.
   `b = 4*pi*imag(diag(P X Q Y P))` berechnen; beide Nambu-Komponenten pro
   eindeutiger, lexikographisch geordneter Ortskoordinate aufsummieren.
4. Mit den gespeicherten Flächen und Masken
   `C = sum(b_position[mask]) / sum(area[mask])` reproduzieren. Die gespeicherte
   `unique_coordinate_order` muss zur tatsächlichen Koordinatenordnung passen.
5. Für die Zusatzdiagnose nur die Maske oder nur den Nenner ändern, wie unten.

Erste Zusatzdiagnose: Die ursprünglichen Quadrat-Knoten mit Koordinatenabstand
mindestens 2 bzw. 3 zum Rand bleiben als feste Knotenmengen ausgewählt,
unabhängig von Kantentausch und Verschiebung. Koordinaten, Projektor und
Voronoi-Flächen der jeweiligen Variante bleiben erhalten.

Die Graphmasken unterscheiden sich in 65/384 Varianten von diesen festen Mengen.
Mit den festen Mengen bestehen 101 statt 99 Varianten beide Chern-Kriterien.
Die einzigen Rasterzellen mit veränderten Erfolgszahlen sind `a=0.05,k=1` und
`a=0.10,k=1`, jeweils um eins erhöht. Bei acht Kantenschritten bleiben sämtliche
Amplituden bei null. Maskenänderung allein erklärt den Befund daher nicht.

Zweite Zusatzdiagnose: Für die bisherigen Graphmasken wurde testweise die
Flächensumme im Nenner durch die Knotenzahl ersetzt. Bei reinen Positionsänderungen
bestehen dann `12, 6, 2, 1` statt `16, 11, 8, 5` Trajektorien. Diese abweichende
Normierung liefert keinen allgemeinen Ausweg; sie wird nicht zum neuen Gate.

## Schlussfolgerung und nächster Schritt

Der Chern-Test ist in diesem kleinen Quadrat-Umfeld die entscheidende
Zulässigkeitsgrenze. Ob die Abweichungen bei größerem System und wachsendem
Abstand der Messregion zum Rand abnehmen, ist offen. Auch eine durch Eingriffe
veränderte lokale Physik bleibt als Erklärung möglich.

Der lokale Marker ist räumlich aufgelöst; sein Verhalten in unterschiedlichen
Regionen ist Teil der Diagnose. Siehe
[Bianco/Resta, Mapping topological order in coordinate space](https://arxiv.org/abs/1111.5697).
Der globale Bott-Index und der lokale Localizer beantworten nicht identische
räumliche Fragen. Siehe
[Loring, A Guide to the Bott Index and Localizer Index](https://arxiv.org/abs/1907.11791).
Diese Arbeiten erklären die Motivation, beweisen aber keine Ursache unserer Daten.

Die Aussage, die null Treffer der früheren amorphen Suche seien hauptsächlich
durch diesen Chern-Test erklärt, wäre zu stark: Dort traten zusätzlich andere
Bott-, Schutzproxy- und Randbefunde auf. Die neue Kalibration untersucht einen
anderen Geometriebereich. Eine Übertragung benötigt eigene Evidenz.

Der [folgende Größen-/Methodenplan](pre_phase_10_size_methods_protocol_v1.md)
definiert deshalb frische Ensembles mit ressourcengleichen Referenzen je Größe
und eine feste Auswahl mehrerer Bulkregionen. Ergebnisse werden vor einer
erneuten Suche methodisch ausgewertet. Alle ursprünglichen Ergebnisdateien
bleiben als historische Quelle erhalten.
