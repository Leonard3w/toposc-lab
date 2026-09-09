# Rand-/Innenvergleich: geprüfter Befund

## Herkunft und Prüfstatus

Auswertung vom 2026-09-09 für `TOPOSC-P10-EDGE-LOCATION-001`.
Das [Protokoll](pre_phase_10_edge_location_protocol_v1.md) wurde unter
`acf122f9f8cc4e8848f3508766c127b435083f6e` eingefroren; gerechnet wurde mit
`62fab4249d22823cf546c0553aca766a09297619`, Python 3.14.7 und je einem
OMP/OpenBLAS/MKL/BLIS-Thread. Der Nutzer startete Vorlauf und Hauptlauf.

Quellen: [Hauptbericht](../../results/phase_10_edge_location_v1/full/report.md),
[Zusammenfassung](../../results/phase_10_edge_location_v1/full/summary.json),
versiegelte Inputs und Rohresultate in `full/evaluations/cell_0000` bis
`cell_0081` desselben Ergebnisstamms.

- 82/82 Bewertungen verfügbar: 40 Paare und zwei Quadratkontrollen.
- Alle 251 im Abschlussinventar aufgeführten Artefakte stimmen mit ihren Hashes
  überein. Alle 82 versiegelten Zellinputs wurden mit dem Inputplan abgeglichen.
- Alle 410 Chern-Mittelwerte wurden aus den archivierten Markerfeldern und
  Flächen reproduziert; maximale Abweichung in dieser Prüfung 0.
- Zusammenfassung aus den Zellresultaten exakt reproduziert; das lesbare JSON
  stimmt mit der versiegelten Analyse überein.
- Anfangs- und Endkontrolle gültig; maximale Eigenwertdifferenz 0.
- Laufzeit laut Abschlussrecord etwa 174.44 Sekunden einschließlich Aufbau.

SHA-256 des Hauptlaufabschlusses `full/complete.json`:
`b8431ed88d3b0a2f4c410e914237f1ff436652d2a235ddba7bb2fae53946fd1a`.
SHA-256 von `full/summary.json`:
`01f1d96c5f3d8cbaa3dbd951cedce1e4485fb6771d4b36ef33eee155622d2552`.

Die Dokumentation ergänzt die vorhandenen Archive. Die nachstehende Untersuchung
der zusätzlich aufgenommenen Maskenorte ist eine nachträgliche Auswertung der
gespeicherten Marker, keine weitere Hamiltonian- oder Eigenwertrechnung.

## Ergebnis des eingefrorenen Vergleichs

Jede Variante ersetzt genau zwei Kanten eines offenen 12×12-Quadrats.
Knotenanzahl 144, Kantenanzahl 264, Koordinaten und Gradfolge bleiben gleich.
Das Eingriffsmuster wird zwischen Randtiefe 0 und 3 verschoben; fünf tangentiale
Anker sowie Rotationen und Spiegelungen ergeben 40 Paare.

| Größe | boundary | interior |
| --- | ---: | ---: |
| Verfügbare Varianten | 40/40 | 40/40 |
| Zentrale absolute Chern-Abweichung, Median | 0.000129833 | 0.003586314 |
| Zentrale absolute Chern-Abweichung, Minimum | 0.000053007 | 0.002693970 |
| Zentrale absolute Chern-Abweichung, Maximum | 0.000161769 | 0.010543959 |
| Lokale mittlere absolute Markeränderung, Median | 0.149930948 | 0.118567700 |
| Bisheriges SIZE-Screening bestanden | 0/40 | 24/40 |
| L, Median | 0.505852749 | 0.506541562 |

Die lokale Änderung ist der Mittelwert von `abs(marker_variant-marker_square)`
über die vorab festgelegte, mit dem Eingriff verschobene 16-Orte-Messregion.
Sie ist keine Chern-Zahl und hat kein Quantisierungsgate von 0.005.

Der primäre gepaarte Kontrast
`abs(C_interior-1)-abs(C_boundary-1)` ist bei allen 40 Paaren positiv:
Median 0.003441363, Bereich [0.002581044, 0.010388134]. Die gepaarte Differenz
der lokalen absoluten Markeränderungen ist bei allen 40 Paaren negativ:
Median -0.031257017, Bereich [-0.032088025, -0.022570617].
Der Randeingriff verändert das Feld in seiner lokalen Messregion stärker,
während der zentrale Mittelwert näher an 1 liegt.

Die fünf Masken ergeben bei unveränderter Toleranz `abs(C-1)<=0.005`:

| Maske | boundary | interior |
| --- | ---: | ---: |
| graph_depth_2 | 0/40 | 40/40 |
| graph_depth_3 | 0/40 | 24/40 |
| fixed_depth_2 | 40/40 | 40/40 |
| fixed_depth_3 | 40/40 | 24/40 |
| central_fraction | 40/40 | 24/40 |

`central_fraction` und `fixed_depth_3` wählen hier dieselben 36 Orte.
Bei den inneren Eingriffen überschreiten die tangentialen Anker y=3 und y=7
in der zentralen Maske die Toleranz; y=4,5,6 bestehen. Das erklärt 16 bzw.
24 Fälle nach den acht Transformationen. Für denselben Anker unterscheiden
sich die Chern-Mittelwerte über diese Transformationen um höchstens etwa
3e-15. Die 40 Platzierungen bilden daher keine 40 unabhängigen Belege.
Es werden keine p-Werte oder Stichproben-Konfidenzintervalle daraus abgeleitet.

Alle 80 Varianten liefern für alle drei Bott-Perioden und alle drei
Localizer-Kappas den Index +1. PH- und Randzustandsdiagnostik bestehen ebenfalls
bei allen Varianten. Das alte Screening scheitert ausschließlich an den
unaufgelösten Chern-Quantisierungen seiner graphbasierten Masken.

Das Quadrat hat zentral `C=0.9992540277027488` und
`L=0.5065415710994936`. Obwohl alle Randvarianten zentral näher an 1 liegen,
liegt L bei sämtlichen 80 Varianten unter dem Quadratwert. Die Differenzen
der inneren Fälle sind sehr klein und begründen keine Aussage über eine
praktisch relevante Schutzverschlechterung. Ein Schutzvorteil ist in diesen
Proxywerten jedenfalls nicht beobachtet; Disorder-Robustheit wurde nicht gemessen.

## Was die Maskenänderung konkret bewirkt

Für alle 40 Randvarianten gilt: `graph_depth_2` enthält die 64 Orte der festen
Tiefen-2-Maske plus einen weiteren Ort. `graph_depth_3` enthält die 36 Orte
der festen Tiefen-3-Maske plus zwei weitere Orte. Kein Ort der jeweiligen festen
Maske fällt heraus. Die graphbasierte Region verschiebt sich somit durch den
Eingriff, obwohl die Koordinaten unverändert sind.

Beispiel p=2, boundary, y=5, ohne Rotation/Spiegelung:

| Region | Orte | Mittlerer Marker |
| --- | ---: | ---: |
| fixed_depth_2 | 64 | 0.995748362 |
| zusätzlich in graph_depth_2 | 1 | 0.661830251 |
| graph_depth_2 gesamt | 65 | 0.990611161 |
| fixed_depth_3 = central_fraction | 36 | 0.999887074 |
| zusätzlich in graph_depth_3 | 2 | 0.851865749 |
| graph_depth_3 gesamt | 38 | 0.992096478 |

Alle Flächen sind 1. Der Gesamtmittelwert ergibt sich unmittelbar aus der
gewichteten Summe der festen Region und der hinzugekommenen Orte. Diese
Identität wurde für beide Tiefen in allen 40 Randvarianten geprüft;
maximaler numerischer Rest etwa 3.4e-16. Das erklärt rechnerisch, wie dieselbe
Markerkarte je nach Region das Quantisierungskriterium besteht oder verfehlt.
Es entscheidet nicht, welche Region die thermodynamische Physik besser schätzt.

Die eingefrorene Zerlegung bewertet den reinen Auswahleffekt am Referenzfeld.
Bei boundary betragen dessen Mediane etwa -0.000218242 (Tiefe 2) und
-0.000155060 (Tiefe 3); die Mediane des Feldterms betragen etwa -0.006912539
und -0.007011583. Dass der Referenz-Auswahlterm klein ist, macht die Maskenwahl
am veränderten Feld nicht unwichtig: Deren Wechselwirkung liegt im Feldterm.
Die Tabelle oben zeigt genau diesen Unterschied.

## Folgerung und nächster Arbeitsschritt

Der Versuch belegt die Messregionsabhängigkeit der endlichen Diagnostik für
diesen Eingriff. Das zentrale Mittel und die lokale Feldänderung messen
verschiedene Eigenschaften. Die unterschiedliche lokale Umgebung von Rand-
und Innenknoten bleibt Teil des Vergleichs; ein isolierter Abstandseffekt
ohne weitere Umgebungsunterschiede wurde nicht untersucht.

Ein gescheitertes altes Screening bleibt ein gescheitertes altes Screening.
Es ist weder eine nachgewiesene Zerstörung der Phase noch schon ein bewiesener
Fehlalarm des Kriteriums. Umgekehrt beweisen Bott/Localizer +1 und ein günstiger
zentraler Mittelwert hier weder eine neue Familie noch höhere Robustheit.

Der [Entwurf zum Messvertrag](phase_10_measurement_contract_draft_v1.md) hält
deshalb die Rollen von räumlicher Messregion, lokaler Diagnose, Screening und
unabhängiger Validierung fest. Vor einer neuen Suche ist als nächster konkreter
Schritt ein numerisches Validierungsprotokoll für diesen Vertrag auszuarbeiten:
mit positiven und trivialen Kontrollen, mehreren Größen und einer vorab
festgelegten Behandlung widersprüchlicher Methodenresultate. Die aktuellen
80 Varianten dienen seiner Entwicklung, nicht seiner unabhängigen Bestätigung.
