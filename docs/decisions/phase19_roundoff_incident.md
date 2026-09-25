# Phase 19: unterbrochener erster Start und Randtoleranz

Der erste wissenschaftliche Start unter `results/phase19-exploration` wurde nach
53 Versuchen automatisch unterbrochen: 52 abgeschlossene Clean-Referenzen und
ein fehlgeschlagener Clean-Versuch. Noch keine Disorder-Realisierungen.
Git-Stand: `ee6f526`; der Lauf und sein Quellarchiv bleiben unverändert erhalten.

Betroffen: `amorphous_planar_a53f1376fcf559e6`, Generator-Seed `4172620195`,
Site 63 bei `[7.000000000000001, 6.999999999999999]`.
Der Abstand zur rechten Boxkante beträgt rechnerisch `-8.881784197001252e-16`.
Die vorhandene affine Normalisierung kann einen solchen Rundungsrest erzeugen.
Die Domain-Prüfung tolerierte bereits Abweichungen bis `1e-10`; die anschließende
Randdiagnostik verlangte dagegen strikt nichtnegative Abstände.

Korrektur: Nur negative Randabstände innerhalb derselben bestehenden Toleranz
werden als null behandelt. Koordinaten, Kanten, Disorder-Seeds, Hamiltonian,
Score und alle Auswahlregeln bleiben unverändert. Tatsächliche Überschreitungen
der Box bleiben Fehler. Ein Regressionstest prüft beide Fälle und bestätigt,
dass die Eingabegeometrie nicht verändert wird.

Dies ist eine technische Inkonsistenz der Toleranzen. Der Fehler entstand nach
den Symmetrieprüfungen in der ergänzenden Randdiagnostik; er ist kein Nachweis
für einen PHS-Fehler, eine neue Phase oder ein Scoreproblem.

Der alte Lauf wird wegen seines eingefrorenen Quellstandes nicht unter geändertem
Code fortgesetzt. Ein neuer, klar getrennter Lauf verwendet exakt dieselbe
Kohorte und dieselben Seeds. Es werden keine Kandidaten entfernt oder ersetzt.
Die 53 verbrauchten Versuche werden vom Gesamtlimit 1600 abgezogen: verbleibend
1547 Versuche bei weiterhin 1510 geplanten Realisierungen. Verbleibendes
Laufzeitbudget: 1740 Sekunden; erster Lauf 59,331 Sekunden. Somit bleibt die
Summe der wissenschaftlichen Ausführungsbudgets unter 1800 Sekunden.

Technische Tests und Berichterstellung sind gesonderte Entwicklungsarbeiten.
Die 52 erfolgreichen alten Referenzen werden nicht zusätzlich als unabhängige
Stichproben in die neue Auswertung aufgenommen.
