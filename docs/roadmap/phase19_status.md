# Phase 19 — Bearbeitungsstand

Stand: 26.09.2026. Auftrag abgeschlossen: inkrementelle Erweiterung auf
eingebettete 2D-Graphen, kleine Exploration und Bericht. Keine Folgephase starten.

- Bestandsprüfung und Gap-Analyse abgeschlossen. Die vorhandene allgemeine
  Graphdarstellung sowie Hamiltonian, Disorder, Score, Diagnostik und Speicher
  werden weiterverwendet.
- Ausgangszustand: 3140 Tests bestanden; alle 2510 Phase-18-Datensätze geprüft;
  drei feste Referenzrechnungen vor und nach der Erweiterung reproduziert.
- Checkpoint: `c666246c830fd99d96c1b44a3aaf037468a5fc07`, Tag
  `phase18-confirmed-before-embedded-graphs`, Arbeitsbranch `phase19-embedded-graphs`.
- Implementiert: expliziter physikalischer Bereich, versionierter Adapter,
  gemeinsamer Constraint-Zugang, outcome-blinde Familienkohorte, Wiederverwendung
  des Validierungslaufs, deskriptiver Rohdatenexport und Visualisierungen.
- Quadratgitter-Gleichheit, BdG-Symmetrie, Pairing-Antisymmetrie und gezielte
  Regression bestanden. Technischer Probelauf: 40/40, null ungültige Ergebnisse.
- Gesamtregression nach der Erweiterung: 3150 Tests bestanden; nach isolierter
  Randtoleranzkorrektur weitere 63 relevante Tests bestanden.
- Wissenschaftlicher Lauf vollständig: 151 Geometrien, 1510 gespeicherte
  Realisierungen, null ungültige Ergebnisse. Einschließlich Erststart 1564 Versuche
  und rund 1069 Sekunden protokollierte aktive Zeit, innerhalb des Gesamtbudgets.

Maßgeblich: [Protokoll](../decisions/phase19_exploratory_protocol.md),
[Gap-Analyse](../decisions/phase19_architecture_gap_analysis.md),
[Bedienung](../phase19_embedded_usage_de.md).
Nach Abschluss keine Bestätigungsstudie oder nächste Entwicklungsphase automatisch beginnen.

Erster Start: 53 Versuche, davon 52 abgeschlossene Clean-Referenzen; angehalten
wegen Rundungsrest bei einem Randabstand. Siehe
[Vorfall und Korrektur](../decisions/phase19_roundoff_incident.md).
Isolierte Korrektur: Geometrien und Seeds unverändert.
Ersatzlauf `results/phase19-exploration-v2`: 1510 vollständig, 1511 Versuche;
eine Sitzungsunterbrechung wiederaufgenommen. Ausgangslauf bleibt erhalten.

Ergebnis: Bei W=3 übertreffen 12/50 umverdrahtete Gitter die reguläre Referenz
im mittleren Score; bei W=6/9 keine der 150 Zufallsgeometrien. Kein bestätigter
Vorteil oder Cross-over bei nur drei Seeds. Freie Familien ohne erfolgreichen
Disorder-Fall; Interior-/Rand-/Chern-Diagnostik überwiegend schwächer.

[Forschungsbericht mit Plots](../decisions/phase19_exploratory_report_de.md),
[Abschlussaudit](../decisions/phase19_exploratory_audit.json).
Die empfohlene unabhängige Studie ist nur beschrieben und wurde nicht gestartet.
