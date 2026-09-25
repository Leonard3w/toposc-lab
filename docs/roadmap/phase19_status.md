# Phase 19 — Bearbeitungsstand

Stand: 25.09.2026. Auftrag: inkrementelle Erweiterung auf eingebettete 2D-Graphen,
kleine Exploration und Bericht, danach stoppen.

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
- Gesamtregression nach der Erweiterung läuft. Die rein geometrische Vorbereitung
  für 50 gültige Exemplare je Zufallsfamilie läuft ohne physikalische Auswahl.
- Wissenschaftlicher Lauf zum Start freigepr?ft: 151 Geometrien, 1510 geplante
  Realisierungen, höchstens 1600 Versuche und 1800 Sekunden Laufbudget.

Maßgeblich: [Protokoll](../decisions/phase19_exploratory_protocol.md),
[Gap-Analyse](../decisions/phase19_architecture_gap_analysis.md),
[Bedienung](../phase19_embedded_usage_de.md).
Nach Abschluss keine Bestätigungsstudie oder nächste Entwicklungsphase automatisch beginnen.

Erster Start: 53 Versuche, davon 52 abgeschlossene Clean-Referenzen; angehalten
wegen Rundungsrest bei einem Randabstand. Siehe
[Vorfall und Korrektur](../decisions/phase19_roundoff_incident.md).
Isolierte Korrektur: 63 relevante Tests bestanden; Geometrien/Seeds unver?ndert.
Ersatzlauf `results/phase19-exploration-v2`: 1510 geplant, verbleibendes
Gesamtbudget 1547 Versuche / 1740 Sekunden.
