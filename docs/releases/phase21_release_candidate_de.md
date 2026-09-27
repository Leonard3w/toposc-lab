# TOPOSC Phase 21: vorbereiteter Release-Stand

Paketversion: **0.1.0**, unverändert. Geprüfte Implementierung:
`bd366072ff94c844bdbf4afec181f1e2da4306bd`.
Wissenschaftliche und Anwendungsbasis: Phase 20,
`3bb7f7185e3e178033b90095e69be67dd0880c15`.
Kein neuer Tag und kein veröffentlichtes GitHub-Release.

Phase 21 ergänzt eine mit Paketversionen und Distributionshashes fixierte
Windows-Umgebung, ein Umgebungsskript, begrenzte Start-/Workflowprüfungen,
Golden-Referenzdaten und eine Reproduktionsanleitung. Die Python-Quellen der
Anwendung und alle wissenschaftlichen Definitionen sind unverändert.

Geprüft wurden Windows 11 x86-64 und CPython 3.14.7 mit NumPy 2.5.3,
SciPy 1.18.1, Qt/PySide6 6.11.2 und Streamlit 1.63.0. Die vollständigen
Versionen einschließlich BLAS/LAPACK stehen im
[Umgebungsbericht](../decisions/phase21/environment.json).

Der frische GitHub-Clone mit eigener venv besteht den unterstützten
Installationsweg, den nativen Einstieg und kleine echte CLI-/GUI-Läufe.
Golden-, Resume- und GUI-Daten stimmen innerhalb 1e-10 überein; abgeschlossene
Stufen werden nicht dupliziert. Beide historischen Datenbanken einschließlich
aller 4020 gespeicherten Ergebnisprüfsummen bleiben unverändert.
Folgeexperiment und Seed-Provenienz sind geprüft.

**Tests:** 3254 bestanden, 0 fehlgeschlagen, 0 übersprungen; 1115,39 s.
Ruff für neue Python-Dateien und `git diff --check` bestanden.

Die Prüfung erfolgte auf demselben Rechner. Kein unabhängiger Rechner-/Laborlauf,
keine allgemeine bitweise numerische Garantie und keine Freigabe weiterer
Plattformen. Qt wurde offscreen und zusätzlich mit dem Windows-Backend geprüft.
Die Startseite rendert unter Windows mit lesbaren Schriften; Offscreen zeigt
fehlende Glyphen. Ein vollständiger manueller WebEngine-Test fehlt.
Die historischen Daten sind separate lokale Artefakte, kein Inhalt
des Clones. Es wird kein neuer wissenschaftlicher Befund behauptet.

[Installation und schnelle Prüfung](../phase21_reproducibility_de.md) ·
[vollständiger Abschlussbericht](../roadmap/phase21_report_de.md) ·
[maschinenlesbare Workflow-Prüfung](../decisions/phase21/reproducibility_report.json)

Phase 21 endet mit diesem Prüfstand. Phase 22 wurde nicht begonnen.
