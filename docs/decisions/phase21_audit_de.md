# Phase 21: Audit vor Änderungen

27.09.2026. Eingefrorene Basis: `3bb7f71` (Phase 20), bereits nach
`origin/phase19-embedded-graphs` übertragen. Auftrag: Reproduzierbarkeit,
Installation, Lebenszyklus und Release-Vorbereitung; keine neue Physik.

Gelesen: README, Phase-20-Bedienung, ursprünglicher Plan, Abschlussbericht,
Architekturbericht, Kompatibilitätsaudit, Testnachweis, pyproject und Launcher.

## Bereits ausreichend

Eine ExperimentConfig mit Schema 1/2, Preview mit Hashprüfung, explizites START,
ResearchService/ResearchEngine für CLI und GUI, unabhängige Worker,
SQLite-Transaktionen, Kontrollanforderungen, Checkpoints, Source-Archiv,
Konfigurationshash, physische Geometriehashes und JSONL-/CSV-Exporte.
Candidate Explorer liest gespeicherte Spektren/Profile. Historische Studien
besitzen einen lesenden Adapter; deren Kontrollen sind deaktiviert.
Folgeexperimente erhalten verlustfreie Geometrien und Herkunft.
Phase-20-Gesamtsuite: 3.243 bestanden, keine Fehler oder Skips.

## Fehlend für einen überprüfbaren Release

- Durchgängige Anleitung ab frischem Clone/venv; README setzt bisher die venv voraus.
- Festgelegte vollständige Paketumgebung und Build-Werkzeuge.
- Maschinenlesbare Qt-/BLAS-/Compiler-Metadaten außerhalb des schmaleren Laufmanifests.
- Eingefrorene kleine Vergleichsdaten und eigenständiger Prüfbericht.
- Reale Prozesse für CLI-/GUI-Start, Pause/Resume und Folgeexperiment im frischen Clone.
- Dokumentierte Grenzen der Reproduktion sowie Release-Zusammenfassung.
- Sechs alte getrackte `.pyc`-Dateien: aus Git entfernen, damit frische Clones
  keine lokalen Interpreter-Caches enthalten; Python-Quellen unverändert lassen.

## Umsetzung

Nur Prüfskripte, Testdaten, Tests, Installationsdokumentation und Release-Metadaten
ergänzen. Keine Änderung an `src/**/*.py`, ExperimentConfig-Serialisierung oder
Physikdefinitionen vorgesehen. Paketversion `0.1.0` bleibt erhalten: bisher
keine etablierte numerische Release-Tagfolge, lediglich ein historischer
Checkpoint-Tag. Der freizugebende Stand wird durch Commit und Prüfnachweis
identifiziert; kein neuer Tag ohne ausdrücklichen Auftrag.

Referenz: Windows x86-64, CPython 3.14.7, ein BLAS-Thread. Ein kleines reguläres
Gitter mit Clean-Stufe und zwei Disorder-Seeds; Referenz und Wiederaufnahme
vergleichen dieselben exakten Stufen. CLI und tatsächliche Qt-Startaktion
verwenden die bestehenden Worker. Ein separater Clone mit neuer venv und
bereinigtem Prozessumfeld prüft Installation und Gesamtsuite. Das ist ein
isolierter Clone auf demselben Rechner, keine unabhängige Maschine.

Die großen historischen Datenbanken sind nicht Bestandteil von Git und werden
separat am vorhandenen Speicherort lesend geprüft. Im frischen Clone darf die
kleine Prüfung nicht von ihnen abhängen. Keine neuen Forschungsresultate.
