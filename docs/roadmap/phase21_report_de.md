# Phase 21: Reproduzierbarkeit, Release-Freeze und technische Verifikation

Stand: 27.09.2026. Abschlussstatus: **PASS im dokumentierten Prüfumfang**.
Die Prüfung ist eine erneute Ausführung in einem isolierten Clone auf demselben
Windows-Rechner. Sie ist keine unabhängige Untersuchung durch Dritte.

## 1. Ziel

Installation, Start, Experimentsteuerung, Speicherung, Wiederaufnahme und
historische Kompatibilität der eingefrorenen Phase 20 überprüfen. Dazu kommen
eine fixierte Paketumgebung, ein kleiner numerischer Regressionstest und
nachvollziehbare Release-Belege. Keine neue Forschungskampagne.

## 2. Quellbasis

Phase 20 wurde zuerst als `3bb7f7185e3e178033b90095e69be67dd0880c15`
gesichert und auf `origin/phase19-embedded-graphs` gepusht. Vor Änderungen wurden
README, Nutzungsanleitung, Benutzerplan, Abschluss-/Architekturbericht,
Kompatibilitätsaudit, Testbelege, Packaging und Launcher gelesen. Der
[Eingangsaudit](../decisions/phase21_audit_de.md) beschreibt die vorhandene Architektur.

Geprüfte Phase-21-Implementierung:
`bd366072ff94c844bdbf4afec181f1e2da4306bd`. Der abschließende Commit ergänzt
ausschließlich Dokumentation und Prüfbelege. Keine Python-Quelldatei unter
`src/`, keine wissenschaftliche Definition und keine Serialisierung wurde
gegenüber der Phase-20-Basis geändert. Die Paketversion bleibt `0.1.0`.

## 3. Hinzugefügte Dateien

- `docs/decisions/figures/phase21/startup_windows.png`
- `docs/decisions/phase21/acceptance.json`
- `docs/decisions/phase21/artifact_hashes.json`
- `docs/decisions/phase21/clean_setup.json`
- `docs/decisions/phase21/clone_source_and_imports.json`
- `docs/decisions/phase21/environment.json`
- `docs/decisions/phase21/full-tests.log`
- `docs/decisions/phase21/full_test_verification.json`
- `docs/decisions/phase21/historical_compatibility.json`
- `docs/decisions/phase21/reproducibility_report.json`
- `docs/decisions/phase21/startup.json`
- `docs/decisions/phase21/streamlit_smoke.json`
- `docs/decisions/phase21/windows_startup.json`
- `docs/decisions/phase21_audit_de.md`
- `docs/phase21_reproducibility_de.md`
- `docs/releases/phase21_release_candidate_de.md`
- `docs/roadmap/phase21_report_de.md`
- `requirements/phase21-windows-py314.txt`
- `scripts/phase21_environment.py`
- `scripts/phase21_startup_smoke.py`
- `scripts/phase21_verify.py`
- `tests/data/phase21_golden.json`
- `tests/data/phase21_reproducibility_config.json`
- `tests/test_phase21_reproducibility.py`

## 4. Geänderte beziehungsweise aus Git entfernte Dateien

- `.gitignore`
- `README.md`
- `src/toposc_lab/observables/__pycache__/__init__.cpython-314.pyc` (aus Git entfernt)
- `src/toposc_lab/observables/__pycache__/spectrum.cpython-314.pyc` (aus Git entfernt)
- `src/toposc_lab/solvers/__pycache__/__init__.cpython-314.pyc` (aus Git entfernt)
- `src/toposc_lab/solvers/__pycache__/exact_diagonalization.cpython-314.pyc` (aus Git entfernt)
- `src/toposc_lab/visualization/__pycache__/__init__.cpython-314.pyc` (aus Git entfernt)
- `src/toposc_lab/visualization/__pycache__/plots.cpython-314.pyc` (aus Git entfernt)

Die sechs entfernten `.pyc` waren versehentlich versionierte Interpretercaches.
Sie sind keine Quellmodule; lokale Dateien wurden nicht gelöscht. Kein vorhandener
Test wurde entfernt oder geändert. Es gibt keine zweite Engine oder Datenbankimplementierung.

## 5. Prüfung der frischen Umgebung

Ein tatsächlicher Clone von GitHub wurde in einem neuen temporären Verzeichnis
erstellt, ohne bestehende venv, Ergebnisse oder Projektcaches. Eine neue venv
verwendet CPython 3.14.7 ohne globale Site-Packages. Alle Abhängigkeiten wurden
über die Hashdatei neu installiert, anschließend das vorhandene Paket mit
`pip install --no-build-isolation --no-deps -e ".[live,app,dev]"`.
`pip check` war erfolgreich. Die Installationsprüfung begann auf `b088b8d`;
danach wurde ausschließlich die Korrektur des Prüfscripts auf `bd36607` geholt.

`PYTHONPATH`, `PYTHONHOME` und `VIRTUAL_ENV` wurden entfernt, User-Site-Packages
deaktiviert und USERPROFILE/APPDATA/LOCALAPPDATA/TEMP/TMP auf leere
Testverzeichnisse gesetzt. NumPy, SciPy, PySide6 und Streamlit werden aus der
neuen venv importiert, TOPOSC aus dem Clone. Der Systeminterpreter und Windows
stammen weiterhin vom selben Rechner; der System-PATH wurde nicht vollständig
isoliert. Das ist kein Test einer neu installierten Maschine.

Der echte native Einstieg `toposc_live.__main__.main` wurde mit leerem Ergebnisordner
gestartet und durch einen Testtimer beendet. Das ist der unterstützte Einstieg
von `python -B -m toposc_live`; kein Doppelclick-Test der Batchdatei.
Ein echtes ResearchPage-Widget führte Preview und START mit realem Service und
getrenntem Worker aus. Streamlit AppTest konnte das bestehende Labor ohne
Exception öffnen. Alle diese Prüfungen liefen ohne neue Produktionsimplementierung.

Belege: [Installation](../decisions/phase21/clean_setup.json),
[Import-/Quellprüfung](../decisions/phase21/clone_source_and_imports.json),
[nativer Start](../decisions/phase21/startup.json),
[Streamlit](../decisions/phase21/streamlit_smoke.json).

## 6. Reproduzierbarkeitsexperiment

Die kanonische Konfiguration ist `tests/data/phase21_reproducibility_config.json`.
Reguläres Quadratgitter: 36 Sites, 60 Kanten, Box [0;5]²; Koordinaten und
Konnektivität gesperrt. t=1, mu=2, Delta=1, Chirality=1,
Kappas 0,1/0,2/0,3; Clean-Seed 210000 und W=3 mit Seeds 210001/210002.
Die bestehende Schwelle Q>=0,20 und alle Diagnosekonventionen bleiben erhalten.
Ein Kandidat, drei exakte Stufen, keine Optimierung.

Kanonischer Konfigurationshash:
`7deca7b9e84a43d4425aa3e43850b77df514126d10b1dc68064d2273837ce8a8`.
Physischer Geometriehash und Kandidaten-ID:
`d1fc93f97f51464f9bd86fbf20a3b30e5bd620d828fc0f0c1af0dcf08c4eeff4`.

Das Golden-Artefakt wurde einmal mit der unveränderten Phase-20-Physik erzeugt.
Die Referenz, der GUI-Lauf und der wiederaufgenommene Lauf stimmen mit ihm
innerhalb 1e-10 überein. Verglichen werden vollständige Eigenwertlisten,
Onsite-Potentiale, Q, Localizer-Gaps und räumliche Diagnostik, Chern-Daten,
verfügbare Rand-/Zentrumwerte und numerische Gültigkeit. Die Golden-Datei
wird durch den Prüfaufruf niemals automatisch erneuert.

SQLite ist maßgeblich. Manifest, aufgelöste Konfiguration, Quell-ZIP,
Geometriearchive, Kandidaten, Stufenversuche, Checkpoints, JSONL, skalare CSV
und Abschlussbericht wurden dagegen geprüft. PNG-Dateien wurden auf Lesbarkeit
geprüft; dies ist kein Pixelvergleich. Beim Folgeexperiment ist der Plotexport
konfigurationsgemäß deaktiviert, daher dort `plot_verified: false`.

## 7. Pause und Wiederaufnahme

Nach genau einer gespeicherten exakten Stufe wurde Pause über den vorhandenen
Service angefordert. Der CLI-Worker beendete sich an der kooperativen Grenze
mit Status PAUSED. Die GUI-Resume-Aktion startete einen neuen Worker aus dem
persistierten Zustand. Die bereits gespeicherte Stufe blieb unverändert.

Referenz und Resume: jeweils **3 exakte Stufen, 3 vollständige Versuche,
1 Kandidat**, keine doppelte Kandidat/Stufe-Kombination. Referenz: 6 Checkpoints;
Resume: 7 einschließlich Pause. Die numerischen Ergebnisse stimmen überein.
Ein gewaltsamer Prozessabbruch oder Stromausfall wurde damit nicht simuliert.

## 8. CLI/GUI-Identität und Folgeexperiment

Der echte CLI-Preview und der GUI-Preview liefern für dieselbe gespeicherte
JSON-Konfiguration dieselbe aufgelöste Darstellung einschließlich Hash. Beide
verwenden `ResearchService.run_experiment()`, `ResearchEngine` und den bestehenden
Studio-Adapter. Das Prüfsystem besitzt keine eigene Auswertungsimplementierung.

Der Ausgabeordner gehört weiterhin zum Konfigurationshash. Daher erhalten
getrennte Vergleichsläufe verschiedene volle Konfigurationshashes; diese müssen
jeweils zwischen Vorschau, Datei, Manifest und Datenbank übereinstimmen.
Schema 1 und Schema 2 wurden nicht neu normalisiert.

Aus dem gespeicherten Referenzkandidaten wurde ein Folgeexperiment über den
vorhandenen GUI-/Serviceweg erzeugt, vorab geprüft und gestartet. Neue Seeds:
210101 und 210102. Herkunftslauf, Kandidaten-ID, verlustfreies Geometriearchiv,
physischer Hash, angeforderte und realisierte Seeds sowie neuer Konfigurationshash
wurden geprüft. Überlappende Disorder-Seeds wurden abgelehnt.

## 9. Historische Kompatibilität

Phase 18: 2510 gespeicherte exakte Ergebnisse, 10 Kandidaten.
Phase 19: 1510 gespeicherte exakte Ergebnisse, 151 Kandidaten;
1511 historische Versuche einschließlich des bereits dokumentierten Abbruchs.
Alle gespeicherten Payload-Prüfsummen sind gültig.

SHA256 der vollständigen Datenbanken vor/nach der Prüfung und gegenüber Phase 20:

- Phase 18: `c65a9af8a720c380a16fde25679dc43ea8e334426e08099322019ebcffb86826`.
- Phase 19: `caddb37c043c30f4d7a6003610def1aa0fec71931b1edb4a1668b53a247f90e1`.

Die Konfigurationshashes stimmen unverändert mit Phase 20 überein. Je ein
historischer Kandidat wurde in der realen Explorer-Darstellung geöffnet;
Spektren/Profile kamen aus den gespeicherten Daten. Historische Steuerknöpfe
waren deaktiviert. Keine Migration und keine historische Neuberechnung.
Die Prüfung sämtlicher Payloads ersetzt keine visuelle Sichtung jedes Kandidaten.
Beleg: [historische Integrität](../decisions/phase21/historical_compatibility.json).

## 10. Umgebung, Versionen und Hashes

Referenz: Windows 11 Build 26200, AMD64, CPython 3.14.7, MSC v.1944 64 bit.
NumPy 2.5.3, SciPy 1.18.1, Matplotlib 3.11.1, PySide6/Qt 6.11.2,
Streamlit 1.63.0, pytest 9.1.1, pip 26.2.1, setuptools 84.0.0.
Die vollständige Paketliste und NumPy-/SciPy-Buildauskunft mit BLAS/LAPACK
stehen im [Umgebungsbericht](../decisions/phase21/environment.json).
Die fixierte Plattform-/Paketprüfung meldet PASS, ohne Versionsabweichungen.
OMP/OpenBLAS/MKL/BLIS wurden für die Prüfung auf einen Thread begrenzt.

Versioniert sind ExperimentConfig Schema 1/2, Dataset Schema 1 und
Geometriearchiv Schema 1. Adapterkennungen bleiben:
`phase17.fixed-sites-chiral-p-wave.v1`, `phase19.embedded-chiral-p-wave.v1`,
`phase20.configurable-chiral-p-wave.v1`.
Der bestehende Schema-1-Standardhash
`513fe0b990081133e181d4fb558a58d9154de407b6b5193a37c6f123b25204c4`
ist durch einen Regressionstest geschützt. Der Source-Commit ist Git-Metadatum;
das bestehende Manifest enthält zusätzlich den Bytehash der archivierten Quellen.

Im Clone lautet dieser Quellhash
`b0603b3f403652e1a31444cf7d0f0c5ba8dc07998c0310e4f15908a01d802a37`,
in der ursprünglichen Arbeitskopie
`3a95b37b8041eda00fb0be3752da163df8f8460aab1ee163efd3406a855f4885`.
Alle 307 versionierten Python-Quelldateien wurden verglichen: Unterschiede
betreffen ausschließlich CRLF/LF-Zeilenenden. Die bestehende Hashsemantik wurde
bewusst nicht geändert. Resume verlangt weiterhin passende Quellen/Umgebung.

## 11. Tests und technische Korrektur

Vollständige Suite: **3254 bestanden, 0 fehlgeschlagen, 0 übersprungen**;
pytest-Laufzeit **1115,39 s** (18:35), einschließlich Prozessstart 1117,31 s.
Exitcode 0; JUnit-Zähler wurden unabhängig aus der XML-Datei gelesen.
[Testbeleg](../decisions/phase21/full_test_verification.json) und
[vollständige Konsolenausgabe](../decisions/phase21/full-tests.log).

Elf neue Phase-21-Tests ergänzen die 3243 vorhandenen Tests. Sie prüfen echte
CLI-/Qt-Workflows, Metadaten, Golden-Daten, Resume, Exporte, Provenienz,
Hashstabilität, historische Nur-Lese-Adapter, Fehlerberichte und absichtlich
beschädigte Exporte. Die vollständige Suite läuft in der neuen Clone-venv.
Ruff für alle neuen Python-Dateien und `git diff --check`: bestanden.

Ein zusätzlicher Verifikationslauf deckte eine Race-Condition **im neuen
Prüfskript** auf: SQLite existiert bereits kurz vor Anlage seiner Tabellen.
Der Prüfer wartet nun auf das Ende der echten GUI-Erstellungsoperation.
Der betroffene Lauf bleibt als FAIL erhalten; ein begonnener vollständiger
Testlauf wurde dafür nach 275,01 s abgebrochen und anschließend vollständig
neu gestartet. Dieser abgebrochene Lauf wird nicht als bestandene Suite gezählt.
Die Anwendungs-/Physikimplementierung musste nicht geändert werden.

## 12. Numerische Reproduzierbarkeit

- **A, strukturell:** exakte Seeds, Kandidatenidentität, Geometriehash,
  Konfiguration, Indizes, Status und Adapter; Datei-/DB-Integrität nach den
  vorhandenen Hashregeln. Erfüllt im geprüften Workflow.
- **B, numerisch:** absolute Toleranz 1e-10, relative Toleranz 0; Grundlage ist
  die bereits vorhandene numerische Toleranz der eingefrorenen Konfiguration.
  Erfüllt für das kleine Referenzexperiment.
- **C, bitweise:** keine allgemeine Zusicherung für numerische Resultate.
  Andere BLAS/LAPACK-Builds und Prozessoren können letzte Bits verändern.

Zeitstempel, Laufzeiten und Run-UUIDs sind keine numerischen Vergleichsgrößen.
Fehlende Diagnostiken werden nicht in numerische Nullen umgedeutet.

## 13. Bekannte Grenzen

Ein Rechner, ein Betriebssystem und eine Python-/Paketkombination wurden
geprüft. Die venv nutzt denselben installierten Basisinterpreter. Kein
unabhängiges Labor, kein Linux/macOS-Test, kein Airgap-Installationspaket,
kein Last-/Langzeittest. Die historischen Datenbanken müssen separat
bereitgestellt werden und sind nicht Bestandteil des Git-Clones.

Der Offscreen-Screenshot zeigt fehlende Schriftglyphen. Ein zusätzlicher
begrenzter Start mit dem echten Windows-Qt-Backend und einem vom Desktop
verborgenen Fenster rendert die Startseite mit lesbaren Schriften:
[Screenshot](../decisions/figures/phase21/startup_windows.png),
[Startbeleg](../decisions/phase21/windows_startup.json).
Das ersetzt keinen vollständigen interaktiven Grafiktreiber-/Bedienungstest.
Das eingebettete WebEngine wurde nicht vollständig manuell bedient;
Streamlit wurde separat mit AppTest geprüft. PNG-Lesbarkeit beweist keine
visuelle Plotqualität. Der Golden-Vergleich ist bewusst klein und bestätigt
nicht sämtliche physikalischen Parameterbereiche.

## 14. Release-Bereitschaft

Die technischen Abnahmekriterien sind im dokumentierten Windows-Prüfumfang
erfüllt. Der reproduzierbare Quellstand ist für eine versionierte Freigabe
vorbereitet. Unabhängige Maschine und vollständige manuelle Desktop-/WebEngine-
Bedienung sind weiterhin nicht verifiziert; keine weitergehende Freigabe behauptet.

Die vorhandene Paketversion `0.1.0` bleibt erhalten; Phase 21 wird durch den
geprüften Commit und die Hashdatei eindeutig identifiziert. Es wurde kein
Release-Tag erzeugt oder gepusht und kein GitHub-Release veröffentlicht.
[Reproduktionsanleitung](../phase21_reproducibility_de.md) und
[Release-Zusammenfassung](../releases/phase21_release_candidate_de.md) gehören
zum vorbereiteten Stand. Maschine-lesbare Einzelprüfungen stehen unter
`docs/decisions/phase21/`; vollständige temporäre Laufdaten bleiben lokal unter
dem im Installationsbeleg genannten Clone-Pfad erhalten.

## 15. Keine neue physikalische Aussage

Phase 21 introduces no new physical result and does not constitute an independent scientific validation of TOPOSC's physical conclusions.

Q-Zulässigkeit, Erfolgsschwelle 0,20, p+ip-/Nambu-Konvention,
uniformes skalares Onsite-Disorder [-W/2;W/2], offene 2D-Ränder,
Wilson-Intervalle, räumliche Proben, Domain-Toleranz, Voronoi-Verfahren und
sämtliche Localizer-/Chern-/Majorana-/Rand-/Zentrumdefinitionen bleiben unverändert.

## 16. Empfehlung für Phase 22

Zuerst die dokumentierte Verifikation auf einem zweiten Rechner durch einen
anderen Anwender wiederholen. Neue wissenschaftliche Fragestellungen und
Budgets anschließend separat festlegen. **Phase 22 wurde nicht begonnen. STOPP.**
