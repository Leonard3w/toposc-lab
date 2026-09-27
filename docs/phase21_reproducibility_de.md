# Phase 21: reproduzierbare Installation und technische Prüfung

Die Referenz ist **Windows 11, x86-64, CPython 3.14.7**, Paketversion 0.1.0.
Phase 20 ist durch Commit `3bb7f7185e3e178033b90095e69be67dd0880c15`
eingefroren. Phase 21 ergänzt Prüfungen, keine neue Physik.
Die vollständige Versions-/BLAS-/Qt-Auskunft liefert das Umgebungsskript.

## Frischer Clone und Installation

Voraussetzungen: Git, Windows x86-64, installierter CPython 3.14.7 und
Internetzugang zum Paketindex. Keine bestehende `.venv` übernehmen.
Die folgenden Befehle laufen in PowerShell. Wenn `py -3.14` eine andere
Patchversion auswählt, für die Referenz den Pfad zu Python 3.14.7 verwenden.

```powershell
git clone --branch phase19-embedded-graphs https://github.com/Leonard3w/toposc-lab.git
cd toposc-lab
git checkout bd366072ff94c844bdbf4afec181f1e2da4306bd
py -3.14 -m venv .venv
.venv\Scripts\python.exe --version
.venv\Scripts\python.exe -m pip install --require-hashes --only-binary=:all: -r requirements/phase21-windows-py314.txt
.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e ".[live,app,dev]"
.venv\Scripts\python.exe -m pip check
```

Der Checkout fixiert den geprüften Implementierungsstand. Der abschließende
Dokumentationscommit ergänzt nur Prüfbelege und Anleitungen. Den beweglichen
Branchnamen nicht als Versionsangabe verwenden. Es wurde kein neuer Release-Tag angelegt.

Die erste Installation fixiert sämtliche Laufzeit-, UI-, Test- und Buildpakete
einschließlich pip und setuptools mit Distributionshashes. Die zweite verwendet
weiterhin das vorhandene Projektpaket mit `[live,app]`; `[dev]` ergänzt die
Testwerkzeuge. `--no-build-isolation --no-deps` verhindert, dass dieser Schritt
ungeprüfte neuere Build- oder Laufzeitpakete herunterlädt. Kein neuer Paketmanager
ist nötig. Die bestehende freie Installation `pip install -e ".[live,app]"`
bleibt möglich, entspricht aber nicht automatisch der festgelegten Referenz.

## Studio starten

`start_toposc.bat` doppelklicken oder:

```powershell
.venv\Scripts\python.exe -B -m toposc_live --root results
```

Ein leerer `results`-Ordner ist zulässig. Keine alten Ergebnisse, IDE-Einstellungen
oder globale Pakete sind erforderlich. Öffnen allein startet kein Experiment.
Für einen begrenzten technischen Starttest ohne interaktive Sitzung:

```powershell
.venv\Scripts\python.exe -B scripts/phase21_startup_smoke.py --output output/phase21_startup
```

Dieser Test ruft denselben Anwendungseinstieg auf und schließt das reale
Fenster über einen Testtimer. Er prüft keine interaktive Desktop-Bedienung.

## Reproduktionsprüfung

```powershell
.venv\Scripts\python.exe -B scripts/phase21_environment.py
.venv\Scripts\python.exe -B scripts/phase21_verify.py --output output/phase21_validation/run
```

Das Prüfsystem erzeugt vier kleine Läufe mit je drei exakten Stufen: CLI-Referenz,
unterbrochener/wiederaufgenommener Lauf, GUI-Start und Folgeexperiment.
Je Lauf eine reguläre Geometrie mit 36 Sites; keine Suchkampagne.
Die GUI-Aktionen verwenden den echten Service und getrennte Workerprozesse.
Die Pause wird erst nach einer gespeicherten exakten Stufe angefordert.
Resume startet einen neuen Worker; abgeschlossene Stufen müssen unverändert bleiben.

Erwartet: `status: PASS` und `reproducibility_report.json`, `environment.json`,
vier Laufverzeichnisse mit SQLite, Manifest, Quellarchiv, Konfiguration,
Kandidaten, Checkpoints, JSONL und CSV. Die Referenz enthält auch einen geprüften
PNG-Export. Ein Fehler erzeugt einen FAIL-Bericht und einen Fehler-Exitcode.
Nicht ausgeführte Prüfungen sind `NOT_RUN`, niemals stillschweigend PASS.
Für Wiederholungen ein neues `--output` verwenden; vorhandene Läufe werden
nicht überschrieben. Die ausführliche Ausgabe bleibt zur Inspektion erhalten.

## Dasselbe Experiment von Hand

Im Studio **Load config** und `tests/data/phase21_reproducibility_config.json`
wählen. Der vorgegebene Ausgabeordner muss frei sein. Preview zeigt drei exakte
Stufen und den Konfigurationshash; erst START startet den Worker.
Live Dashboard beobachten; Pause/Resume wirken zwischen exakten Stufen.
Das kleine Experiment kann für eine manuelle Pause zu schnell fertig sein;
der automatisierte Test überwacht dafür die gespeicherten Stufen.
Im Candidate Explorer Spektrum und Ortsprofile öffnen. Für ein Folgeexperiment
frische Seeds `[210101, 210102]` wählen und erneut Preview/START verwenden.

Alternativ derselbe gespeicherte Konfigurationsvertrag per CLI:

```powershell
.venv\Scripts\python.exe -B -m toposc_lab.research preview --config tests/data/phase21_reproducibility_config.json
.venv\Scripts\python.exe -B -m toposc_lab.research create output/phase21_golden --config tests/data/phase21_reproducibility_config.json
.venv\Scripts\python.exe -B -m toposc_lab.research run output/phase21_golden
.venv\Scripts\python.exe -B -m toposc_lab.research inspect output/phase21_golden
```

## Vergleichsregeln und Hashes

Das unveränderte `ExperimentConfig.fingerprint` erfasst die vollständige
Konfiguration, **einschließlich Ausgabeordner**. Die eingecheckte Konfiguration
hat ihren eigenen Golden-Hash. Für getrennte technische Läufe wird ausschließlich
der Ausgabeordner geändert; deren jeweils anderer voller Hash muss zwischen
CLI, GUI, Datei und Datenbank übereinstimmen. Keine neue Normalisierung von Schema 2.
Schema 1 bleibt unverändert und besitzt einen separaten Hash-Regressionstest.

Level A: Kandidaten-ID, physischer Geometriehash, Seeds, Status, gültige Indizes,
Adapterkennung und Struktur werden exakt verglichen. Gespeicherte JSONL-Daten
müssen mit SQLite identisch sein. Archive und Konfigurationen werden mit den
bestehenden Hash-/Checksum-Verfahren geprüft.

Level B: vollständige Eigenwertlisten, Onsite-Potentiale, Q, Localizer-Gaps,
räumliche Proben, Chern-Diagnostik und verfügbare Rand-/Zentrumwerte werden
mit absoluter Toleranz **1e-10**, relativer Toleranz null verglichen. Diese
Toleranz stammt aus der eingefrorenen numerischen Experimentkonfiguration.
Unbekannte/fehlende Werte werden nicht durch null ersetzt. Laufzeit, Zeitstempel
und Experiment-UUID sind keine numerischen Vergleichsgrößen.

Level C: Keine allgemeine bitweise Gleichheit numerischer Ergebnisse zugesichert.
Andere BLAS-/LAPACK-Bibliotheken, Prozessoren oder Betriebssysteme können letzte
Bits ändern. Die Verifikation ist ein Regressionstest gegen Phase 20, keine
unabhängige Lösung der zugrunde liegenden physikalischen Probleme.

Die Quelldateien werden im vorhandenen Manifest byteweise gehasht. Unterschiedliche
Git-Zeilenenden können diesen Hash verändern, auch bei gleicher Logik. Resume
verlangt den exakt passenden archivierten Quellstand und dessen Umgebung.

## Historische Daten und vollständige Suite

Die großen Phase-18/19-Datenbanken sind nicht im Git-Repository enthalten.
Sie müssen separat bereitgestellt werden. Der frische Clone funktioniert ohne sie.
Wenn verfügbar, werden sie ausschließlich lesend geprüft:

```powershell
.venv\Scripts\python.exe -B scripts/phase21_verify.py --output output/phase21_with_history --historical-root C:\Pfad\zu\results
```

Der Bericht nennt Vorher-/Nachher-Hashes, unveränderte Konfigurationshashes,
gespeicherte Kandidaten/Spektren und deaktivierte historische Kontrollen.
Kein historisches Eigensystem wird neu berechnet.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:QT_QPA_PLATFORM='offscreen'
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:BLIS_NUM_THREADS='1'
.venv\Scripts\python.exe -B -m pytest -q
git diff --check
```

## Fehlerdiagnose und Grenzen

Fehlende Qt-/Streamlit-Imports: verwendeten Interpreter und vollständige
Installation prüfen. `pip check` muss erfolgreich sein. `PYTHONPATH` und
`PYTHONHOME` sollten im frischen Clone nicht auf andere Arbeitskopien zeigen.
Ausgabeordner belegt: neuen Ordner wählen. Resume-Hashfehler: archivierten
Quellstand verwenden, keine Prüfsummen manuell ändern. Paket-Hashfehler:
Installation abbrechen und Herkunft/Lockdatei prüfen.

Offscreen-Qt verifiziert die native Oberfläche, ersetzt jedoch keinen manuellen
Grafiktreiber-/Browserlabor-Test. Im isolierten Offscreen-Screenshot fehlen
Schriftglyphen; daraus wird keine korrekte Desktop-Darstellung abgeleitet.
Ein zusätzlicher Starttest mit dem Windows-Qt-Backend rendert die Startseite
mit lesbaren Schriften; der Abschlussbericht enthält den Screenshot.
Windows x86-64 ist die überprüfte Plattform;
Linux/macOS und andere Python-Versionen sind damit nicht freigegeben. Ein neuer
Clone mit eigener venv auf demselben Rechner ist keine unabhängige Maschine.
Phase 21 beansprucht weder neue physikalische Resultate noch eine unabhängige
wissenschaftliche Bestätigung der bisherigen Schlussfolgerungen.
