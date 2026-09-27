# Phase 20 — Research Studio: Audit und Umsetzung

Ausgangspunkt: `cf3806a`, 26.09.2026. Auftrag: die vorhandene Anwendung als
einheitliches Forschungswerkzeug ausbauen. Keine neue Forschungskampagne,
keine neue Physik, keine Änderung von Q, keine automatische nächste Phase.

## Audit vor der Umsetzung

**Bereits vorhanden:** `research.config.ExperimentConfig`, `ResearchService`,
`ResearchEngine`, `ResearchStore`, transaktionale Stufen, Kontrollanforderungen,
Checkpoint/Resume, Quellarchiv und Prüfsummen. Registrierte Zufallssuche,
Evolution, MAP-Elites und Surrogat-MAP-Elites. Wiederverwendbare Geometry,
Generatorprotokolle, Geometrieprüfung, exakte p+ip-Auswertung und Diagnostik.
Native Kampagnen-, Gruppen- und Forschungsansichten mit eigenständigen Workern.

**Teilweise vorhanden:** Konfigurationseditor (Formular und JSON), Kandidaten-
und Ergebnisansichten, Laufvorschau nur für ältere Kampagnen. Phase 18/19 nutzt
denselben Store und dieselben Stufen, aber eine feste Kohorte statt Suchobjekten.
Historische Physikadapter frieren Parameter absichtlich ein. Freie Generatoren
unterstützen bislang vorwiegend die festgelegten N=64-Rezepte.

**Fehlend:** gemeinsame Metadaten für Formular und Konfiguration, explizite
Locks, Presets, moderne Vorschau mit überprüfter Konfigurationsidentität,
generische Folgeexperimente aus ausgewählten Kandidaten, ein gemeinsamer
Einstieg und lesende Anpassung historischer Kohorten für den Explorer.

**UI-Entscheidung:** TOPOSC LIVE wird inkrementell erweitert. Es besitzt schon
Prozessentkopplung, native Windows-Steuerung, Monitoring, Kampagnen und Gruppen.
Ein kompletter Wechsel würde funktionierende Infrastruktur ersetzen. Die im
Repository zusätzlich vorhandenen Streamlit-Modell-/Lehrlabore bleiben als
integrierter Werkzeugbereich zugänglich; ihr doppeltes Research-Dashboard wird
nicht als zweite Forschungsoberfläche weitergeführt. `toposc-ui` wird ein Alias
des gemeinsamen Einstiegs. Die Entscheidung wurde im Chat zur Auswahl gestellt;
nach „weiter“ wird die empfohlene Desktop-Integration zugrunde gelegt.

## Architektur und Verträge

1. Die bestehende `ExperimentConfig` erhält Schema 2. Schema 1 wird unverändert
   gelesen und serialisiert; seine Hashes bleiben erhalten. Zusätzliche Studio-
   Metadaten ersetzen keine vorhandenen Physik- oder Laufparameter.
2. Formular, JSON-Datei, Presets, CLI und Vorschau verwenden dieselbe aufgelöste
   Konfiguration. Die Vorschau erzeugt noch keinen Lauf. Erst START übergibt
   genau diese Konfiguration an den vorhandenen Service.
3. Ein neuer versionierter Physikadapter gibt bestehende Parameter frei. Die
   historischen Adapter bleiben eingefroren. Alle Hamiltonians, Disorder-
   Operationen, Eigenwertberechnungen und Diagnostiken stammen aus vorhandenem
   Code. Nicht unterstützte Kombinationen werden abgelehnt.
4. Generatoren erhalten validierte Parameter mit unveränderten alten Defaults.
   Sampling und feste Kandidatenlisten werden als Strategien an die bestehende
   Engine angebunden. Keine zweite Datenbank, kein zweiter Checkpoint-Mechanismus.
5. Locks gehören zum gespeicherten Vertrag. Ein Suchraum kann nur Fähigkeiten
   aktivieren, die seine Strategie tatsächlich unterstützt. Nicht unterstützte
   Physikmutationen, Sparse-Solver und Optimierungsmodi bleiben deaktiviert.
6. Historische Datensätze werden ausschließlich gelesen. Ein Folgeexperiment
   kopiert die verlustfrei gespeicherten Geometrien samt Herkunft in eine neue
   Konfiguration; Auswahl allein startet keine Rechnung.
7. Rohobservablen, numerische Gültigkeit und Optimierungsziel bleiben getrennt.
   Fehlende Werte erscheinen als nicht verfügbar, nicht als Null oder Prognose.

## Umsetzungsschritte und Nachweis

- [x] Repository, beide Oberflächen und Backend-Schnittstellen geprüft.
- [x] Ausgangsprüfung: 95 Tests bestanden, 174,60 s. Befehl:
  `.venv/Scripts/python.exe -m pytest tests/test_research_ui.py tests/test_live_gui.py tests/test_live_configuration.py tests/test_research_engine.py tests/test_embedded_research.py tests/test_validation_study.py -q`
- [x] Schema 2, Presets, Metadaten, Locks und Vorschau.
- [x] Parametrisierbare Generatoren und versionierter Adapter mit Regression.
- [x] Strategien, gemeinsamer Runner, historische Leseadapter und Folgeexperimente.
- [x] Bestehende UI, Explorer, integrierte Labore und Windows-Start.
- [x] Gezielte Tests, kleine echte Integrationsläufe, gesamte Suite und Abschlussbericht:
  **3.243 bestanden**, 977,80 s. Details im
  [Abschlussbericht](phase20_studio_report_de.md) und
  [Prüfnachweis](phase20_test_verification.json).

Die 95 Tests sind die neu ausgeführte Ausgangsprüfung. Die im Phase-19-Bericht
genannten vollständigen Tests sind historische Nachweise, keine erneute Messung.
