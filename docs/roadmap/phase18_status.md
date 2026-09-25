# Phase 18: unabhängige Fixed-Site-Validierung

Status: **Bestätigung und Forschungsbericht abgeschlossen; STOPP**.
2.510/2.510 Realisierungen abgeschlossen, keine Ausfälle, 67,9 Minuten.
Alle 45 korrigierten mittleren Q-Kontraste sind zugunsten des regulären Gitters;
kein Cross-over im Raster W=6–9. Niedrig-W-Vorteil nicht geprüft.
Integritätsaudit aller Realisierungen bestanden; sechs zusätzliche Statistiktests bestanden.
Ergebnis: [Bestätigungsbericht](../decisions/phase18_confirmation_report_de.md).
Der neuere Auftrag beschränkt die Arbeit auf die unveränderte Bestätigung mit
10 Geometrien, W=6/6.75/7.5/8.25/9 und 50 neuen Seeds je Kombination.
Lauf: `results/phase18-confirmation`, 2500 Unordnungsrealisierungen plus
10 saubere Referenzen; 2560 Versuche und 7200 Sekunden als Obergrenzen.
Der vor Sichtung festgelegte ergänzende Auswertungsplan steht unter
`docs/decisions/phase18_confirmation_analysis_plan.md`.
Nach Forschungsbericht gestoppt. Keine neuen Features, Suche oder Folgestudien.

Die folgenden Angaben zur ersten Etappe dokumentieren den vorherigen Pilotstand.
Auftrag vom 25.09.2026; vollständiger Nutzerplan:
[phase18_user_plan.md](phase18_user_plan.md).

## Autorisierter Umfang dieser Etappe

Bestandsprüfung, versioniertes Protokoll, Implementierung fehlender Diagnostik,
relevante Tests und ein kleiner vollständiger Pilot. Anschließend Auswertung,
eingefrorene Bestätigungskonfiguration und gemessene Budgetabschätzung.
**Die große Bestätigung benötigt ein ausdrücklich festgelegtes Nutzerbudget.**
Kein historischer Lauf wird verändert, fortgesetzt oder neu etikettiert.

## Arbeitspakete

- [x] Auftrag und lokalen Phase-17.2-Quellstand lesen.
- [x] Vorhandene Physik, Randdefinition, Localizer und Persistenz prüfen.
- [x] Historische Evidenz read-only auditieren und Kandidaten einfrieren.
- [x] Kontrollgeometrien nur nach Strukturmerkmalen konstruieren; Confounder ausweisen.
- [x] Zentrumskriterium unverändert lassen; separate räumliche und Randdiagnostik ergänzen.
- [x] Gepaarte Statistik, Fehlernenner, Einzelergebnisse und Berichte implementieren.
- [x] Budget/Checkpoint/Pause/Resume auf bestehender Research-Infrastruktur wiederverwenden.
- [x] Numerische Referenz-, Invarianz-, Fehler- und Wiederaufnahmetests bestehen.
- [x] Beschränkten 100-Orte-Pilot ausführen; Zeit/Operationen messen.
- [x] Kurven, Karten, Randprofile, Zielkonflikte und deutschen Bericht erzeugen.
- [x] Bestätigungsraster nach vorher festgelegter Pilotregel einfrieren, nicht starten.

## Fachliche Entscheidungen vor dem Pilot

- Gleiche 100 Orte und 180 Kanten, Grad 2-6, Länge höchstens 2,
  nur explizit unverbundene Überquerungen fester Orte; historische Modellparameter.
- Unverändertes Zentrumskriterium Q >= 0.20, kappas 0.1/0.2/0.3.
- Zehn Geometrien: Referenz, drei historische Vertreter, sechs Strukturkontrollen.
- Pilot: zwei neue Seeds je W=1.2, 3, 6, 9, 12 plus ein sauberer Lauf je Geometrie.
  Maximal 110 geplante Realisierungen, 120 Versuche einschließlich Unterbrechungen,
  kooperatives Limit 3600 s, ein Worker und ein BLAS-Thread.
- Bestätigung: 50 neue Seeds, unveränderte Kohorte, Raster regelbasiert nach Pilot.
  Keine Signifikanztests oder Überlegenheitsentscheidung aus dem Pilot.
- Räumliche Proben und Projektor-/Randdiagnostik auf jeder Realisierung;
  Aufwand einschließlich Zusatzdiagonalisierungen ausdrücklich zählen.
- Konfidenzintervalle resampeln ganze gemeinsame Seed-Blöcke über alle Breiten;
  bestätigende Vergleichsfamilie sind die drei historischen Kandidaten vs. regulär.

## Zurückgestellt

Größenfamilien, Nachbarparameter, abstandsabhängige Kopplungen, Chiralitäts-/
Transportdiagnostik. Hopping-Unordnung und Kantenausfälle sind getrennte
Folgestudien; Unterstützung im allgemeinen Paket ist kein Research-Adapter-Support.

## Aktuelle Befunde

Der Phase-17-Adapter friert Modellparameter und Zentrumskriterium ein. Das alte
Randgewicht mittelt die vier Zustände mit kleinstem |E| auf boundary_sites
(äußerste Reihe). Local Chern Marker existiert für Klasse D und offene endliche
Geometrien, benötigt expliziten Bulk-Bereich und ist bei E=0-Eigenwerten undefiniert.
Er bleibt eine zusätzliche Diagnose ohne belegte Mobilitätslücke.

## Implementierung und Verifikation

Neue Module `research/validation*.py`; CLI-Anleitung `docs/phase18_validation_usage_de.md`.
Vorhandene ResearchEngine-Stufenabrechnung und Control-Verarbeitung werden geerbt;
SQLite, Lease und Provenienzprüfung wiederverwendet. Kein neuer generischer Suchmotor.
Die allgemeine Search-Engine weist Validierungsstudien ausdrücklich zurück.

Read-only Audit in `results/phase18_historical_audit/`: 254 vollständige Kandidaten,
4.572 Stufen; Kandidaten- und Versuchsdigests identisch zum Phase-17.2-Bericht.
Eingefrorene Kohorte in `examples/phase18_validation_cohort.json`:
Referenz; historische IDs ad56ec, 1bbbaa, 4810c1; sechs rein strukturelle Kontrollen.
Kohortenhash d5c497362eac8414726664a090ed2b91ecf1ddeee706149a047986e411716aea.

21 neue Tests bestanden, einschließlich echtem Prozessabbruch (exit 73),
Resume, unveränderten Primärmetriken, gemeinsamen Unordnungsfeldern,
entartungsinvarianten Projektoren, Fehlernennern und Seed-Block-Statistik.
Ein anfänglich fehlender Testimport wurde behoben. Ruff und scoped Mypy
(Python 3.14, silent imports, fehlende Fremdstubs ignoriert) bestanden.
Pilot `results/phase18-pilot/`: 110/110 Realisierungen, 110 Versuche,
keine numerischen Fehler. 149 s Laufzeit, 127.16 s Exakt-Stufen, davon
114.55 s Zusatzdiagnostik. Erweiterte Regression: **141 Tests bestanden**.
Vollständiger Replay mit unverändertem Quellstand: null neue Exakt-Aufrufe,
Versuch- und Ergebnisdigests identisch. Audit sämtlicher Pilotresultate PASS.

## Abschluss und nächster zulässiger Schritt

- [Deutscher Ergebnisbericht](../decisions/phase18_pilot_report_de.md)
- Zusammenfassung: `docs/decisions/phase18_pilot_summary.json`, Kurven-CSV daneben.
- Abbildungen: `docs/decisions/phase18_figures/` (Chern-Titel im separaten
  Darstellungsskript korrigiert; Pilot-Laufzeitquelle unverändert).
- Verifikation: `docs/decisions/phase18_verification.json`.
- Startbereite Konfiguration: `examples/phase18_validation_confirmation.json`.
- Vorbereitete reversible Interventionen: `examples/phase18_prepared_interventions.json`.

Regelgemäß gewähltes W-Raster: 6,6.75,7.5,8.25,9; neue Seeds 181001-181050.
2.510 geplante Realisierungen. Vorschlag: 2.560 Versuche, 7.200 s Zeitlimit,
ein Worker/BLAS-Thread, ca. 2 GB freie Platte. Erwartet ca. 50-60 min auf
diesem Rechner. **Vorschlag ist kein freigegebenes Budget.** Kein
Bestätigungsordner erstellt, kein großer Lauf gestartet.

Fachlicher Befund: schwacher-W-Vorteil des alten Spitzenkandidaten, bei stärkerer
Unordnung im Pilot schlechter als regulär. Zwei Seeds reichen nicht zur
Bestätigung. Zentrumsvorteil verbessert nicht die schlechteste der neun
Innenlücken. Randstreifenbreite und Zustandsanzahl im Energiefenster sind relevant.
Das Bestätigungsraster untersucht den Übergangsbereich; es repliziert nicht
exakt den alten Niedrig-W-Mittelwert. Diese Reichweite bleibt ausdrücklich offen.

Quellhash Pilot: b18bb71739b0cf864682b55c31b253442352f9f51ef7ba8d1909aa989e34a6fa.
Rohdaten einschließlich SQLite/source.zip liegen Git-ignoriert in results/;
Studienordner separat sichern. Die alte Programmübersichts-PDF wurde nicht
nachträglich umgeschrieben; der neue Bericht ist die aktuelle Phase-18-Ergänzung.
