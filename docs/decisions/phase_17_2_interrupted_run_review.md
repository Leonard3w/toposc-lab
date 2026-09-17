# Phase 17.2 — Auswertung des abgebrochenen Laufs und gezieltere Suche

Datum: 2026-09-17. Vorheriger Arbeitsstand gesichert in Git: `79d4ee8`.
Experiment 002 bleibt unverändert; kein weiterer wissenschaftlicher Lauf gestartet.

## Ergebnis der Auswertung

Die SQLite-Datenbank von `results/research/experiment002-long-connectivity` enthält
**254 vollständig ausgewertete Kandidaten: 252 gesuchte und zwei Baselines**.
Alle **4.572 verbuchten Exakt-Stufen** haben Status `complete`; keine fehlgeschlagene
oder noch offene Stufe steht im Ledger. Der Abbruch erfolgte somit außerhalb einer
offenen Exakt-Stufe. Die Datenbank steht bei Zyklus 42; die letzte exportierte
`state.json` bei Zyklus 41 und 4.464 Stufen. Ihr `RUNNING` ist veraltet und kein
Beleg für einen laufenden Worker. Der gespeicherte Prozess existierte bei Prüfung
nicht mehr. Der Vorbereitungsbericht Phase 17.1 beschreibt den früheren Zustand
vor dem tatsächlichen Start und ist kein aktueller Laufstatus.

Audit: `results/experiment002_review/audit.json`; Tabelle: `candidates.csv`;
Abbildung: `evidence.png` im selben Verzeichnis. Reproduzierbar mit:

```powershell
.venv/Scripts/python.exe -B scripts/research_audit.py results/research/experiment002-long-connectivity --output results/experiment002_review
```

Geprüft wurden alle Kandidaten-Prüfsummen, bei allen vollständigen Kandidaten die
Geometriebedingungen, sämtliche gespeicherten Deskriptoren durch Neuberechnung,
Zielwert und Validierungsstatus durch erneute Zusammenfassung sowie die Identität
aller Kandidaten-Stufen mit dem separat prüfsummengeschützten Exakt-Journal.
Der Audit benötigt **null neue Exakt-Aufrufe** und schreibt außerhalb des Experiments.
Er ist kein erneutes Diagonalisieren oder vollständiger SQLite-Integritätstest.

## Was physikalisch interessant ist

Alle Werte beziehen sich auf das deklarierte endliche 100-Site-Modell, feste
Kopplungen und dieselben vier Unordnungs-Seeds bei W = 0, 0.4, 0.8, 1.2.
Die Randgewichte sind saubere Low-Energy-Diagnostik, keine Majorana-Zertifikate.
Die Random-rewired-Baseline ist eine einzelne Geometrie aus acht akzeptierten
lokalen Mutationsversuchen; sie repräsentiert nicht die Verteilung des gesamten
erweiterten Suchraums.

| Kandidat (ID-Präfix) | Mittlere Localizer-Qualität | Sauberes Randgewicht | Kleinster Betrag einer Energie | Ersetzte reguläre Kanten |
| --- | ---: | ---: | ---: | ---: |
| Regulär | 0.376689 | 0.997067 | 0.146207 | 0 % |
| Random-rewired-Baseline | 0.373238 | 0.816984 | 0.091136 | 4.44 % |
| Bestes bisheriges Ziel, `ad56ec` | 0.390580 | 0.819853 | 0.016023 | 3.33 % |
| Kompromiss, `e795d4` | 0.387950 | 0.942301 | 0.120136 | 2.78 % |
| Stärkere Randlokalisierung, `7c720e` | 0.385422 | 0.980575 | 0.134945 | 3.89 % |

`ad56ec` verbessert den bisherigen Zielwert gegenüber regulär um **3.687 %**,
senkt aber das mittlere saubere Randgewicht deutlich. Ein kleinerer Betrag einer
Energie beweist keine geschützte Nullmode und ist keine Bulk-Gap-Messung.
`7c720e` ist deshalb ein besonders interessanter zusätzlicher Prüf-Kandidat:
mehr Localizer-Qualität bei weitgehend erhaltener Randlokalisierung.
`e795d4` entstand erst in Generation 41, nach dem letzten Dateiexport; dieser Fund
wäre bei ausschließlichem Lesen des alten Checkpoint-Berichts verloren gegangen.
Seine schlechteste gespeicherte Qualität bei W=1.2 ist 0.386510, gegenüber
0.384219 beim Gewinner des Mittelwert-Ziels. Auch die Rangfolge hängt also von
der physikalischen Frage ab. Der neue Bericht erhält die gesamte Pareto-Liste
aus Zielwert und sauberem Randgewicht, ohne eine neue Erfolgsdefinition einzuführen.

148/252 gesuchte Kandidaten und beide Baselines bestehen alle angefragten
Unordnungstests. Der binäre Erfolgsanteil ist dort gesättigt: aus 100 % gegenüber
100 % folgt keine höhere Unordnungsrobustheit. Vier Seeds liefern wenig Präzision;
die vier W=0-Wiederholungen sind identische störungsfreie Hamiltonians und keine
vier unabhängigen Unordnungsbeobachtungen. Die Bestenauswahl auf denselben Seeds
ist explorativ. Die saubere Wiederholungsrechnung prüft Reproduzierbarkeit,
ersetzt aber keine unabhängige Unordnungsbestätigung.

Der Localizer misst lokale topologische Information und Schutz innerhalb seiner
Voraussetzungen; seine Lücke ist nicht identisch mit einer Bulk-Spektrallücke.
Siehe [Cerjan und Loring, Local invariants identify topology in metals and gapless systems](https://arxiv.org/abs/2112.08623)
und [Tutorial zum Spectral Localizer](https://arxiv.org/abs/2411.03515).
Aus den vorhandenen Center-Probe-Ergebnissen folgen weder eine thermodynamische
Phase noch W_c, räumlich durchgehender topologischer Schutz, chirale Majorana-Moden
oder ein Vorteil realer langreichweitiger Kopplungen mit entfernungsabhängiger Stärke.

## Was die Suche lernen sollte

| Anteil ersetzter regulärer Kanten | Vollständige gesuchte Kandidaten | Mittlerer Zielwert | Bester Zielwert |
| --- | ---: | ---: | ---: |
| [0, 0.05) | 122 | 0.369003 | 0.390580 |
| [0.05, 0.15) | 56 | 0.230945 | 0.388244 |
| [0.15, 0.30) | 21 | 0.082022 | 0.193304 |
| [0.30, 1] | 53 | 0.051025 | 0.141866 |

Exploitation lieferte 81 Treffer über regulär bei 123 Auswertungen, Unsicherheit
4/62, reine Neuheit 0/61, Warmstart 0/6. Insgesamt liegen 85/252 über regulär.
Das sind **keine randomisierten Algorithmusvergleiche**: Kanäle, Generationen und
Abstammungen hängen zusammen. Eine Kausalbehauptung gegen lange Bindungen wäre
unzulässig. Die Daten begründen aber einen kontrollierten Versuch mit häufigerem
lokalem Verfeinern und einem weiterhin vorhandenen Explorationsanteil.

Implementiert:

- `quality_parent_probability` mischt gleichmäßige Archivzellen-Auswahl mit
  Auswahl aus dem nach exaktem Zielwert besten `quality_parent_fraction`-Anteil.
  Defaults 0 bzw. 0.25 erhalten die bisherige Elternpolitik. Nur gültige
  Exakt-Ergebnisse gelangen über das bestehende Archiv in die Zucht.
- `batch_novelty=true` berücksichtigt bereits im aktuellen Batch ausgewählte
  Kandidaten bei der Neuheitsbewertung. Kandidaten desselben noch leeren
  Bereichs verdrängen dadurch seltener andere verfügbare Bereiche.
- Neue Initialisierungsskalen 1 und 2 sind bereits vom Geometriemodul unterstützt
  und werden im neuen Beispiel neben 5, 10, 20, 40 ausdrücklich genutzt.
- Sättigung, Ausbeute der Auswahlkanäle und Zielkonflikte erscheinen in den
  Diagnosen und Forschungsberichten. Fehlgeschlagene, unvollständige und
  vorhergesagte Werte bleiben aus dieser neuen Bewertung ausgeschlossen.

Das Prinzip, verschiedene hochwertige Lösungen zu erhalten, folgt der
Surrogate-Assisted-Illumination-Motivation; die konkreten Gewichte hier sind eine
aus dem Lauf abgeleitete Hypothese, kein belegtes Optimum.
[Gaier et al., Surrogate-Assisted Illumination](https://arxiv.org/abs/1702.03713).

## Laufzeit und Speicher

Gespeichert sind 21.085 s Gesamtzeit und 893 s Exakt-Zeit (etwa 4.24 %).
Die Gesamtzeit ist der letzte persistierte Stand und keine exakt gemessene
Prozesslebensdauer bis zum Abbruch. Daraus folgt keine alleinige Zuordnung aller
übrigen Zeit zu einem bestimmten Engpass. Im Code wurden jedoch konkrete
skalierende Kosten gefunden und beseitigt:

- Exakte Jaccard-Distanzen über alle acht Symmetrien verwenden Integer-Bitmasken
  statt wiederholter Python-Mengenoperationen. Keine Näherung oder Lockerung.
- `seen` speichert kompakte Masken; Geometrie und Provenienz bleiben im
  Kandidatenjournal. Checkpoint-Version 2 liest auch Version 1. Die strenge
  Quellcodeprüfung des Engines bleibt davon unabhängig bestehen.
- Große Majorana-Diagnostikarrays werden nicht in die Zuchthistorie kopiert;
  vollständige Kandidaten- und Stufenresultate bleiben gespeichert.
- Das Zählen von Checkpoints verwendet SQL COUNT statt alle alten Populationen
  zu laden. Die alten 42 Checkpoints hatten rund 1.85 GB JSON-Nutzdaten.
- Checkpoint-Berichte laden eine reduzierte Momentaufnahme; Modelltraining
  verwendet die Exakt-Historie statt sämtliche vorgeschlagenen Kandidaten.
- Dashboard-Momentaufnahmen laden standardmäßig Checkpoint-Zusammenfassungen
  aus der geprüften Berichtshistorie und nur tatsächlich benötigte partielle
  Stufen. Vollständige Recovery-Checkpoints bleiben ausdrücklich abrufbar:
  `ResearchService.snapshot(path, full_checkpoints=True)`.

Direkter Test aus dem letzten gespeicherten Suchzustand, unveränderte
Standardpolitik, acht neue Vorschläge: **17.540 s archiviert / 0.546 s optimiert**,
also **32.1×** in diesem begrenzten Proposal-Test. IDs, Seeds und Mutationsmetadaten
sind identisch. Checkpoint-JSON: **93,149,505 / 26,716,831 Bytes**. Dies ist keine
32×-Behauptung für die gesamte Anwendung; Wiederherstellung und Serialisierung
sind nicht Teil der gemessenen Proposal-Zeit. Null Exakt-Aufrufe.
Artefakt: `results/experiment002_review/performance.json`.
Ein zweiter Durchlauf mit dem versionierten Benchmark-Skript reproduziert
dieselben Vorschläge und Byte-Größen: 14.277 / 0.538 s, entsprechend 26.5×
(`performance-repeat.json`). Die Zeitangaben schwanken mit der Rechnerlast.

Reproduktion gegen das vertrauenswürdige lokale Quellarchiv:

```powershell
.venv/Scripts/python.exe -B scripts/research_search_benchmark.py results/research/experiment002-long-connectivity --output results/experiment002_review/performance-repeat.json
```

## Nächster wissenschaftlicher Schritt

`examples/research_experiment003_focused_pilot.json` ist ein validiertes,
**nicht gestartetes** Pilot-Beispiel: zehn Zyklen, Pool 96, Batch 6, 1.200
Exakt-Versuche, 7.200 s kooperatives Zeitlimit. 60 Kandidaten plus zwei Baselines
kosten beim bisherigen Protokoll 1.116 Stufen und lassen 84 Versuche Reserve.
70/20/10 % Auswahl für Exploitation/Unsicherheit/Neuheit; 70 % qualitätsgerichtete
Elternwahl aus dem besten Archivviertel, 15 % frische Linien, 75/25 % lokale/große
Mutationen. Parameter und physikalisches Protokoll sind explizit gespeichert.

Eine faire Prüfung benötigt einen gleich budgetierten Lauf mit alter Suchpolitik
und denselben Initialisierungen sowie mehrere Such-Seeds. Für eine saubere
Ablation zuerst nur Elternpolitik und Batch-Neuheit ändern; das vollständige
Pilot-Rezept ändert zusätzlich Initialisierung, Mutation und Auswahlgewichte.
Ein Pilot auf den bisherigen Unordnungs-Seeds ist Methodenentwicklung, kein
unabhängiger wissenschaftlicher Bestätigungstest.

Vor einem neuen Überlegenheitsanspruch die drei oben genannten Kandidaten und
beide Baselines einfrieren, dann dieselben 16 **neuen** Unordnungs-Seeds 17401–17416
bei W=0.4, 0.8, 1.2 vergleichen. Saubere Rechnung plus saubere Bestätigung und
48 Unordnungsstufen kosten 50 Stufen pro Geometrie, insgesamt 250 vor Wiederholungen.
W=0 nicht als redundantes Unordnungsensemble zählen. Zunächst die gepaarten
Differenzen je Seed über die Breiten zusammenfassen; Breiten und adaptive
Suchkandidaten nicht als unabhängige Wiederholungen behandeln. Weitere Breiten,
räumliche Localizer-Karten und eine definierte Größenfortsetzung sind getrennte
Prüfungen. Dieser Vorschlag wurde nicht ausgeführt; das Programm verspricht
keine höhere Trefferqualität ohne diesen Vergleich.

## Verifikation

Die abschließenden Prüfergebnisse und der Quellhash stehen im begleitenden
`phase_17_2_verification.json`. Die alten Experimente wurden weder fortgesetzt
noch umkonfiguriert. Zum Fortsetzen von Experiment 002 ist weiterhin sein
archivierter Quellstand erforderlich; die neue Politik gehört in ein neues
Experiment.
