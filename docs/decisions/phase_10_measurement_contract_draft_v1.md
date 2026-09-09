# Messvertrag nach dem Rand-/Innenvergleich: Entwurf v1

## Status und Zweck

Entwurf vom 2026-09-09, erstellt nach Kenntnis von CAL-001, SIZE-001 und
EDGE-LOCATION-001. Grundlage ist der
[geprüfte Rand-/Innenbefund](phase_10_edge_location_v1_analysis.md).
Dieser Text beschreibt die vorgeschlagene Mess- und Auswertungsarchitektur.
Er ist kein numerisch eingefrorenes Experiment, kein bereits validiertes
Erfolgskriterium und keine Änderung der
[Forschungscharter](pre_phase_9_research_charter.md).

Ziel: Eine spätere Geometriesuche soll physikalisch interpretierbare Kandidaten
finden. Eine Geometrie darf nicht allein dadurch besser bewertet werden, dass
ihr Eingriff die ausgewählte Messregion günstig verschiebt oder umgeht.

## Vorgeschlagener minimaler Vertrag

### 1. Messregion vor der Kandidatenbewertung festlegen

Für Vergleiche im selben festen Quadrat wird die primäre räumliche Region
einmal auf dessen Referenzkoordinaten definiert und als unveränderte Menge
physischer Site-IDs gespeichert. Ihre Größe, Form und Randdistanz werden im
jeweiligen Experiment vorab festgelegt. Veränderungen des Graphabstands dürfen
diese primäre Region nicht nachträglich ändern.

Das ist ein Vorschlag für vergleichbare endliche Messungen, keine Behauptung,
dass eine zentrale Region grundsätzlich die richtige Topologie liefert.
Die Daten zeigen bereits, dass ein Eingriff außerhalb dieser Region im Mittel
schwach sichtbar sein kann. Deshalb gehört die lokale Diagnose gemäß Punkt 2
verbindlich zum Bericht; die zentrale Chern-Nähe allein wird nicht zum Suchziel.

Für Geometrien ohne gemeinsame Site-IDs oder Quadrat-Referenz muss das nächste
Protokoll eine gemeinsame physische Domäne, Randdefinition, Abstandsregel und
Flächengewichtung spezifizieren und vor der Suche validieren. Die fünf alten
Quadratmasken sind kein allgemein anwendbarer Vertrag für beliebige planare
Graphfamilien. Solange dieser Domänenvertrag fehlt, ist ihre Übertragung offen.

### 2. Lokale Wirkung und alternative Masken sichtbar halten

Pro Bewertung werden die vollständige Markerkarte, Ortsreihenfolge,
Flächengewichte und alle vorab vereinbarten Masken gespeichert. Bei bekannten
Eingriffen kommen die betroffenen Orte/Kanten und eine vorab definierte lokale
Messregion hinzu. Markeränderungen beziehen sich auf die passende Referenz.

Graphbasierte Regionen bleiben zusätzliche Diagnosen. Ihr Bericht enthält
hinzugekommene und entfernte Site-IDs sowie ihre Markerbeiträge. Bei abweichenden
Mittelwerten werden Feldänderung, Regionsänderung und deren Wechselwirkung
untersucht. Masken werden nicht anhand der erreichten Quantisierung ausgewählt.
Identische Regionen zählen nur einmal als Evidenz.

### 3. Ergebniszustände und Aussagen unterscheiden

| Ergebnis | Zulässige Aussage im Bericht |
| --- | --- |
| Konstruktion verletzt den Ressourcenvertrag | Geometrisch unzulässig; keine gültige Physikbewertung |
| Solver/Methode nicht verfügbar | Technischer oder Anwendbarkeitsfehler mit Begründung |
| Anwendbare Methoden liefern unterschiedliche Indizes oder eine erforderliche Schätzung ist unaufgelöst | Gemischte bzw. unaufgelöste Topologieevidenz |
| Das eingefrorene Screening ist erfüllt | Screening bestanden unter genau diesem Protokoll |
| Separate Größen-/Disorder-Validierung mit gültigen Kontrollen bestanden | Validierung bestanden innerhalb ihres angegebenen Umfangs |

Ein Ergebnis mit gemischter Evidenz erhält keinen automatisch positiven
Topologiestatus. Die Einordnung benennt genau die widersprechenden Methoden
und Regionen. Auch ein technischer Erfolg des Laufs ist kein physikalischer
Erfolg eines Kandidaten. Fehler bleiben im jeweiligen geplanten Nenner.

### 4. Schutz und Robustheit eigenständig prüfen

L bleibt der vorab konfigurierte Localizer-Schutzproxy; der volle endliche
Spektralabstand bleibt eine andere Größe. Chern-Nähe, L, Randzustandsdiagnostik,
Ressourcen und Disorder-Ergebnisse werden einzeln berichtet. Ein Ranking oder
eine Kombination dieser Größen benötigt eine gesonderte eingefrorene Regel.

Für den aktuellen Eingriff liefern die Daten keinen L-Vorteil gegenüber dem
Quadrat. Eine Verbesserung von `abs(C-1)` darf deshalb nicht als nachgewiesener
Robustheitsgewinn interpretiert werden. Ein solcher Gewinn benötigt die
separate Validierung aus der Charter und passende starke Referenzfamilien.

### 5. Schwellenänderungen als neue Entscheidungen behandeln

Die bisherigen Schwellen und Ergebnislabels von CAL-001, SIZE-001 und
EDGE-LOCATION-001 bleiben Bestandteil ihrer Originalprotokolle. Der Wert 0.005
bleibt in deren Auswertung unverändert. Er wird weder erhöht, um mehr Fälle
zu akzeptieren, noch ungeprüft als universelle Größentoleranz übernommen.

Auch der Wechsel zur festen primären Region ist eine neue methodische
Entscheidung nach Kenntnis der Ergebnisse. Er darf keinen alten Nullbefund
rückwirkend in einen Treffer verwandeln. Jede zukünftige Erfolgsregel erhält
eine neue Protokollversion mit getrennt archiviertem Code- und Protokollhash.

## Welche Prüfung vor der nächsten Suche fehlt

Das nächste zu erarbeitende Artefakt ist ein eigenständiges numerisches
Validierungsprotokoll für den vorgeschlagenen Messvertrag. Es muss vor neuen
Physikrechnungen folgende Punkte konkret festlegen:

1. Ein Kontrollpanel mit in der gewählten endlichen Modellkonfiguration
   nachvollziehbar erwarteter topologischer und trivialer Referenz sowie Fällen,
   bei denen die Diagnose ausdrücklich als unaufgelöst gelten muss. Die Auswahl
   darf nicht allein nach dem neuen Gate selbst als richtig zertifiziert werden.
2. Mehrere vorab gewählte Systemgrößen mit dokumentierter Fortsetzung desselben
   Eingriffsmusters. Randdistanz, Abstand zur Messregion, Defektzahl/-dichte,
   Perioden und Localizer-Kappa müssen einen begründeten Größenvertrag erhalten.
   Die alte Mehrgrößenkampagne mit Einzeltauschtrajektorien ersetzt diese Prüfung
   des neuen Zwei-Kanten-Eingriffs nicht.
3. Primäre räumliche Region, zusätzliche Regionen, lokale Messregion und alle
   Toleranzen einschließlich der Regel für widersprüchliche Methodenresultate.
   Erfolgs- und Fehlentscheidungen an den Kontrollen sind separat auszuwerten.
4. Ein Entwicklungs- und ein davon getrenntes, noch nicht inspiziertes
   Bestätigungspanel. Bei zufälligen Konstruktionen oder Disorder sind frische,
   getrennte Seeds nötig. Rotierte/spiegelverwandte Kopien erhöhen die Zahl
   unabhängiger Bestätigungen nicht. Die heutigen 80 Varianten sind Entwicklungsdaten.
5. Feste Fallzahl, Ressourcenbudget, Vorlaufkriterien, Auswertungsregeln und
   technischer Abbruch-/Resume-Vertrag. Alle geplanten Fälle werden unabhängig
   von günstigen oder ungünstigen Wissenschaftswerten aufbewahrt und berichtet.

Erst dieses Protokoll macht die nächste Rechnung startbar. Nach dessen
Validierung kann ein eigenes Suchprotokoll den Messvertrag übernehmen.
Ein automatischer Suchstart oder eine weitere Softwarephase folgt aus diesem
Dokument nicht.

## Umsetzung und Abnahme dieses Dokuments

Für diesen Entwurf sind keine Änderungen an den produktiven Gates, CLI-Befehlen
oder Datenmodellen erforderlich. Vorhandene Roharchive reichen zur Dokumentation
des Befunds aus. Die vorgeschlagenen Zusatzfelder für spätere Berichte sind
Messregion-ID/Definition, Site- und Flächenzuordnung, Eingriffsüberlappung,
methodenspezifische Verfügbarkeit und der benannte Evidenzstatus.

Ein Commit dieses Dokuments dokumentiert den Vorschlag und seine Herkunft.
Er friert keine noch unbestimmten Modellparameter, Größen, Fallzahlen, Seeds
oder neuen Erfolgsschwellen ein. Diese gehören in das nächste separate
Validierungsprotokoll.
