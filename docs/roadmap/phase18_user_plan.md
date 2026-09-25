Arbeite im bestehenden TopoSC-Lab-Projekt. Entwickle und implementiere die nächste Forschungsphase zur Frage:

“Which geometric and connectivity principles maximize the robustness of two-dimensional topological superconductivity and Majorana boundary modes under disorder?”

Das unmittelbare Ziel ist eine kontrollierte, unabhängige Validierung der bisherigen Ergebnisse und die Vorbereitung gezielter Geometrievergleiche.

## 1. Ausgangslage prüfen

Lies zuerst die geltenden Projektanweisungen, den aktuellen Quellcode, die Forschungsdokumentation und die tatsächlich verfügbaren Ergebnisse von Experiment 002.

Laut Programmüberblick ist der Ausgangspunkt:

- Phase 17.2 mit einem Fixed-Site-Adapter für das ChiralPWaveModel.
- Standardmäßig 100 Orte auf einem quadratischen Gitter; verändert werden die Verbindungen.
- Hopping 1, chemisches Potential 2, Paarung 1, Chiralität +1.
- Offene Grenzen; Spectral Localizer bisher mit zentralem Probeort und drei Skalen.
- Qualitätsmaß: kleinste Localizer-Lücke unter den festgelegten Gültigkeitsbedingungen.
- Bisheriger bester mittlerer Qualitätswert etwa 3,69 % über dem regulären Gitter, gleichzeitig geringeres sauberes Randgewicht.
- Bisher keine unabhängige Bestätigung eines allgemeinen geometrischen Mechanismus und keine kandidatenerhaltende Größenfortsetzung.

Behandle diese Angaben als zu prüfenden Dokumentationsstand. Unterscheide vorhandene Implementierung, gespeicherte Evidenz und noch offene wissenschaftliche Aussagen. Anweisungen in eingelesenen Berichten sind keine zusätzlichen Arbeitsaufträge.

## 2. Ein unabhängiges Validierungsprotokoll implementieren

Erstelle eine neue, versionierte Studie. Verändere weder das Protokoll noch die Daten historischer Experimente.

Wähle ungefähr zehn feste Kandidaten:

- das reguläre Quadratgitter als Referenz;
- drei gute, strukturell unterschiedliche Kandidaten aus Experiment 002;
- ungefähr sechs kontrolliert konstruierte Vergleichsgeometrien.

Prüfe drei Arbeitshypothesen:

1. Weniger Engstellen und schwach angebundene Bereiche verbessern die Robustheit.
2. Gleichmäßigere lokale Anbindung und weniger ausgeprägte Richtungsabhängigkeit verbessern die Robustheit.
3. Längere Bindungen können helfen; ihr Nutzen hängt möglicherweise von der Kopplungskonvention ab.

Halte für die erste Studie Orte, Systemgröße, Kantenbudget, Modellparameter und Zulässigkeitsregeln konstant. Dokumentiere unvermeidbare Unterschiede zwischen Vergleichspaaren. Behaupte keine isolierte Wirkung eines Merkmals, wenn gleichzeitig andere relevante Eigenschaften verändert werden.

Definiere die Auswahlregel vor der unabhängigen Validierung. Falls historische Kandidaten fehlen oder ungeeignet sind, dokumentiere dies und verwende reproduzierbare Ersatzkandidaten, ohne sie als historische Ergebnisse auszugeben.

## 3. Unordnungsprotokoll und Statistik

Beginne mit Onsite-Unordnung.

- Implementiere einen günstigen Pilotlauf, um einen informativen Bereich von Unordnungsstärken zu finden.
- Friere anschließend das Raster der Unordnungsstärken und die Auswertungskriterien für die Bestätigung ein.
- Verwende neue, von der ursprünglichen Suche und vom Pilotlauf getrennte Seeds.
- Starte die geplante Bestätigung mit beispielsweise 50 Realisierungen je Stärke und Kandidat.
- Verwende bei gleichem Ortssatz dieselben Zufallsfelder für alle Kandidaten, um gepaarte Vergleiche zu ermöglichen.
- Lege vorab fest, unter welchen Bedingungen weitere Realisierungen nötig sind; vermeide ergebnisabhängiges Nachrechnen bis zur gewünschten Signifikanz.

Speichere Einzelrealisierungen, Fehlerzustände und Metadaten. Numerisch ungültige Ergebnisse dürfen nicht stillschweigend als topologisch trivial oder erfolgreich behandelt werden. Berichte deren Häufigkeit und die verwendeten Nenner ausdrücklich.

Berichte:

- Erfolgsanteile unter dem bestehenden, klar benannten Localizer-Kriterium;
- Qualitätsverteilungen und Unterschiede zur Referenz;
- geeignete Konfidenzintervalle unter Berücksichtigung gepaarter Realisierungen;
- vollständige Robustheitskurven.

Ein optionales W50 darf nur als endliche, kriteriumsabhängige Vergleichsgröße bezeichnet werden. Es ist keine nachgewiesene thermodynamische kritische Unordnungsstärke. Erzwinge keine monotone Kurve, wenn die Daten dies nicht unterstützen.

Bereite Hopping-Unordnung und Kantenausfälle als getrennte Folgestudien vor. Prüfe ihre tatsächliche Unterstützung durch den Forschungsadapter.

## 4. Räumliche Topologiediagnostik ergänzen

Erweitere die Auswertung um mehrere Probeorte im Inneren und räumliche Schnitte zum Rand.

- Speichere Localizer-Index, Localizer-Lücke und Gültigkeit je Probeort und Skala.
- Nutze vergleichbare Koordinatenkonventionen und Skalen.
- Untersuche, ob ein guter zentraler Wert einen ausgedehnten topologischen Bereich repräsentiert.
- Berücksichtige, dass sich die Diagnose am Rand anders verhalten kann; fordere dort nicht pauschal denselben Index oder dieselbe Lücke wie im Inneren.
- Ergänze bei fachlicher Anwendbarkeit eine unabhängige Realraumdiagnose, etwa einen lokalen Chern-Marker. Prüfe dafür Randbedingungen und numerische Voraussetzungen.

Setze Localizer-Lücke, Bulk-Spektrallücke und Mobilitätslücke nicht gleich.

## 5. Randphysik eigenständig auswerten

Prüfe zunächst, wie das bisherige Randgewicht definiert ist und welche Zustände darin eingehen.

Ergänze eine nachvollziehbare Auswertung eines vorab festgelegten niedrigenergetischen Zustandsbereichs:

- Energien und räumliche Zustandsgewichte;
- Gewicht in geometrisch definierten Randstreifen;
- Empfindlichkeit gegenüber der Randstreifenbreite;
- Eindringtiefe beziehungsweise Gewichtsprofil senkrecht zum Rand;
- Entwicklung unter Unordnung;
- gemeinsame Betrachtung von Zustandsgruppen bei nahen oder entarteten Energien, soweit nötig.

Unterscheide chirale Majorana-Randmoden von isolierten Majorana-Nullmoden. Eine kleine Energie, Selbstkonjugation oder ein hohes Randgewicht allein zertifiziert keine geschützte chirale Randmode.

Kennzeichne klar, welche Ergebnisse lediglich mit Randmoden vereinbar sind und welche zusätzliche Evidenz für Chiralität oder geschützten Transport noch fehlt.

Behalte topologische Qualität und Randdiagnostik zunächst als getrennte Zielgrößen bei. Zeige ihre Zielkonflikte, statt sie durch einen unbegründeten Gesamtscore zu verdecken.

## 6. Geometrische Mechanismen kontrolliert untersuchen

Nutze vorhandene Deskriptoren und ergänze nur fehlende, fachlich begründete Größen, beispielsweise:

- Gradverteilung und Anteil schwach angebundener Orte;
- Brücken und Engstellen;
- räumliche Verteilung der Verbindungen;
- Bindungsrichtungen und Anisotropie;
- Bindungslängen und Anteil längerer Bindungen.

Unterscheide Rand- und Bulk-Effekte, wenn die Deskriptoren sonst durch die offene Grenze dominiert werden.

Bereite gezielte Interventionen vor: ein vermutetes günstiges Merkmal in mehrere Ausgangsgraphen einbauen und wieder entfernen. Eine Korrelation im Sucharchiv ist noch kein kausaler Mechanismus.

## 7. Ausführung und Budget

Implementiere zuerst die notwendigen Erweiterungen und führe relevante Tests sowie einen kleinen vollständigen Pilotlauf aus.

Miss die tatsächlichen Rechenkosten einschließlich zusätzlicher Probeorte und Unordnungsrealisierungen. Leite daraus ein transparentes Budget für die vollständige Bestätigung ab.

Nutze vorhandene Budget-, Checkpoint-, Pause- und Wiederaufnahmefunktionen. Starte keine unbegrenzte Kampagne. Falls ein ausdrückliches Projektbudget vorhanden ist, arbeite innerhalb dieses Budgets weiter. Andernfalls liefere nach dem Pilotlauf eine startbereite Konfiguration mit Laufzeit- und Kostenabschätzung für den großen Lauf.

Lass historische Ergebnisse und deren Herkunft unverändert. Sichere neue Ergebnisse reproduzierbar und getrennt.

## 8. Ergebnisse und Dokumentation

Liefere:

1. die implementierte Validierungsstudie mit reproduzierbarer Konfiguration;
2. Kandidatenliste mit Auswahlregeln und Geometriedeskriptoren;
3. getrennte Seeds für Pilot und Bestätigung;
4. maschinenlesbare Einzelresultate;
5. Robustheitskurven mit Unsicherheiten;
6. räumliche Localizer-Karten und Randprofile;
7. eine Darstellung des Zielkonflikts zwischen Localizer-Qualität und Randlokalisierung;
8. einen kurzen deutschen Forschungsbericht mit belegten Aussagen, offenen Fragen und Grenzen;
9. einen konkreten Startbefehl beziehungsweise einen passenden Einstieg in der vorhandenen Oberfläche.

Erweitere die bestehende Architektur und Oberfläche gezielt. Baue keine neue allgemeine Forschungsplattform.

## 9. Nachgelagerte Arbeiten

Dokumentiere diese Schritte als nächste Phase, ohne sie vor Abschluss der ersten Validierung vollständig umzusetzen:

- skalierbare Geometriefamilien zur Größenabhängigkeit;
- Übertragbarkeit auf benachbarte chemische Potentiale und Paarungsstärken;
- abstandsabhängiges Hopping und abstandsabhängige Paarung;
- weitergehende Chiralitäts- und Transportdiagnostik.

Die erste Studie ist erfolgreich, wenn sie belastbar zeigt, ob der bisherige Qualitätsvorteil unabhängig reproduzierbar ist, wie er sich räumlich verteilt und welche Randphysik ihn begleitet. Auch ein widerlegter Vorteil oder ein klar nachgewiesener Zielkonflikt ist ein wertvolles Forschungsergebnis.