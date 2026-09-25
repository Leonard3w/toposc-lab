# Bestätigung: Auswertungsplan vor Sichtung neuer Ergebnisse

Nutzerfreigabe: bestehender Lauf mit zehn Geometrien, W=6,6.75,7.5,8.25,9,
50 neuen Seeds je Kombination. Budget wie angeboten: 2560 Versuche, 7200 s,
ein Worker/BLAS-Thread. 2500 Unordnungsrealisierungen plus zehn saubere Referenzen.
Keine Änderung an src/, Physik, Score, Kohorte, Seeds oder Suchalgorithmen.
Nach Abschluss auswerten und stoppen; keine weitere Entwicklungsphase.

## Reichweite

Das angeforderte Raster enthält keine schwache Unordnung W=1.2 oder 3.
Reproduzierbarkeit des früheren Niedrig-W-Vorteils ist damit nicht unmittelbar
testbar. Pilot und historische Suche werden nicht in die Bestätigung gepoolt.
Ein Cross-over wird nur innerhalb des tatsächlich bestätigten Rasters geprüft.
Ein negativer Unterschied bei allen W beweist keinen zuvor positiven Unterschied.

## Bestehender primärer Endpunkt

Unverändert: pro Seed gleichgewichteter mittlerer Q-Unterschied über fünf W,
drei historische Geometrien gegen regulär, vorhandene Bonferroni-98.333%-Bootstrap-
Intervalle. Präzisionsgrenze CI-Halbbreite 0.01; keine automatische Erweiterung.

## Zusätzliche vom Nutzer angeforderte Auswertung

Separates nachgelagertes Analyseskript liest gespeicherte Resultate und verändert
die Simulationspipeline nicht. Alle 50 Geometrie/W-Zellen: gültiges n, fehlende /
ungültige Werte, Mittel, Median, Stichproben-SD (ddof=1), SEM, punktweises
95%-Bootstrap-CI, Quantile 0/5/25/50/75/95/100%, Erfolgsanteil mit Wilson-95%-CI.
Direkte Vergleiche über identische Seeds; fehlende Paare explizit ausweisen.

Cross-over-Regel vor Sichtung: für dieselbe Geometrie mindestens ein niedrigeres
W mit statistisch positivem mittlerem Q-Unterschied und ein höheres W mit
statistisch negativem Unterschied. Simultane Bonferroni-t-Intervalle über alle
45 nichttrivialen Geometrie/W-Vergleiche (Familienniveau 95%, df=n-1) als
konservative multiple-Vergleichs-Auswertung; punktweise gepaarte Bootstrap-CIs
zusätzlich. t-Intervalle sind bei n=50 approximativ, nicht verteilungsfrei;
keine Aussage bei fehlenden/ungültigen primären Paaren. Rein numerischer
Vorzeichenwechsel ohne simultanen Nachweis ist ein unsicherer Trend.
Keine Interpolation eines kritischen W, keine monotone Glättung.

Plots vorab festgelegt: alle Robustheitskurven mit CIs, alle gepaarten
Baseline-Differenzen, Verteilungen bei W=6,7.5,9 (Endpunkte und Mitte),
Ranking über das gleichgewichtete Raster ausschließlich deskriptiv.

## Physikalische Gegenprüfung ohne Scoreänderung

Vorhandene gespeicherte Diagnostik verwenden: kleinste Localizer-Lücke der neun
Innenproben über alle Skalen (unabhängig vom Index zusätzlich Indizes ausweisen),
Anteil konsistenter nichtnull Innenproben, Minimum |E| des vollen Spektrums,
Bulk-Mittel des lokalen Chern-Markers, Fensterzustandszahl, Randstreifen d<1/2/3,
mittlere Randdistanz und Majorana-Diagnostik der vier alten Low-Energy-Zustände.
Zentrumsgewicht aus dem gespeicherten Fensterprojektor auf den festen 16 Orten
x,y in {3,4,5,6}; innen = Randabstand >=2. Beide Gebiete vorab festgelegt.
Werte nur für verfügbare gültige Diagnostik, jedes n angeben.

Keine echte Bulk-Spektral- oder Mobilitätslücke liegt vor: Innen-Localizer-Lücke
und Minimum |E| ausdrücklich nicht umbenennen. Unterschiede und Korrelationen
zwischen Q und Diagnostik sind deskriptiv; keine nachträgliche Kausalbehauptung.
Korrelationen je W und Geometrie sowie gepaarte Unterschiede darstellen, keine
scheinbar unabhängigen 2500 Beobachtungen über korrelierte Seed/W-Blöcke poolen.
Hohe Randgewichte oder Selbstkonjugation zertifizieren keine chirale Majorana-Mode.

## Bericht und Stopp

Getrennte Kategorien: robuste Befunde im getesteten Protokoll; unsichere Trends;
numerische Artefakte beziehungsweise mögliche Begrenzung des Scores.
Alle fünf Nutzerfragen beantworten, auch wenn die korrekte Antwort nicht
identifizierbar ist. Rohdaten vollständig getrennt unter results/phase18-confirmation/.
