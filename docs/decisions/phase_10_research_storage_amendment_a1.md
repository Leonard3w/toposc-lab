# Speicher-Amendment A1: unveränderliche Generationen-Checkpoints

ID: `TOPOSC-P10-EVO-RS-001-A1`, 2026-09-09.

## Anlass und Evidenz

Der Nutzer hat die Korrektur nach zwei gleichartigen technischen Abbrüchen
beauftragt. Der betroffene Implementierungs-Commit ist
`21e1112c6e56ae73feb56db544df46acbc9fa37b`. Das wissenschaftliche
[Protokoll v1](pre_phase_10_research_protocol_v1.md) unter
`71e159f9eeb67de17b551cb820c4f3599830ca3b` bleibt unverändert.

Im Ordner `results/phase_10_research_v1/preflight/trial_00` brachen
`execution_0000` und `execution_0001` nach Generation 1 beim Ersetzen von
`checkpoint.zip` mit WinError 5 ab. Das Referenzpanel ist versiegelt, beide
Generation-0-Checkpoints sind lesbar. Die jeweils 16 Kandidatenergebnisse
sind gültig gespeichert und stimmen beim kanonischen Replay-Vergleich überein.
Die Fehler stehen im Ereignisjournal am 2026-09-09 um 00:54:46 und 00:56:29 UTC.
Es wurde kein vollständiges Paar und kein Hauptlauf abgeschlossen oder daraus
ein Suchleistungsbefund abgeleitet.

Die geprüften Dateien haben keinen Schreibschutz; der Nutzer hat laut ACL
Vollzugriff. Die Ursache einer eventuellen externen Sperre ist nicht bewiesen.
Windows benötigt für das Ersetzen entsprechende Zugriffs-/Freigaberechte;
siehe [Microsofts MoveFileEx-Dokumentation](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-movefileexa).
Weder Virenschutz noch Dateirechte werden für diesen Fix verändert.

## Begrenzte technische Änderung

Der Forschungsrunner ersetzt keinen Checkpoint mehr. Er veröffentlicht nach
Generation 0 bis 3 jeweils `checkpoint_generation_0000.zip` bis
`checkpoint_generation_0003.zip`. Der vorhandene Checkpoint-Writer bietet dafür
`overwrite=False`: erst validieren, temporär im Zielordner schreiben, fsyncen
und schließen, dann durch einen exklusiven Hardlink veröffentlichen. Die temporäre
Datei wird anschließend entfernt. Vorhandene Ziele werden nie gelöscht oder
überschrieben; bei Veröffentlichungsfehlern bricht der Lauf weiterhin ab.
Hardlinks werden vom bestehenden Kampagnen-Ledger bereits verwendet.

Das Standardverhalten `overwrite=True` der allgemeinen API bleibt unverändert.
ZIP-Schema, wissenschaftliche Inhalte, Hashes und Decoder bleiben kompatibel.
Beim Replay werden alle Generationen-Checkpoints sowie alte `checkpoint.zip`
geprüft; neue Dateinamen müssen zur gespeicherten Generation passen. Das Manifest
vermerkt dieses Amendment und `immutable-generation-files.v1` als Speicherlayout.
Der zusätzliche Platzbedarf entspricht vier erhaltenen Präfixen statt nur einem.

## Fortsetzungsweg ohne Änderung alter Resultate

Der Nutzer committet die getestete Korrektur einschließlich dieses Amendments.
Der neue vollständige Commit wird automatisch im nächsten Manifest festgehalten.
Die strikte Code-Prüfung von `--resume` wird nicht aufgeweicht. Alte Manifeste,
Ereignisse, Checkpoints und Ergebnisse werden weder umgeschrieben noch gelöscht.

Nach dem Commit startet der Nutzer ausschließlich den technischen Vorlauf in
`results/phase_10_research_v1_storage_fix1` erneut. Dies ist eine offengelegte
technische Wiederholung mit den unveränderten reservierten Vorlauf-Seeds
`10_799_900..10_799_909`, keine neue unabhängige wissenschaftliche Replikation.
Die alten Abbrüche bleiben als Entwicklungsnachweis erhalten und werden nicht
in neue Ergebnis-Ledgers importiert. Unterschiedliche Code-Provenienz wird somit
nicht als identischer Replay ausgegeben. Die alten und neuen Vorlaufzeiten
werden nicht zu einer einzelnen erfolgreichen Ausführung zusammengefasst.

Es gibt keine ergebnisabhängige Wahl neuer Seeds, keine Anpassung der Gates,
Schwelle, Fitness, Operatoren, Budgets oder Statistik. Hauptsuch-, Referenz- und
Held-out-Seeds bleiben für ihre vorgesehenen Rollen reserviert; eine neue
wissenschaftliche Seedserie ist für diesen reinen Entwicklungsfix nicht nötig.
Erst nach bestandenem neuen Vorlauf darf der Nutzer `--full` im neuen Ordner
starten. Weitere Unterbrechungen derselben Code-Version werden dort mit
`--resume` behandelt. Alle Befehle stehen in der
[Bedienungsanleitung](../phase_10_usage_de.md#unseren-forschungslauf-starten).

## Verifikation

Regressionen prüfen exklusives Speichern trotz geöffnetem älterem Checkpoint
und erzwungenem Ersetzungsfehler, Nichtüberschreiben vorhandener Ziele,
Bereinigung eigener temporärer Dateien bei Fehlern sowie alte und neue
Checkpoint-Namen, Korruption und falsche Generationsnamen. Die synthetische
Kampagne muss Vorlauf, 32 Hauptpaare und Resume bestehen, auch wenn jeder
Aufruf von `os.replace` einen Fehler auslösen würde. Diese Testdaten sind keine
physikalischen Forschungsergebnisse. Ein echter neuer Vorlauf bleibt Nutzeraktion.

Unter Python 3.14.7 mit deaktiviertem Bytecode bestanden 2.546 vollständige
Repositorytests sowie die fokussierten Speicher-/Kampagnentests. Striktes mypy
bestand für die drei geänderten Quellmodule (`--strict --python-version 3.14
--follow-imports=silent`); die fünf geänderten Python-Dateien sind Ruff-sauber.
Projektweites Ruff behält seine 147 Altbefunde, `git diff --check` ist sauber.

Eine zusätzliche isolierte Speicherprobe lud den vorhandenen großen
Generation-0-Checkpoint nur lesend und veröffentlichte Kopien im eigenen
temporären Verzeichnis. Sowohl normale Ersetzung als auch exklusive
Veröffentlichung bei geöffnetem älterem Checkpoint funktionierten dort.
Der konkrete externe Auslöser der beiden Nutzerabbrüche ist daher weiterhin
offen; der Fix vermeidet den betroffenen Ersetzungsschritt. Es wurden keine
neuen physikalischen Bewertungen für diese Speicherprobe vorgenommen.
